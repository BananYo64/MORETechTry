import os
import json
import io
import threading
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from hashlib import md5

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import noisereduce as nr
from pydub import AudioSegment
from transformers import pipeline
import librosa
import logging
import vosk
from scipy import signal

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@lru_cache(maxsize=1)
def load_models():
    try:
        vosk_model_ru = vosk.Model("models/vosk-model-small-ru-0.22")
        vosk_model_en = vosk.Model("models/vosk-model-small-en-us-0.15")
        logger.info("Vosk models loaded successfully")
        return vosk_model_ru, vosk_model_en
    except Exception as e:
        logger.error(f"Failed to load Vosk models: {e}")
        return None, None

vosk_model_ru, vosk_model_en = load_models()

try:
    translator_ru_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en", device=-1)
    translator_en_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru", device=-1)
    interviewer = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device=-1,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
except Exception as e:
    logger.warning(f"Transformer models not available: {e}")
    translator_ru_en = translator_en_ru = interviewer = None

class InterviewState(Enum):
    INTRODUCTION = "introduction"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CLOSING = "closing"

@dataclass
class CandidateResponse:
    text: str
    language: str
    duration: float = 0.0
    confidence: float = 0.0
    emotional_tone: str = "neutral"
    logical_coherence: float = 0.0
    speech_rate: float = 0.0
    filler_words: int = 0
    technical_depth: float = 0.0
    pause_count: int = 0
    total_pause_duration: float = 0.0

class VoskRecognizerPool:
    def __init__(self, model_ru, model_en, pool_size=2):
        self.pool_ru = [vosk.KaldiRecognizer(model_ru, 16000) for _ in range(pool_size)]
        self.pool_en = [vosk.KaldiRecognizer(model_en, 16000) for _ in range(pool_size)]
        self.lock_ru = threading.Lock()
        self.lock_en = threading.Lock()
    
    def recognize(self, audio_data: bytes, language: str) -> Tuple[str, float]:
        recognizers = self.pool_ru if language == "ru" else self.pool_en
        lock = self.lock_ru if language == "ru" else self.lock_en
        
        with lock:
            recognizer = recognizers.pop(0)
            try:
                recognizer.AcceptWaveform(audio_data)
                result = json.loads(recognizer.FinalResult())
                text = result.get('text', '').strip()
                confidence = result.get('confidence', 0.0)
            finally:
                recognizers.append(recognizer)
        
        return text, confidence

class PerformanceMonitor:
    def __init__(self):
        self.processing_times = []
        self.recognition_accuracies = []
    
    def add_processing_time(self, time_ms: float):
        self.processing_times.append(time_ms)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def add_recognition_accuracy(self, confidence: float):
        self.recognition_accuracies.append(confidence)
        if len(self.recognition_accuracies) > 100:
            self.recognition_accuracies.pop(0)
    
    def get_stats(self):
        return {
            "avg_processing_time_ms": np.mean(self.processing_times) if self.processing_times else 0,
            "avg_recognition_confidence": np.mean(self.recognition_accuracies) if self.recognition_accuracies else 0,
            "total_processed": len(self.processing_times)
        }

class QuestionCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, text: str, lang: str) -> str:
        return md5(f"{text}_{lang}".encode()).hexdigest()
    
    def get(self, text: str, lang: str) -> Optional[str]:
        key = self.get_key(text, lang)
        return self.cache.get(key)
    
    def put(self, text: str, lang: str, question: str):
        key = self.get_key(text, lang)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = question

def optimize_audio_processing(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    
    b, a = signal.butter(4, 100/(sample_rate/2), btype='highpass')
    audio = signal.filtfilt(b, a, audio)
    
    return audio

def detect_language(text: str) -> str:
    cyrillic_count = sum(1 for char in text if 'а' <= char <= 'я' or 'А' <= char <= 'Я')
    latin_count = sum(1 for char in text if 'a' <= char <= 'z' or 'A' <= char <= 'Z')
    
    total_chars = max(1, len(text))
    cyrillic_ratio = cyrillic_count / total_chars
    latin_ratio = latin_count / total_chars
    
    if cyrillic_ratio > 0.3 and cyrillic_ratio > latin_ratio:
        return "ru"
    elif latin_ratio > 0.3:
        return "en"
    return "unknown"

def preprocess_text_for_llm(text: str, lang: str) -> str:
    words = text.split()
    cleaned_words = []
    prev_word = ""
    
    for word in words:
        if word != prev_word:
            cleaned_words.append(word)
        prev_word = word
    
    cleaned_text = " ".join(cleaned_words)
    
    corrections = {
        'ru': {'шт': 'что', 'када': 'когда', 'чё': 'что'},
        'en': {'u': 'you', 'r': 'are', 'btw': 'by the way'}
    }
    
    for wrong, correct in corrections.get(lang, {}).items():
        cleaned_text = cleaned_text.replace(wrong, correct)
    
    return cleaned_text

class InterviewAI:
    def __init__(self, use_llm: bool = True):
        self.current_state = InterviewState.INTRODUCTION
        self.conversation_history = []
        self.use_llm = use_llm
        self.vosk_pool = VoskRecognizerPool(vosk_model_ru, vosk_model_en)
        self.monitor = PerformanceMonitor()
        self.question_cache = QuestionCache()
        
        self.fillers = {'ru': {"ну", "ээ", "мм", "типа", "как бы", "вот", "значит"}, 
                       'en': {"um", "uh", "like", "you know", "well", "actually"}}
        
        self.technical_terms = {'ru': {'алгоритм', 'код', 'база', 'данных', 'фреймворк', 'api', 'сервер'}, 
                               'en': {'algorithm', 'code', 'database', 'framework', 'api', 'server', 'client'}}
        
        self.connectors = {'ru': ["поэтому", "однако", "следовательно", "таким образом"], 
                          'en': ["therefore", "however", "moreover", "consequently"]}
        
        self.fallback_questions = {
            'en': "Tell me more about that.",
            'ru': "Расскажите подробнее об этом."
        }

    async def process_audio(self, audio_bytes: bytes, content_type: str) -> List[CandidateResponse]:
        try:
            if content_type == 'audio/mpeg':
                format = "mp3"
            elif content_type == 'audio/wav':
                format = "wav"
            elif content_type == 'audio/ogg':
                format = "ogg"
            else:
                raise HTTPException(400, f"Unsupported audio format: {content_type}")
            
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
            audio = audio.set_channels(1).set_frame_rate(16000)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            
            samples = nr.reduce_noise(y=samples, sr=16000, prop_decrease=0.7)
            samples = optimize_audio_processing(samples, 16000)
            
            response = self.analyze_segment(samples, 16000)
            if response:
                self._process_response(response)
                return [response]
            return []
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

    def analyze_segment(self, audio: np.ndarray, sample_rate: int) -> Optional[CandidateResponse]:
        try:
            text_ru, confidence_ru = self._recognize_with_vosk(audio, sample_rate, "ru")
            text_en, confidence_en = self._recognize_with_vosk(audio, sample_rate, "en")
            
            detected_lang = detect_language(text_ru if confidence_ru > confidence_en else text_en)
            
            if detected_lang == "ru" and text_ru.strip():
                text, language, confidence = text_ru, "ru", confidence_ru
            elif detected_lang == "en" and text_en.strip():
                text, language, confidence = text_en, "en", confidence_en
            elif confidence_ru > confidence_en and text_ru.strip():
                text, language, confidence = text_ru, "ru", confidence_ru
            elif text_en.strip():
                text, language, confidence = text_en, "en", confidence_en
            else:
                return None
            
            self.monitor.add_recognition_accuracy(confidence)
            
            duration = len(audio) / sample_rate
            pauses, total_pause = self._detect_pauses(audio, sample_rate)
            
            effective_duration = max(0.1, duration - total_pause)
            word_count = len(text.split())
            speech_rate = word_count / effective_duration * 60 if effective_duration > 0 else 0
            
            text_lower = text.lower()
            filler_words = self._count_filler_words(text_lower, language)
            emotion = self._detect_emotion(text_lower, language)
            coherence = self._calculate_coherence(text_lower, language, word_count)
            tech_depth = self._calculate_technical_depth(text_lower, language)
            
            return CandidateResponse(
                text=text, 
                language=language, 
                duration=duration, 
                confidence=confidence,
                emotional_tone=emotion, 
                logical_coherence=coherence, 
                speech_rate=speech_rate,
                filler_words=filler_words, 
                technical_depth=tech_depth,
                pause_count=len(pauses), 
                total_pause_duration=total_pause
            )
            
        except Exception as e:
            logger.error(f"Error analyzing segment: {e}")
            return None

    def _recognize_with_vosk(self, audio: np.ndarray, sample_rate: int, language: str) -> Tuple[str, float]:
        try:
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            return self.vosk_pool.recognize(audio_int16.tobytes(), language)
        except Exception as e:
            logger.error(f"Vosk recognition error ({language}): {e}")
            return "", 0.0

    def _detect_pauses(self, audio: np.ndarray, sample_rate: int) -> Tuple[List[float], float]:
        try:
            energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            
            max_energy = np.max(energy)
            silence_threshold = max_energy * 0.05 if max_energy > 0 else 0.01
            
            min_pause_duration = 0.3
            pauses = []
            in_pause = False
            pause_start = 0
            hop_duration = 512 / sample_rate
            
            for i, e in enumerate(energy):
                if e < silence_threshold and not in_pause:
                    in_pause = True
                    pause_start = i * hop_duration
                elif e >= silence_threshold and in_pause:
                    in_pause = False
                    pause_duration = (i * hop_duration) - pause_start
                    if pause_duration >= min_pause_duration:
                        pauses.append(pause_duration)
            
            if in_pause:
                pause_duration = (len(energy) * hop_duration) - pause_start
                if pause_duration >= min_pause_duration:
                    pauses.append(pause_duration)
            
            return pauses, sum(pauses)
            
        except Exception as e:
            logger.error(f"Pause detection error: {e}")
            return [], 0.0

    def _count_filler_words(self, text_lower: str, language: str) -> int:
        words = text_lower.split()
        filler_set = self.fillers.get(language, set())
        return sum(1 for word in words if word in filler_set)

    def _detect_emotion(self, text_lower: str, language: str) -> str:
        if language == 'ru':
            positive_words = {'хорошо', 'отлично', 'успешно', 'нравится', 'интересно', 'рад'}
            negative_words = {'плохо', 'сложно', 'трудно', 'проблема', 'ошибка', 'стресс'}
        else:
            positive_words = {'good', 'great', 'excellent', 'happy', 'like', 'interesting'}
            negative_words = {'bad', 'difficult', 'hard', 'problem', 'error', 'stress'}
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def _calculate_coherence(self, text_lower: str, language: str, word_count: int) -> float:
        connectors = self.connectors.get(language, [])
        connector_count = sum(1 for connector in connectors if connector in text_lower)
        
        if word_count > 0:
            return min(1.0, connector_count / (word_count / 10))
        return 0.5

    def _calculate_technical_depth(self, text_lower: str, language: str) -> float:
        terms = self.technical_terms.get(language, set())
        found_terms = sum(1 for term in terms if term in text_lower)
        return min(1.0, found_terms / 3.0)

    def _process_response(self, response: CandidateResponse):
        self.conversation_history.append({
            'role': 'candidate',
            'text': response.text,
            'analysis': {
                'technical_depth': response.technical_depth,
                'emotional_tone': response.emotional_tone,
                'language': response.language
            }
        })
        self._update_state(response.technical_depth)

    def generate_next_question(self, text: str, lang: str) -> str:
        cached_question = self.question_cache.get(text, lang)
        if cached_question:
            return cached_question
        
        if not self.use_llm or not interviewer:
            return self.fallback_questions.get(lang, "Tell me more.")
        
        try:
            cleaned_text = preprocess_text_for_llm(text, lang)
            
            if lang == 'ru':
                en_text = translator_ru_en(cleaned_text)[0]['translation_text']
            else:
                en_text = cleaned_text
            
            context = " ".join([msg['text'] for msg in self.conversation_history[-3:] if msg['role'] == 'candidate'])
            
            prompt = f"Context: {context}\nCandidate response: '{en_text}'\nGenerate concise follow-up question:"
            
            generated = interviewer(prompt, max_length=120)[0]["generated_text"]
            question_en = generated.replace(prompt, "").strip().split('\n')[0]
            
            if lang == 'ru':
                question = translator_en_ru(question_en)[0]['translation_text']
            else:
                question = question_en
            
            self.question_cache.put(text, lang, question)
            self.conversation_history.append({'role': 'interviewer', 'text': question})
            
            return question
            
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            return self.fallback_questions.get(lang, "Tell me more.")

    def _update_state(self, tech_depth: float):
        response_count = len([m for m in self.conversation_history if m['role'] == 'candidate'])
        
        if self.current_state == InterviewState.INTRODUCTION:
            if tech_depth > 0.3 or response_count >= 2:
                self.current_state = InterviewState.TECHNICAL
        elif self.current_state == InterviewState.TECHNICAL:
            if tech_depth < 0.2 and response_count >= 4:
                self.current_state = InterviewState.BEHAVIORAL
        elif self.current_state == InterviewState.BEHAVIORAL:
            if response_count >= 6:
                self.current_state = InterviewState.CLOSING

interview_ai = InterviewAI(use_llm=True)

@app.post("/api/receive_audio")
async def receive_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(400, "Only audio files are supported")
    
    if vosk_model_ru is None or vosk_model_en is None:
        raise HTTPException(500, "Speech recognition models not available")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")
    
    try:
        responses = await interview_ai.process_audio(contents, file.content_type)
        if not responses:
            raise HTTPException(400, "No speech detected in audio")
        
        response = responses[0]
        next_question = interview_ai.generate_next_question(response.text, response.language)
        
        return JSONResponse({
            "recognized_text": response.text,
            "language": response.language,
            "next_question": next_question,
            "analysis": {
                "confidence": round(response.confidence, 2),
                "emotional_tone": response.emotional_tone,
                "speech_rate": round(response.speech_rate, 1),
                "pause_count": response.pause_count,
                "total_pause_duration": round(response.total_pause_duration, 2),
                "technical_depth": round(response.technical_depth, 2),
                "filler_words": response.filler_words,
                "logical_coherence": round(response.logical_coherence, 2),
                "duration": round(response.duration, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio")

@app.get("/api/status")
async def get_status():
    stats = interview_ai.monitor.get_stats()
    return JSONResponse({
        "status": "running",
        "offline_mode": True,
        "current_state": interview_ai.current_state.value,
        "conversation_length": len(interview_ai.conversation_history),
        "performance_stats": stats,
        "models_loaded": {
            "vosk_russian": vosk_model_ru is not None,
            "vosk_english": vosk_model_en is not None,
            "translation": translator_ru_en is not None,
            "llm": interviewer is not None
        }
    })

@app.post("/api/reset")
async def reset_interview():
    interview_ai.conversation_history = []
    interview_ai.current_state = InterviewState.INTRODUCTION
    interview_ai.question_cache = QuestionCache()
    return JSONResponse({"status": "reset", "message": "Interview state reset"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)