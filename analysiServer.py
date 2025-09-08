import os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import requests
import transformers
import sentencepiece

app = FastAPI()
UPLOAD_DIR = "/tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Переводчики
analyzer = transformers.pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device_map="cpu"
)

def preprocess_text(text):
    # Удаление лишних пробелов и переносов строк
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Удаление специальных символов (опционально)
    text = re.sub(r'[^\w\s.,!?а-яА-Яa-zA-Z]', '', text)
    
    return text

def analyze_dialogue_parts(dialogue, candidate_labels, speaker_labels=None):
    if speaker_labels is None:
        speaker_labels = {
            'candidate': ['кандидат', 'соискатель', 'candidate', 'applicant'],
            'hr': ['hr', 'рекрутер', 'интервьюер', 'recruiter', 'interviewer']
        }
    
    # Разделение диалога на реплики
    lines = dialogue.split('\n')
    candidate_text = []
    hr_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Определение говорящего
        speaker = identify_speaker(line, speaker_labels)
        
        if speaker == 'candidate':
            candidate_text.append(clean_replica(line, speaker_labels['candidate']))
        elif speaker == 'hr':
            hr_text.append(clean_replica(line, speaker_labels['hr']))
    
    # Анализ реплик кандидата
    candidate_analysis = None
    if candidate_text:
        candidate_full_text = ' '.join(candidate_text)
        candidate_analysis = analyzer(candidate_full_text, candidate_labels, multi_label=True)
    
    # Анализ реплик HR
   
    return {
        'candidate': candidate_analysis,
    }

def identify_speaker(line, speaker_labels):
    """
    Определяет говорящего по меткам
    """
    line_lower = line.lower()
    
    for speaker, labels in speaker_labels.items():
        for label in labels:
            if label.lower() in line_lower:
                return speaker
    
    return 'unknown'

def clean_replica(text, labels_to_remove):
    """

    """
    for label in labels_to_remove:
        text = re.sub(rf'{label}:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()
    
    # Категории для анализа
    categories = [
        "профессиональные навыки",
        "мотивация", 
        "коммуникативные навыки",
        "опыт работы",
        "корпоративная культура",
        "технические вопросы",
        "личные качества"
    ]

@app.post("/api/receive")
async def receive(dialogue: str = Form(...)):

    #Код обработки

    detailed_analysis = analyze_dialogue_parts(dialogue, categories)
    
    if detailed_analysis['candidate']:
        result = "Кандидат: " + "; ".join(
        f"{label}: {score:.3f}" for label, score in zip(
            detailed_analysis['candidate']['labels'],
            detailed_analysis['candidate']['scores']
        )
    )
    
    # Переводим ответ кандидата на английский
    return JSONResponse({"answer": next_q_ru})

