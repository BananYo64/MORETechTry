import os
import shutil
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import transformers

app = FastAPI()

# CORS, чтобы браузер не ругался
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ Лучше ограничить своим доменом/хостом
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Переводчики
translator_ru_en = transformers.pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en", device=-1)
translator_en_ru = transformers.pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru", device=-1)

# Модель интервьюера
interviewer = transformers.pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda:0",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

def ru_to_en(text: str) -> str:
    result = translator_ru_en(text)
    return result[0]['translation_text']

def en_to_ru(text: str) -> str:
    result = translator_en_ru(text)
    return result[0]['translation_text']


# --- Универсальный эндпоинт ---
class CandidateAnswer(BaseModel):
    candidate_answer_ru: str

@app.post("/api/receive")
async def receive(
    request: Request,
    candidate_answer_ru: str = Form(None),  # если данные пришли как form-data
):
    try:
        if candidate_answer_ru is None:
            # пробуем как JSON
            data = await request.json()
            candidate_answer_ru = data.get("candidate_answer_ru")

        if not candidate_answer_ru:
            raise HTTPException(status_code=400, detail="Нет текста ответа")

        # Переводим ответ кандидата на английский
        candidate_answer_en = ru_to_en(candidate_answer_ru)

        # Генерируем следующий вопрос на английском
        prompt_en = (
            f"You are an HR interviewer. The candidate answered: '{candidate_answer_en}'. "
            "Ask the next clarifying question. The response must be strictly in the form "
            "of a single question, without any explanations, introductory text, or analysis."
        )
        generated = interviewer(prompt_en)[0]["generated_text"]
        next_q_en = generated.replace(prompt_en, "").strip()

        # Переводим вопрос обратно на русский
        next_q_ru = en_to_ru(next_q_en)

        return JSONResponse({"answer": next_q_ru})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Отдельный эндпоинт для аудио ---
@app.post("/api/receive-audio")
async def receive_audio(audio: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, audio.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    return {"message": "Аудио получено", "filename": audio.filename}
