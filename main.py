import os
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware
import json

# =========================
# CONFIG
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY não definida nas variáveis de ambiente")

app = FastAPI(title="Smart Inbox API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELOS
# =========================

class EmailResponse(BaseModel):
    category: str
    response: str

# =========================
# UTILIDADES
# =========================

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_txt(file: UploadFile) -> str:
    return file.file.read().decode("utf-8").strip()

# =========================
# IA (GROQ)
# =========================

def analyze_email_with_ai(email_text: str) -> EmailResponse:
    prompt = f"""
Você é um assistente de atendimento por email.

1- Classifique o email abaixo como:
- produtivo
- improdutivo

2- Gere uma resposta educada, clara e profissional em português brasileiro.

Email:
\"\"\"{email_text}\"\"\"

Responda EXATAMENTE neste formato JSON:
{{
  "category": "produtivo ou improdutivo",
  "response": "texto da resposta"
}}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Erro Groq {response.status_code}: {response.text}")

    content = response.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
        return EmailResponse(
            category=parsed["category"],
            response=parsed["response"],
        )
    except Exception:
        raise RuntimeError(f"Resposta inválida da IA: {content}")

# =========================
# ENDPOINTS
# =========================

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/analyze", response_model=EmailResponse)
async def analyze_email(
    text: str = Form(None),
    file: UploadFile = File(None),
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="Envie texto ou arquivo")

    content = ""

    if text:
        content = text.strip()

    elif file:
        if file.filename.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        elif file.filename.endswith(".txt"):
            content = extract_text_from_txt(file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato não suportado. Use .txt ou .pdf",
            )

    if not content:
        raise HTTPException(status_code=400, detail="Conteúdo vazio")

    return analyze_email_with_ai(content)
