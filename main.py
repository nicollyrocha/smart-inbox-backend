from click import prompt
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from transformers import pipeline
from pypdf import PdfReader

import re

STOPWORDS = {
    "o", "a", "e", "de", "do", "da", "em", "um", "uma",
    "para", "por", "com", "que", "os", "as"
}

# -----------------------------
# Inicialização da aplicação
# -----------------------------

app = FastAPI(title="Smart Inbox AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Modelo Hugging Face
# -----------------------------

classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3"
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=150
)

# -----------------------------
# Funções auxiliares
# -----------------------------


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zà-ú\s]", "", text)

    return " ".join(
        word for word in text.split()
        if word not in STOPWORDS
    )

def extract_text_from_pdf(file) -> str:
    """
    Extrai texto de um arquivo PDF.
    PDFs escaneados (imagem) não são suportados.
    """
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text.strip()


def classify_email(text: str) -> str:
    """
    Classifica o email como produtivo ou improdutivo
    usando zero-shot classification.
    """
    labels = ["produtivo", "improdutivo"]
    result = classifier(text, labels)
    return result["labels"][0]

def generate_reply_ai(email_text: str, category: str) -> str:
    if category == "produtivo":
      prompt = f"""
Write a polite and professional reply to an email where the sender is requesting information or action:
"""


      result = generator(
        prompt,
        max_length=60,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=2.2,
        no_repeat_ngram_size=3,
    )

      return result[0]["generated_text"].strip()
    else:
            prompt = f"""
            Write a polite, professional and short reply to an email that does not require any action or answer, like a thank you note or an acknowledgment.
            Do NOT offer any additional information or tell the sender you'll contact them. It's just a simple acknowledgment.
            """


    result = generator(
        prompt,
        max_length=60,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=2.2,
        no_repeat_ngram_size=3,
    )

    return result[0]["generated_text"].strip()


def build_final_reply(ai_suggestion: str, category: str) -> str:
    if category == "produtivo":
        return f"Hello! {ai_suggestion} Our team will get back to you shortly."
    else:
        return f"Hello! {ai_suggestion}"



# -----------------------------
# Endpoint principal
# -----------------------------

@app.post("/analyze")
async def analyze_email(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not text and not file:
        return {"error": "No content provided"}

    # Decide a origem do conteúdo
    if file:
        if file.content_type == "application/pdf":
            content = extract_text_from_pdf(file.file)
        else:
            # TXT
            content = (await file.read()).decode("utf-8")
    else:
        content = text

    # Evita textos vazios
    if not content or not content.strip():
        return {"error": "Não foi possível extrair texto do email"}

    # Classificação + resposta
    processed_text = preprocess_text(content)
    category = classify_email(processed_text)
    ai_suggestion = generate_reply_ai(content, category)
    reply = build_final_reply(ai_suggestion, category)

    return {
        "category": category,
        "reply": reply,
        "model": "valhalla/distilbart-mnli-12-3 + google/flan-t5-base"
    }

# -----------------------------
# Endpoint de teste
# -----------------------------

@app.get("/")
def root():
    return {"status": "API Smart Inbox rodando"}
