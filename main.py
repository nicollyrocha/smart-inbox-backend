from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from transformers import pipeline
from pypdf import PdfReader

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
    model="google/flan-t5-base",
    max_length=150
)

# -----------------------------
# Funções auxiliares
# -----------------------------

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
     prompt = (
        "Write a professional reply in Brazilian Portuguese.\n"
        "Do not repeat the email content.\n"
        f"Category: {category}\n"
        "Reply:"
    )

     result = generator(
        prompt,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.8,
    )

     return result[0]["generated_text"].strip()

def build_final_reply(ai_suggestion: str, category: str) -> str:
    if category == "produtivo":
        return f"Olá! {ai_suggestion} Nossa equipe retornará em breve."
    else:
        return f"Olá! {ai_suggestion} Agradecemos o contato."



# -----------------------------
# Endpoint principal
# -----------------------------

@app.post("/analyze")
async def analyze_email(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not text and not file:
        return {"error": "Nenhum conteúdo enviado"}

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
    category = classify_email(content)
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
