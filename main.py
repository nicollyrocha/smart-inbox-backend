from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AutoMail AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção pode restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmailRequest(BaseModel):
    texto: str


@app.post("/classificar")
def classificar_email(email: EmailRequest):
    prompt = f"""
Você é um assistente que classifica emails e sugere respostas.

Classifique o email abaixo em apenas UMA das categorias:
- produtivo (exige ação ou resposta)
- improdutivo (não exige ação imediata)

Depois, gere uma resposta automática educada e profissional.

Retorne no formato:
Categoria: <produtivo ou improdutivo>
Resposta: <texto da resposta>

Email:
\"\"\"
{email.texto}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content

        categoria = "produtivo" if "produtivo" in content else "improdutivo"
        resposta = content.split("Resposta:")[-1].strip()

        return {
            "categoria": categoria,
            "resposta": resposta
        }

    except Exception as e:
        return {
            "erro": "Erro ao processar o email",
            "detalhes": str(e)
        }


@app.get("/")
def root():
    return {"status": "API AutoMail AI rodando"}