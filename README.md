---

# 📄 README – Backend (FastAPI + Groq)

```md
# 🧠 Smart Inbox – Backend

API backend do projeto **Smart Inbox**, responsável por processar emails, classificá-los como **Produtivo** ou **Improdutivo** e gerar respostas automáticas utilizando Inteligência Artificial.

---

## 🚀 Tecnologias Utilizadas

- **Python 3.10+**
- **FastAPI** – Framework web moderno e performático
- **Pydantic** – Validação de dados
- **Groq API** – Geração de respostas e classificação com modelos LLM
- **Requests** – Comunicação HTTP
- **python-dotenv** – Gerenciamento de variáveis de ambiente
- **PyPDF** – Leitura de arquivos PDF
- **Uvicorn** – Servidor ASGI

---

## 🎯 Funcionalidades

- Recebimento de emails via:
  - Texto direto
  - Arquivos `.txt`
  - Arquivos `.pdf`
- Extração e normalização do texto
- Classificação do email em:
  - **Produtivo**
  - **Improdutivo**
- Geração de resposta automática adequada à categoria
- Retorno estruturado via JSON
- API pronta para integração com frontend web

---

## 📦 Instalação

Clone o repositório e crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 🔑 Variáveis de Ambiente

Crie um arquivo .env na raiz do projeto com o seguinte conteúdo:

```
GROQ_API_KEY=seu_token_aqui
```

## ▶️ Execução Local

```
uvicorn main:app --reload
```

A API estará disponível em: http://localhost:8000

## 🤖 Uso de Inteligência Artificial

A API utiliza a Groq API, com um modelo de linguagem de grande porte (LLM), para:

- Classificar o email via prompt engineering

- Gerar respostas contextualizadas e profissionais

- Controlar o formato da resposta via instruções explícitas no prompt

O ajuste da IA foi feito por meio de:

- Estruturação do prompt

- Controle de temperatura

- Padronização do formato de saída (JSON)
