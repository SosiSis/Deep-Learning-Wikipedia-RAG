 [![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SosiSis/Deep-Learning-Wikipedia-RAG)


# Deep Learning Wikipedia RAG (LangChain + Groq)

A focused RAG assistant that answers deep-learning questions using a small, curated subset of Wikipedia.

## How it Works
- **Ingestion**: pull selected Wikipedia pages, chunk with `RecursiveCharacterTextSplitter`
- **Embeddings**: `all-MiniLM-L6-v2` (fast, local)
- **Index**: FAISS (saved under `vectorstore/faiss_index`)
- **LLM**: Groq (`llama-3.1-*` via `langchain-groq`)
- **Prompting**: system prompt enforces expert tone + cites page titles

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt

# Configure Groq
cp .env.example .env  # (Windows PowerShell) copy .env.example .env
# Set GROQ_API_KEY in .env

# Ingest curated Wikipedia pages
python -m src.ingest

# Ask a question (CLI)
python -m src.app_cli "Why does batch norm help with covariate shift?"

# Optional mini UI
python -m src.app_gradio


```

### Windows notes
If `faiss-cpu` fails to build:
```powershell
pip install --only-binary :all: faiss-cpu
```

## Data & License
This project uses a **small subset** of Wikipedia for educational purposes. Wikipedia content is CC BY-SA; include page titles in responses and in your publication.

## Example Questions
See `tests/sample_queries.txt`.
