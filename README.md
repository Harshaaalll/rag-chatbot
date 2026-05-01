# RAG Chatbot with Conversational Memory

> Production retrieval-augmented generation chatbot over proprietary PDF documents. Multi-turn conversation support using LangChain, FAISS, and local StableLM Zephyr 3B inference.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.200-1C3C3C?style=flat-square)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Facebook-0668E1?style=flat-square)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

A production RAG (Retrieval Augmented Generation) chatbot that lets you ask natural language questions over any collection of PDF documents. Built to handle multi-turn conversations where follow-up questions reference earlier context.

Built during Data Analyst internship at DG Liger Consulting for extracting intelligence from proprietary business documents.

---

## Why RAG over direct LLM prompting?

| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| Hallucination | LLM invents plausible-sounding answers | Answers grounded in retrieved documents |
| Private data | LLM has no knowledge of your documents | Any document can be indexed |
| Cost | Full context in every prompt = expensive | Only 3-5 relevant chunks sent per query |
| Accuracy | General knowledge, often wrong on specifics | Specific to your exact document content |

---

## Architecture

```
Document Ingestion (one-time):
─────────────────────────────
PDFs / Web Pages
      │
      ▼
LangChain Document Loaders
      │
      ▼
RecursiveCharacterTextSplitter
  chunk_size=500, overlap=50
      │
      ▼
all-MiniLM-L6-v2 Embeddings
  (384-dimensional vectors)
      │
      ▼
FAISS Vector Index
  (stored on disk)


Query Time (per question):
─────────────────────────
User Question
      │
      ▼
ConversationBufferMemory
  (rephrases with history)
      │
      ▼
Query Embedding
  (all-MiniLM-L6-v2)
      │
      ▼
FAISS Similarity Search
  (top-3 chunks, cosine sim)
      │
      ▼
Prompt Construction
  system + context + history + question
      │
      ▼
StableLM Zephyr 3B (GGUF Q4_K_M)
  local inference via llama.cpp
      │
      ▼
Grounded Answer + Sources
```

---

## Key Technical Decisions

### Why StableLM Zephyr 3B locally vs API?
- **Zero ongoing cost** — no API fees per query
- **Data privacy** — proprietary client documents never leave the machine
- **No latency from network** — inference runs locally
- **Q4_K_M quantisation** — 4-bit weights reduce model size from ~6GB to ~2GB, enabling CPU inference
- Trade-off: smaller than GPT-4, but for grounded Q&A over retrieved context, quality is sufficient

### Why ConversationalRetrievalChain over RetrievalQA?
`RetrievalQA` treats each question in isolation. If a user asks:
- Q1: "What are the consulting fees?"
- Q2: "And what's included in that?"

`RetrievalQA` on Q2 would search for "And what's included in that?" — meaningless without context.

`ConversationalRetrievalChain` uses `ConversationBufferMemory` to rephrase Q2 into a standalone query: "What is included in the consulting fees?" — which retrieves correct context.

### Why RecursiveCharacterTextSplitter?
Unlike character or word splitters, `RecursiveCharacterTextSplitter` tries to split on natural boundaries — paragraphs first, then sentences, then words, then characters. This preserves semantic coherence within chunks. A chunk that ends mid-sentence loses context at its boundary.

---

## Project Structure

```
rag-chatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # Model paths, chunk size, overlap, k
├── src/
│   ├── ingest.py            # Document loading + indexing pipeline
│   ├── retriever.py         # FAISS vector store + similarity search
│   ├── chain.py             # ConversationalRetrievalChain setup
│   ├── model.py             # StableLM Zephyr 3B loader (GGUF)
│   └── api.py               # FastAPI endpoint
├── data/
│   └── documents/           # Place PDFs here for indexing
├── models/
│   └── .gitkeep            # Download StableLM GGUF here
├── tests/
│   ├── test_retriever.py
│   └── test_chain.py
├── notebooks/
│   └── 01_rag_demo.ipynb
└── Dockerfile
```

---

## Installation

```bash
git clone https://github.com/harshalbhambhani/rag-chatbot
cd rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download StableLM Zephyr 3B GGUF model
# Place in models/ directory
# Model: stablelm-zephyr-3b.Q4_K_M.gguf
# Source: https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF
```

---

## Requirements

```
langchain==0.0.200
sentence-transformers==2.2.2
faiss-cpu==1.7.4
llama-cpp-python==0.1.85
pypdf==3.8.1
pydantic==1.10.7
fastapi==0.95.0
uvicorn==0.21.1
```

---

## Usage

```python
from src.ingest import build_index
from src.chain import build_chain

# Step 1: Index your documents (run once)
build_index(documents_dir="data/documents/", index_path="data/faiss_index")

# Step 2: Build the conversational chain
chain = build_chain(index_path="data/faiss_index")

# Step 3: Chat
response = chain({"question": "What are the main service offerings?"})
print(response["answer"])
print("Sources:", response["source_documents"])

# Follow-up — the chain remembers context
response2 = chain({"question": "And what are the pricing tiers for those?"})
print(response2["answer"])
```

---

*Built at DG Liger Consulting · Jun–Jul 2024*
*Harshal Bhambhani · BITS Hyderabad · 2026*