"""
FastAPI Backend for RAG Chatbot

Exposes the conversational RAG chain as a REST API.
Supports multi-turn conversation via session-based memory.

Endpoints:
    POST /chat          — Ask a question, get grounded answer + sources
    POST /reset         — Clear conversation history
    GET  /health        — Health check (used by Kubernetes probes)
    GET  /history       — Retrieve current conversation history
"""

import logging
import os
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.chain import RAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DG Liger RAG Chatbot",
    description=(
        "Conversational Q&A over proprietary business documents. "
        "Built with LangChain, FAISS, and StableLM Zephyr 3B (local inference)."
    ),
    version="1.0.0",
)

# ─── Request / Response Models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class ResetResponse(BaseModel):
    message: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    index_size: int
    model_loaded: bool

class HistoryMessage(BaseModel):
    role: str
    content: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[HistoryMessage]

# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    chain: Optional[RAGChain] = None

state = AppState()


@app.on_event("startup")
async def load_chain():
    """
    Load RAG chain at startup.

    Model and index paths can be configured via environment variables
    or fall back to sensible defaults.

    Why load at startup:
    - LLM loading takes 5-15 seconds (GGUF model reading + allocation)
    - FAISS index loading proportional to number of vectors
    - Per-request loading would make the API unusably slow
    - Single shared instance is thread-safe for inference
    """
    model_path = os.getenv(
        "MODEL_PATH", "models/stablelm-zephyr-3b.Q4_K_M.gguf"
    )
    index_path = os.getenv("INDEX_PATH", "data/faiss_index")

    logger.info("Loading RAG chain...")

    try:
        state.chain = RAGChain(
            model_path=model_path,
            index_path=index_path,
        )
        logger.info("RAG chain loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Startup failed: {e}")
        logger.warning(
            "API will start in degraded mode. "
            "/chat endpoint will return 503 until models are available."
        )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns model_loaded=False if startup failed.
    Kubernetes readiness probe uses this to determine when
    the pod is ready to receive traffic.
    """
    if state.chain is None:
        return HealthResponse(
            status="degraded",
            index_size=0,
            model_loaded=False,
        )

    return HealthResponse(
        status="healthy",
        index_size=state.chain.vectorstore.index.ntotal,
        model_loaded=True,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question and receive a grounded answer.

    The chain maintains conversation history across calls
    from the same session, enabling follow-up questions.

    Args:
        question:   Natural language question
        session_id: Optional session identifier (default: "default")

    Returns:
        answer:     LLM response grounded in retrieved document chunks
        sources:    Filenames of documents used to generate the answer
        session_id: Echo of the session ID for client tracking

    Example:
        curl -X POST http://localhost:8000/chat \\
             -H "Content-Type: application/json" \\
             -d '{"question": "What are the main consulting services?"}'
    """
    if state.chain is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG chain not loaded. "
                "Ensure model and FAISS index are available and restart."
            ),
        )

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    result = state.chain.ask(request.question)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id or "default",
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(session_id: str = "default"):
    """
    Clear conversation history for a session.

    Call this when starting a new conversation topic
    or when the user wants to start fresh.
    """
    if state.chain is None:
        raise HTTPException(status_code=503, detail="Chain not loaded.")

    state.chain.reset_memory()

    return ResetResponse(
        message="Conversation history cleared.",
        session_id=session_id,
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history(session_id: str = "default"):
    """
    Retrieve current conversation history.

    Useful for debugging why follow-up questions are or aren't
    being rephrased correctly.
    """
    if state.chain is None:
        raise HTTPException(status_code=503, detail="Chain not loaded.")

    messages = state.chain.get_history()

    formatted = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        formatted.append(HistoryMessage(role=role, content=msg.content))

    return HistoryResponse(session_id=session_id, messages=formatted)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
