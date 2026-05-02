"""
Conversational RAG Chain

Builds a multi-turn conversational retrieval chain using LangChain.
Users can ask follow-up questions that reference earlier context.

Key design decision — ConversationalRetrievalChain over RetrievalQA:
- RetrievalQA treats every question independently
- If user asks "What are the services?" then "What's the pricing for those?"
  RetrievalQA sends "What's the pricing for those?" to the vector DB
  This retrieves nothing useful — no context for what "those" refers to
- ConversationalRetrievalChain maintains ConversationBufferMemory
  It rephrases follow-up questions into standalone queries before searching:
  "What's the pricing for those?" → "What is the pricing for DG Liger services?"
  Now the vector search retrieves the correct context

Why StableLM Zephyr 3B locally over GPT-4 API:
- Client documents are proprietary business materials
- Sending them to OpenAI's servers violates confidentiality
- StableLM Zephyr 3B runs entirely on local hardware
- Q4_K_M quantisation reduces model size to ~2GB (CPU-runnable)
- Quality is sufficient for grounded Q&A over retrieved context
- For production with higher quality requirements: swap to Llama 2 7B
  or any GGUF model without changing this code
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import LlamaCpp
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "LangChain dependencies not installed.\n"
        "Run: pip install langchain llama-cpp-python"
    )


# System prompt for the RAG chatbot
# Why explicit constraints:
# - "Only answer from context" prevents hallucination
# - "Say I don't know" is better than a confident wrong answer
# - For DG Liger's use case: wrong business information = bad client advice
RAG_PROMPT_TEMPLATE = """You are a helpful assistant for DG Liger Consulting.
Answer questions based ONLY on the provided context from company documents.

Rules:
- If the context contains the answer, provide it clearly and concisely
- If the context does not contain enough information, say:
  "I don't have that information in the available documents."
- Never make up information or draw from general knowledge
- Keep answers focused and professional

Context from documents:
{context}

Conversation history:
{chat_history}

Question: {question}

Answer:"""


class RAGChain:
    """
    Multi-turn conversational RAG chain over business documents.

    Maintains conversation history across turns and rephrases
    follow-up questions before searching the vector store.
    """

    def __init__(
        self,
        model_path: str,
        index_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_retrieval: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 512,
        n_ctx: int = 4096,
    ):
        """
        Args:
            model_path:      Path to StableLM Zephyr GGUF model file
                             e.g. "models/stablelm-zephyr-3b.Q4_K_M.gguf"
            index_path:      Path to FAISS index created by DocumentIngester
            embedding_model: Must match the model used during ingestion
            top_k_retrieval: Number of chunks to retrieve per query.
                             3 gives good context without exceeding LLM context.
            temperature:     LLM sampling temperature.
                             0.1 = conservative and factual (correct for Q&A)
                             Higher values = more creative but less reliable
            max_tokens:      Maximum tokens for LLM response
            n_ctx:           Context window size for GGUF model
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Install: pip install langchain llama-cpp-python"
            )

        self.logger = logging.getLogger(__name__)
        self.top_k = top_k_retrieval

        # Load embeddings (must match ingestion model)
        self.logger.info(f"Loading embedding model: {embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load FAISS index
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                f"Run ingest.py first to build the index."
            )

        self.logger.info(f"Loading FAISS index from: {index_path}")
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self.logger.info(
            f"Index loaded: {self.vectorstore.index.ntotal} vectors"
        )

        # Load StableLM Zephyr 3B via llama.cpp
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Download StableLM Zephyr 3B GGUF from HuggingFace:\n"
                f"https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF\n"
                f"Place the .gguf file in the models/ directory."
            )

        self.logger.info(f"Loading LLM from: {model_path}")
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            # n_gpu_layers=-1 if GPU available, 0 for CPU
            n_gpu_layers=0,
            verbose=False,
        )
        self.logger.info("LLM loaded.")

        # Conversation memory
        # ConversationBufferMemory stores full conversation history
        # Why not ConversationSummaryMemory:
        # - For typical business Q&A sessions (10-20 turns), full history fits in context
        # - Summary memory can lose specific details needed for accurate follow-ups
        # - "What were the pricing details you mentioned?" needs the actual details, not a summary
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        # Build prompt
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )

        # Build the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k},
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False,
        )

        self.logger.info("RAG chain ready.")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer with sources.

        Args:
            question: Natural language question from the user

        Returns:
            Dict with:
            - answer:    The LLM's response grounded in retrieved context
            - sources:   List of source document filenames used
            - chunks:    The actual retrieved text chunks (for verification)

        How ConversationalRetrievalChain handles follow-ups:
        1. User asks "What are the main services?"
           Chain sends this directly to the retriever
        2. User asks "What's the pricing for those?"
           Chain first calls the LLM to rephrase using conversation history:
           "What is the pricing for DG Liger's main consulting services?"
           Then sends the rephrased question to the retriever
        3. The retrieved chunks + full chat history are sent to the LLM
        4. LLM generates a grounded answer
        """
        if not question or not question.strip():
            return {
                "answer": "Please ask a question.",
                "sources": [],
                "chunks": [],
            }

        try:
            result = self.chain({"question": question})

            # Extract source file names
            sources = []
            chunks = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    sources.append(Path(source).name)
                    chunks.append(doc.page_content)

            # Deduplicate sources
            sources = list(dict.fromkeys(sources))

            self.logger.debug(
                f"Q: {question[:60]}...\n"
                f"Sources: {sources}\n"
                f"A: {result['answer'][:100]}..."
            )

            return {
                "answer": result["answer"].strip(),
                "sources": sources,
                "chunks": chunks,
            }

        except Exception as e:
            self.logger.error(f"Chain error: {e}")
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "sources": [],
                "chunks": [],
            }

    def reset_memory(self):
        """Clear conversation history to start a fresh session."""
        self.memory.clear()
        self.logger.info("Conversation memory cleared.")

    def get_history(self) -> list:
        """Return current conversation history."""
        return self.memory.chat_memory.messages
