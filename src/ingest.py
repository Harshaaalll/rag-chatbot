"""
Document Ingestion Pipeline for RAG Chatbot

Loads PDFs and other documents, splits them into chunks,
generates embeddings, and stores in a FAISS vector index.

This is a one-time operation — run it once per document collection.
The resulting FAISS index is then loaded at inference time.

Key decisions:
1. RecursiveCharacterTextSplitter over CharacterTextSplitter:
   - Tries to split on natural boundaries: paragraphs → sentences → words
   - Preserves semantic coherence within chunks
   - A chunk ending mid-sentence loses its own meaning

2. chunk_size=500, overlap=50:
   - 500 characters ≈ 100-120 words — enough for one business concept
   - 50-character overlap ensures context at chunk boundaries
   - Smaller chunks (100 chars) too fragmented for business Q&A
   - Larger chunks (1000 chars) may exceed LLM context budget

3. all-MiniLM-L6-v2 for embeddings:
   - 384-dimensional vectors — compact and fast for FAISS search
   - Strong semantic understanding for business English
   - 5x faster than larger models with minimal quality loss
   - Production-proven in many enterprise RAG deployments

4. FAISS over managed vector DBs for DG Liger use case:
   - Client documents are proprietary — cannot use cloud vector DBs
   - FAISS runs entirely in memory on local hardware
   - Sub-millisecond search across thousands of chunks
   - No ongoing infrastructure cost
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain.document_loaders import (
        PyPDFLoader,
        DirectoryLoader,
        TextLoader,
        UnstructuredFileLoader,
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "LangChain not installed.\n"
        "Run: pip install langchain sentence-transformers faiss-cpu pypdf"
    )


class DocumentIngester:
    """
    Loads documents, chunks them, embeds chunks, and builds FAISS index.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            chunk_size:      Characters per chunk.
                             500 ≈ 100 words — one focused concept per chunk.
            chunk_overlap:   Characters repeated between consecutive chunks.
                             50 ensures boundary context is never lost.
            embedding_model: SentenceTransformer model for chunk embeddings.
                             all-MiniLM-L6-v2: fast, 384-dim, strong quality.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Install: pip install langchain sentence-transformers faiss-cpu pypdf"
            )

        self.logger = logging.getLogger(__name__)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Split on these boundaries in order of preference
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.logger.info("Embedding model loaded.")

    def load_documents(self, documents_dir: str) -> List[Document]:
        """
        Load all supported documents from a directory.

        Supported file types:
        - PDF files (.pdf) — primary format for business documents
        - Text files (.txt) — plain text documents
        - Other formats handled by UnstructuredFileLoader

        Args:
            documents_dir: Path to directory containing documents

        Returns:
            List of LangChain Document objects with text and metadata
        """
        docs_path = Path(documents_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

        all_documents = []

        # Load PDFs
        pdf_files = list(docs_path.glob("**/*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                all_documents.extend(docs)
                self.logger.info(f"Loaded PDF: {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                self.logger.error(f"Failed to load {pdf_file.name}: {e}")

        # Load text files
        txt_files = list(docs_path.glob("**/*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                docs = loader.load()
                all_documents.extend(docs)
                self.logger.info(f"Loaded TXT: {txt_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {txt_file.name}: {e}")

        self.logger.info(
            f"Total documents loaded: {len(all_documents)} pages/sections "
            f"from {len(pdf_files)} PDFs and {len(txt_files)} text files"
        )
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.

        Why RecursiveCharacterTextSplitter:
        - Tries paragraph splits (\n\n) first — preserves semantic blocks
        - Falls back to sentence splits (. ) if chunks still too large
        - Falls back to word splits ( ) if needed
        - Last resort: character splits
        This hierarchy means chunks stay semantically coherent at each level.

        Args:
            documents: List of loaded Document objects

        Returns:
            List of chunked Document objects with inherited metadata
        """
        chunks = self.splitter.split_documents(documents)

        self.logger.info(
            f"Splitting complete: {len(documents)} documents → "
            f"{len(chunks)} chunks"
        )

        # Log chunk size statistics
        sizes = [len(c.page_content) for c in chunks]
        if sizes:
            self.logger.info(
                f"Chunk sizes — min: {min(sizes)}, "
                f"max: {max(sizes)}, "
                f"avg: {sum(sizes) // len(sizes)} chars"
            )

        return chunks

    def build_index(
        self,
        documents_dir: str,
        index_path: str,
    ) -> FAISS:
        """
        Full ingestion pipeline: load → split → embed → index → save.

        Args:
            documents_dir: Directory containing source documents
            index_path:    Where to save the FAISS index

        Returns:
            FAISS vectorstore ready for retrieval

        Process:
        1. Load all documents from directory
        2. Split into 500-char overlapping chunks
        3. Generate 384-dim embeddings for each chunk
        4. Build FAISS flat index (exact search)
        5. Save index to disk for reuse
        """
        # Load
        documents = self.load_documents(documents_dir)
        if not documents:
            raise ValueError(f"No documents found in: {documents_dir}")

        # Split
        chunks = self.split_documents(documents)

        # Embed and build FAISS index
        self.logger.info(
            f"Building FAISS index from {len(chunks)} chunks. "
            f"This may take a few minutes..."
        )
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save to disk
        Path(index_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(index_path)

        self.logger.info(
            f"FAISS index built and saved to: {index_path}\n"
            f"Total vectors indexed: {vectorstore.index.ntotal}"
        )

        return vectorstore

    def load_index(self, index_path: str) -> FAISS:
        """
        Load existing FAISS index from disk.

        Args:
            index_path: Path where index was saved by build_index()

        Returns:
            Loaded FAISS vectorstore
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at: {index_path}\n"
                f"Run build_index() first to create it."
            )

        vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.logger.info(
            f"FAISS index loaded from: {index_path} "
            f"({vectorstore.index.ntotal} vectors)"
        )
        return vectorstore


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    documents_dir = sys.argv[1] if len(sys.argv) > 1 else "data/documents"
    index_path = sys.argv[2] if len(sys.argv) > 2 else "data/faiss_index"

    ingester = DocumentIngester()
    ingester.build_index(documents_dir, index_path)
    print(f"\nIndex ready at: {index_path}")
    print("You can now start the chatbot with: python main.py")
