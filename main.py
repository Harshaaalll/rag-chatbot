"""
RAG Chatbot — Command Line Interface

Interactive terminal chat over your document collection.
Type questions, get grounded answers with source citations.

Usage:
    # First build the index (one time)
    python src/ingest.py data/documents/ data/faiss_index

    # Then start chatting
    python main.py

    # Or specify custom paths
    python main.py --model models/stablelm-zephyr-3b.Q4_K_M.gguf \\
                   --index data/faiss_index

Commands during chat:
    /reset   — Clear conversation history
    /history — Show conversation so far
    /quit    — Exit the chatbot
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose LangChain/llama.cpp logs
    format="%(levelname)s: %(message)s",
)

from src.chain import RAGChain


def print_welcome():
    print("\n" + "═" * 60)
    print("  DG Liger RAG Chatbot")
    print("  Ask questions about your documents")
    print("  Commands: /reset  /history  /quit")
    print("═" * 60 + "\n")


def print_answer(result: dict):
    print(f"\n📋 Answer:\n{result['answer']}")
    if result["sources"]:
        print(f"\n📄 Sources: {', '.join(result['sources'])}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot CLI — Q&A over your documents"
    )
    parser.add_argument(
        "--model",
        default="models/stablelm-zephyr-3b.Q4_K_M.gguf",
        help="Path to StableLM Zephyr 3B GGUF model file",
    )
    parser.add_argument(
        "--index",
        default="data/faiss_index",
        help="Path to FAISS index directory",
    )
    args = parser.parse_args()

    print_welcome()
    print("⏳ Loading models... (this takes 10-30 seconds on first run)\n")

    try:
        chain = RAGChain(
            model_path=args.model,
            index_path=args.index,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(
            "\nSetup steps:\n"
            "1. Place your PDF documents in data/documents/\n"
            "2. Run: python src/ingest.py\n"
            "3. Download StableLM Zephyr 3B GGUF to models/\n"
            "   https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF\n"
        )
        sys.exit(1)

    print("✅ Ready! Ask your first question.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "/quit":
            print("Goodbye!")
            break

        if question.lower() == "/reset":
            chain.reset_memory()
            print("🔄 Conversation history cleared.\n")
            continue

        if question.lower() == "/history":
            history = chain.get_history()
            if not history:
                print("📭 No conversation history yet.\n")
            else:
                print("\n📜 Conversation History:")
                for msg in history:
                    role = "You" if msg.type == "human" else "Bot"
                    print(f"  {role}: {msg.content[:100]}...")
                print()
            continue

        # Ask the chain
        result = chain.ask(question)
        print_answer(result)


if __name__ == "__main__":
    main()
