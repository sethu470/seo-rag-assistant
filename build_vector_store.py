
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from chunk_documents import chunk_documents


load_dotenv()

VECTOR_STORE_DIR = Path(__file__).parent / "chroma_db"


def build_vector_store():
    """Create a local Chroma database from the SOP chunks."""

    chunks = chunk_documents()

    if not chunks:
        print("No chunks found. Check the documents folder.")
        return

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name="seo_sop_knowledge",
    )

    print("Vector store created successfully!")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Database location: {VECTOR_STORE_DIR}")
    print(f"Collection: seo_sop_knowledge")

    return vector_store


if __name__ == "__main__":
    build_vector_store()
