
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

VECTOR_STORE_DIR = Path(__file__).parent / "chroma_db"


def search_knowledge_base(query: str, k: int = 3):
    """Retrieve the most relevant SOP chunks for a question."""

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name="seo_sop_knowledge",
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings,
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    print(f"\nQuestion: {query}")
    print(f"Results found: {len(results)}")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {document.metadata.get('source_file')}")
        print(f"Section: {document.metadata.get('section')}")
        print(f"Distance: {distance:.4f}")
        print(f"Content:\n{document.page_content}")


if __name__ == "__main__":
    search_knowledge_base(
        "How should I investigate pages that are not indexed by Google?"
    )
