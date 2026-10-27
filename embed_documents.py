
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from chunk_documents import chunk_documents


load_dotenv()


def create_embeddings():
    """Generate an embedding for one test chunk."""

    chunks = chunk_documents()

    if not chunks:
        print("No chunks found. Check the documents folder.")
        return

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    test_text = chunks[0].page_content

    vector = embeddings.embed_query(test_text)

    print("Embedding test successful!")
    print(f"Source: {chunks[0].metadata['source_file']}")
    print(f"Section: {chunks[0].metadata['section']}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


if __name__ == "__main__":
    create_embeddings()
