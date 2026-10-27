
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_all_documents



def convert_to_documents() -> list[Document]:
    """Convert loaded SOP sections into LangChain documents."""

    loaded_documents = load_all_documents()
    documents = []

    for source in loaded_documents:
        current_section = None
        section_content = []

        def save_section():
            """Save collected content for the current section."""

            if not section_content:
                return

            documents.append(
                Document(
                    page_content="\n".join(section_content),
                    metadata={
                        "source_file": source["source_file"],
                        "title": source["title"],
                        "section": current_section,
                    },
                )
            )

        for item in source["sections"]:
            section = item["section"]

            # When the section changes, save the previous section.
            if (
                current_section is not None
                and section != current_section
            ):
                save_section()
                section_content = []

            current_section = section

            content = item["text"].strip()

            # Skip standalone headings that duplicate section metadata.
            if (
                item["type"] == "paragraph"
                and content == section
            ):
                continue

            # Mark content extracted from tables.
            if item["type"] == "table":
                content = "[TABLE]\n" + content

            section_content.append(content)

        # Save the final section of the current SOP.
        save_section()

    return documents



def chunk_documents() -> list[Document]:
    """Split section documents into smaller overlapping chunks."""

    documents = convert_to_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return chunks

def validate_chunks(chunks: list[Document]) -> None:
    """Report basic chunk quality checks."""

    sizes = [len(chunk.page_content) for chunk in chunks]

    print("\n--- Chunk Validation ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"Smallest chunk: {min(sizes)} characters")
    print(f"Largest chunk: {max(sizes)} characters")
    print(f"Average chunk size: {sum(sizes) / len(sizes):.1f} characters")

    missing_source = sum(
        not chunk.metadata.get("source_file")
        for chunk in chunks
    )

    missing_section = sum(
        not chunk.metadata.get("section")
        for chunk in chunks
    )

    empty_chunks = sum(
        not chunk.page_content.strip()
        for chunk in chunks
    )

    print(f"Chunks missing source filename: {missing_source}")
    print(f"Chunks missing section metadata: {missing_section}")
    print(f"Empty chunks: {empty_chunks}")

    print("\n--- Short Chunks (under 100 characters) ---")

    for index, chunk in enumerate(chunks):
        if len(chunk.page_content.strip()) < 100:
            print(f"\nChunk index: {index}")
            print(f"Source: {chunk.metadata['source_file']}")
            print(f"Section: {chunk.metadata['section']}")
            print(f"Content: {chunk.page_content!r}")

if __name__ == "__main__":
    chunks = chunk_documents()
    validate_chunks(chunks)

    print(f"Total chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks[:10], start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Source: {chunk.metadata['source_file']}")
        print(f"Section: {chunk.metadata['section']}")
        print(f"Characters: {len(chunk.page_content)}")
        print(chunk.page_content[:250].replace("\n", " "))
