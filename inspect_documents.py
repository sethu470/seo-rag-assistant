
from document_loader import load_all_documents

documents = load_all_documents()

for document in documents:
    print("\n" + "=" * 80)
    print(f"DOCUMENT: {document['title']}")
    print(f"FILE: {document['source_file']}")

    headings = sorted({
        item["section"]
        for item in document["sections"]
        if item["type"] == "paragraph"
        and item["style"].startswith("Heading")
    })

    print("\nRECOGNIZED HEADING PATHS:")
    for heading in headings:
        print(f"- {heading}")

    print("\nFIRST THREE TABLES:")
    table_count = 0

    for item in document["sections"]:
        if item["type"] == "table":
            table_count += 1
            print(f"\nSection: {item['section']}")
            print(item["text"][:600])

            if table_count == 3:
                break
