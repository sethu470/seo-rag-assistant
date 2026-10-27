
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def get_heading_level(paragraph: Paragraph) -> int | None:
    """Return the heading level if this paragraph uses a Heading style."""

    style_name = paragraph.style.name if paragraph.style else ""

    if style_name.startswith("Heading "):
        try:
            level = int(style_name.split()[-1])
            if 1 <= level <= 9:
                return level
        except ValueError:
            pass

    return None


def extract_table(table: Table) -> str:
    """Convert a Word table into readable text with row labels."""

    rows = []

    for row in table.rows:
        cells = [
            " ".join(cell.text.split())
            for cell in row.cells
        ]

        # Skip rows that contain no meaningful text.
        if not any(cells):
            continue

        rows.append(cells)

    if not rows:
        return ""

    # Treat the first row as the table's column headings.
    headers = rows[0]
    formatted_rows = []

    for row in rows[1:]:
        fields = []

        for index, value in enumerate(row):
            if not value:
                continue

            if index < len(headers) and headers[index]:
                fields.append(f"{headers[index]}: {value}")
            else:
                fields.append(f"Column {index + 1}: {value}")

        if fields:
            formatted_rows.append("; ".join(fields))

    # For a table with only one non-empty row, retain that row.
    if not formatted_rows:
        formatted_rows.append("; ".join(
            f"Column {index + 1}: {value}"
            for index, value in enumerate(headers)
            if value
        ))

    return "\n".join(formatted_rows)


def load_docx(file_path: Path) -> dict:
    """Extract paragraphs and tables in document order."""

    doc = Document(file_path)

    content = []
    heading_stack = []

    for block in doc.iter_inner_content():

        if isinstance(block, Paragraph):
            text = block.text.strip()

            if not text:
                continue

            level = get_heading_level(block)

            if level is not None:
                # Keep parent headings and replace headings at this level.
                heading_stack = [
                    item for item in heading_stack
                    if item["level"] < level
                ]

                heading_stack.append({
                    "level": level,
                    "text": text,
                })

            section_path = " > ".join(
                item["text"] for item in heading_stack
            ) or "Introduction"

            content.append({
                "type": "paragraph",
                "text": text,
                "style": block.style.name if block.style else "",
                "section": section_path,
            })

        elif isinstance(block, Table):
            table_text = extract_table(block)

            if not table_text:
                continue

            section_path = " > ".join(
                item["text"] for item in heading_stack
            ) or "Introduction"

            content.append({
                "type": "table",
                "text": table_text,
                "style": "Table",
                "section": section_path,
            })

    return {
        "source_file": file_path.name,
        "title": doc.core_properties.title or file_path.stem,
        "sections": content,
    }


def load_all_documents() -> list[dict]:
    """Load all DOCX files from the documents folder."""

    files = sorted(DOCUMENTS_DIR.glob("*.docx"))
    results = []

    for file_path in files:
        try:
            results.append(load_docx(file_path))
        except Exception as exc:
            print(f"Failed to load {file_path.name}: {exc}")

    return results


if __name__ == "__main__":
    documents = load_all_documents()

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        paragraphs = sum(
            item["type"] == "paragraph"
            for item in document["sections"]
        )
        tables = sum(
            item["type"] == "table"
            for item in document["sections"]
        )

        print(f"\nDocument: {document['source_file']}")
        print(f"Title: {document['title']}")
        print(f"Paragraphs: {paragraphs}")
        print(f"Tables: {tables}")

        for item in document["sections"][:5]:
            preview = item["text"].replace("\n", " ")[:100]

            print(
                f"  [{item['type']}] "
                f"[{item['section']}] {preview}"
            )
