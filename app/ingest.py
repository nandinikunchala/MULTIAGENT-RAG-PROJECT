from pathlib import Path
import re

from pypdf import PdfReader

from app.vectorstore import get_vector_store


def load_pdf(path):
    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return "\n\n".join(pages)


def clean_text(text):
    """
    Clean common PDF extraction noise.
    """

    # Normalize spaces
    text = text.replace("\xa0", " ")
    text = text.replace("\u0000", "")

    # Remove repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove page numbers such as:
    # Page 42 / 44
    text = re.sub(
        r"Consistency AI\s*·\s*Project 1\s*·\s*Multi-Agent RAG Build Guide\s*Page\s*\d+\s*/\s*\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove standalone page numbers
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )

    # Remove common navigation noise
    noise_patterns = [
        r"(?m)^\s*&?\s*resume bullets\s*$",
        r"(?m)^\s*Convert the build into wins\s*$",
        r"(?m)^\s*60-second pitch.*$",
        r"(?m)^\s*Interview questions.*$",
        r"(?m)^\s*What to build next.*$",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text, chunk_size=1000):
    """
    Create chunks from cleaned PDF text.

    Paragraphs are combined until approximately
    chunk_size characters are reached.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) + 2 <= chunk_size:

            if current:
                current += "\n\n" + paragraph
            else:
                current = paragraph

        else:

            if current:
                chunks.append(current)

            if len(paragraph) > chunk_size:

                for i in range(0, len(paragraph), chunk_size):
                    piece = paragraph[i:i + chunk_size].strip()

                    if piece:
                        chunks.append(piece)

                current = ""

            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def ingest_pdf(path):

    # 1. Read PDF
    text = load_pdf(path)

    # 2. Clean extracted text
    text = clean_text(text)

    # 3. Create chunks
    chunks = chunk_text(text)

    # 4. Create documents
    documents = [
        {
            "text": chunk,
            "source": str(path),
            "chunk_id": i
        }
        for i, chunk in enumerate(chunks)
    ]

    # 5. Create vector store
    store = get_vector_store()

    # 6. Add documents and embeddings
    store.add_documents(documents)

    return store


if __name__ == "__main__":

    pdf_path = Path("paper.pdf")

    store = ingest_pdf(pdf_path)

    print("PDF ingestion completed.")
    print("Number of documents:", len(store.documents))