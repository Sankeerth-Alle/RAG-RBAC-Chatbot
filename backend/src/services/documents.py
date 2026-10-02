"""Validate and index documents uploaded by authenticated users."""

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import Config, get_vector_db_path


MAX_UPLOAD_SIZE = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}


def _document_texts(filename: str, content: bytes) -> list[str]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported file types are TXT, Markdown, CSV, and PDF.")

    if extension == ".pdf":
        try:
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
        except Exception as exc:
            raise ValueError("The PDF could not be read.") from exc
        if not text.strip():
            raise ValueError("The PDF contains no extractable text.")
        return [text]

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded file must be UTF-8 encoded.") from exc

    if extension != ".csv":
        if not text.strip():
            raise ValueError("The uploaded file is empty.")
        return [text]

    rows = csv.DictReader(io.StringIO(text, newline=""))
    if not rows.fieldnames:
        raise ValueError("The CSV must include a header row.")
    documents = [
        "\n".join(f"{key}: {value}" for key, value in row.items() if key and value is not None)
        for row in rows
    ]
    documents = [document for document in documents if document.strip()]
    if not documents:
        raise ValueError("The CSV contains no data rows.")
    return documents


def index_uploaded_file(
    filename: str,
    content: bytes,
    access_level: str,
    username: str,
) -> int:
    """Add an uploaded file to Chroma, with access derived from the caller."""
    safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if len(content) > MAX_UPLOAD_SIZE:
        raise ValueError("File exceeds the 10 MB limit.")

    texts = _document_texts(safe_filename, content)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    base_metadata = {
        "access_level": access_level,
        "source_file": safe_filename,
        "source": f"uploads/{access_level}/{safe_filename}",
        "uploaded_by": username,
        "uploaded_at": uploaded_at,
    }
    documents = [
        Document(page_content=text, metadata=base_metadata.copy())
        for text in texts
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = splitter.split_documents(documents)

    try:
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name=Config.VECTOR_DB_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(get_vector_db_path()),
        )
        stored_model = (vector_store._collection.metadata or {}).get("embedding_model")
        if stored_model != Config.EMBEDDING_MODEL:
            raise RuntimeError(
                "The vector database embedding model is missing or incompatible. "
                "Rebuild the vector database before uploading documents."
            )
        vector_store.add_documents(documents)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Unable to index the uploaded document.") from exc

    return len(documents)