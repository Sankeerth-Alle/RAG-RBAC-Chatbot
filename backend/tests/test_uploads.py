import pytest

from services import documents


def test_upload_index_stores_authenticated_role_and_safe_filename(monkeypatch):
    indexed = {}

    class FakeCollection:
        metadata = {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}

    class FakeStore:
        _collection = FakeCollection()

        def __init__(self, **kwargs):
            pass

        def add_documents(self, items):
            indexed["documents"] = items

    monkeypatch.setattr(documents, "HuggingFaceEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(documents, "Chroma", FakeStore)
    count = documents.index_uploaded_file("folder\\notes.md", b"Engineering notes", "engineering", "Tony")

    assert count == 1
    metadata = indexed["documents"][0].metadata
    assert metadata["source_file"] == "notes.md"
    assert metadata["access_level"] == "engineering"
    assert metadata["uploaded_by"] == "Tony"


def test_csv_upload_becomes_searchable_records():
    records = documents._document_texts(
        "employees.csv",
        b"Full_Name,Department\nAadhya Patel,Engineering\n",
    )
    assert records == ["Full_Name: Aadhya Patel\nDepartment: Engineering"]


def test_upload_rejects_unsupported_file_type():
    with pytest.raises(ValueError, match="Supported file types"):
        documents._document_texts("archive.exe", b"not a document")