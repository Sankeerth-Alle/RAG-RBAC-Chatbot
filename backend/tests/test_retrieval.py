from types import SimpleNamespace

from google.api_core.exceptions import ResourceExhausted

from agents import nodes


def make_document(content, access_level):
    return SimpleNamespace(
        page_content=content,
        metadata={"access_level": access_level, "source_file": "hr_data.csv"},
    )


class FakeCollection:
    metadata = {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}

    def get(self, **kwargs):
        return {
            "documents": [
                "employee_id: FINEMP1000\nfull_name: Aadhya Patel\nsalary: 1332478.37",
                "employee_id: FINEMP1004\nfull_name: Aadhya Saxena\nsalary: 1922205.04",
            ],
            "metadatas": [
                {"access_level": "hr", "source_file": "hr_data.csv"},
                {"access_level": "hr", "source_file": "hr_data.csv"},
            ],
        }


class FakeStore:
    _collection = FakeCollection()

    def __init__(self, **kwargs):
        pass

    def similarity_search(self, question, k, filter):
        return [make_document("wrong access", "engineering")]


def test_exact_hr_lookup_returns_only_matching_employee(monkeypatch):
    monkeypatch.setattr(nodes, "HuggingFaceEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(nodes, "Chroma", FakeStore)
    result = nodes.retrieve({"question": "What is Aadhya Patel's salary?", "access_level": "hr"})
    assert len(result["context"]) == 1
    assert "Aadhya Patel" in result["context"][0].page_content
    assert "Aadhya Saxena" not in result["context"][0].page_content


def test_hr_can_retrieve_engineering_documents(monkeypatch):
    monkeypatch.setattr(nodes, "HuggingFaceEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(nodes, "Chroma", FakeStore)
    result = nodes.retrieve({"question": "What is engineering?", "access_level": "hr"})
    assert len(result["context"]) == 1
    assert result["context"][0].metadata["access_level"] == "engineering"


def test_engineering_cannot_retrieve_hr_documents(monkeypatch):
    class HrOnlyStore(FakeStore):
        def similarity_search(self, question, k, filter):
            return [make_document("confidential HR information", "hr")]

    monkeypatch.setattr(nodes, "HuggingFaceEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(nodes, "Chroma", HrOnlyStore)
    result = nodes.retrieve({"question": "What is HR?", "access_level": "engineering"})
    assert result["context"] == []


def test_empty_context_does_not_call_the_model():
    result = nodes.generate({"question": "unknown employee", "access_level": "hr", "context": []})
    assert result["answer"]["sources"] == []
    assert "authorized information" in result["answer"]["answer"]


def test_generation_reports_gemini_quota_exhaustion(monkeypatch):
    class QuotaLimitedModel:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            raise ResourceExhausted("daily quota exhausted")

    monkeypatch.setattr(nodes, "init_chat_model", lambda *args, **kwargs: QuotaLimitedModel())
    result = nodes.generate({
        "question": "What is the deployment codename?",
        "context": [make_document("The codename is ORBIT-731.", "engineering")],
    })
    assert "quota is currently exhausted" in result["answer"]["answer"]
    assert result["answer"]["sources"] == []
