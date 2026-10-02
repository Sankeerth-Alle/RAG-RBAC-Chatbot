from fastapi.testclient import TestClient

import main


def test_health_is_public():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_authentication():
    response = TestClient(main.app).post("/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_document_upload_requires_authentication():
    response = TestClient(main.app).post(
        "/documents/upload",
        files={"file": ("notes.txt", b"engineering notes", "text/plain")},
    )
    assert response.status_code == 401


def test_upload_uses_authenticated_role(monkeypatch):
    indexed = {}

    def fake_index(filename, content, access_level, username):
        indexed.update(
            filename=filename,
            content=content,
            access_level=access_level,
            username=username,
        )
        return 1

    monkeypatch.setattr(main, "index_uploaded_file", fake_index)
    client = TestClient(main.app)
    login = client.post("/login", json={"username": "Tony", "password": "password123"})
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"engineering notes", "text/plain")},
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert response.status_code == 200
    assert indexed == {
        "filename": "notes.txt",
        "content": b"engineering notes",
        "access_level": "engineering",
        "username": "Tony",
    }
    assert response.json()["visible_to"] == ["engineering"]


def test_chat_uses_authenticated_user_and_clean_response(monkeypatch):
    monkeypatch.setattr(
        main.chatbot,
        "chatbot_service",
        lambda role, question: {"answer": "Engineering answer", "sources": ["engineering_master_doc.md"]},
    )
    client = TestClient(main.app)
    login = client.post("/login", json={"username": "Tony", "password": "password123"})
    token = login.json()["access_token"]
    response = client.post(
        "/chat",
        json={"message": "Tell me about engineering"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Engineering answer",
        "sources": ["engineering_master_doc.md"],
        "user": {"username": "Tony", "role": "engineering"},
    }


def test_chat_rejects_client_role():
    client = TestClient(main.app)
    login = client.post("/login", json={"username": "Tony", "password": "password123"})
    token = login.json()["access_token"]
    response = client.post(
        "/chat",
        json={"message": "hello", "role": "hr"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
