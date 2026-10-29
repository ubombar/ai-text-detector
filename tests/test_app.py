"""Fast tests: no models are loaded."""
import bcrypt
import pytest
from fastapi.testclient import TestClient

from app import analysis, config
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "APP_USERNAME", "u")
    monkeypatch.setattr(config, "APP_PASSWORD_HASH", bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode())
    return TestClient(app)


def test_requires_login(client):
    assert client.get("/", follow_redirects=False).status_code == 303
    assert client.post("/api/analyze", json={"text": "x"}).status_code == 401


def test_bad_login(client):
    assert client.post("/login", data={"username": "u", "password": "no"}).status_code == 401


def test_login_and_pages(client):
    r = client.post("/login", data={"username": "u", "password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200


def test_verdict():
    assert analysis.verdict(0.5, 0.9, "lower_is_ai", 0.02) == "ai"
    assert analysis.verdict(1.2, 0.9, "lower_is_ai", 0.02) == "human"
    assert analysis.verdict(0.89, 0.9, "lower_is_ai", 0.02) == "uncertain"
    assert analysis.verdict(3.0, 2.0, "higher_is_ai", 0.3) == "ai"


def test_overall_needs_two_votes(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_LLM_PAIR", True)
    assert analysis.overall({"binoculars": "ai", "fast_detectgpt": "ai", "classifier": "human"})["verdict"] == "ai"
    assert analysis.overall({"binoculars": "ai", "fast_detectgpt": "uncertain", "classifier": "human"})["verdict"] == "inconclusive"


def test_overall_classifier_only(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_LLM_PAIR", False)
    assert analysis.enabled_methods() == ["classifier", "gltr"]
    assert analysis.overall({"classifier": "ai", "gltr": "human"})["verdict"] == "ai"
