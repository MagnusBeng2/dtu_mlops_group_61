import logging
from fastapi.testclient import TestClient

from src.models.predict_model import app


def test_api_request_valid():
    client = TestClient(app)
    response = client.get("/translate/The house is wonderful")
    assert response.status_code == 200, "API request failed."
    assert response.json() == {
        "en": "The house is wonderful",
        "de translation": "Das Haus ist wunderbar.",
    }, "API response is not as expected." 


def test_api_request_empty_str():
    client = TestClient(app)
    response = client.get("/translate/")
    assert response.status_code == 404
