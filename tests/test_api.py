import logging
from fastapi.testclient import TestClient
from src.models.predict_model import app

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from unittest.mock import patch

@patch("src.models.predict_model.model")
def test_api_request_valid(mock_model):
    mock_model.forward.return_value = ["Das Haus ist wunderbar."]
    client = TestClient(app)
    response = client.get("/translate/The house is wonderful")

    assert response.status_code == 200
    assert response.json() == {
        "en": "The house is wonderful",
        "de translation": "Das Haus ist wunderbar."
    }
