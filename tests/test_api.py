import logging
from fastapi.testclient import TestClient
from src.models.predict_model import app

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api_request_valid():
    client = TestClient(app)
    response = client.get("/translate/The house is wonderful")

    # Log the response for debugging
    logger.debug(f"API Response Status Code: {response.status_code}")
    logger.debug(f"API Response JSON: {response.json()}")

    # Assertions
    assert response.status_code == 200, "API request failed."
    assert response.json()["en"] == "The house is wonderful", "English translation mismatch."

    german_translation = response.json().get("de translation", None)
    assert german_translation is not None, "Missing 'de translation' in response."
    assert german_translation == "Das Haus ist wunderbar.", (
        f"German translation mismatch. Expected 'Das Haus ist wunderbar.' "
        f"but got '{german_translation}'."
    )


def test_api_request_empty_str():
    client = TestClient(app)
    response = client.get("/translate/")

    # Log the response for debugging
    logger.debug(f"API Response Status Code (Empty String): {response.status_code}")
    logger.debug(f"API Response Text (Empty String): {response.text}")

    # Assertions
    assert response.status_code == 404, "Empty string request should return a 404 error."
