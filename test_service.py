import pytest
import requests
import subprocess
import time

# URL of the FastAPI service
url = 'http://127.0.0.1:8000/predict'

@pytest.fixture(scope="session", autouse=True)
def start_server():
    # Start the FastAPI server
    process = subprocess.Popen(["uvicorn", "app:app", "--reload"])
    time.sleep(5)  # wait for the server to start
    yield
    process.terminate()

def test_predict():
    # Example data similar to the training data
    data = {"text": "EPI = Echo planar imaging ."}

    # Send the POST request to the FastAPI service
    response = requests.post(url, json=data)

    # Assert the response status code and content
    assert response.status_code == 200
    result = response.json()
    print(result)
    assert "predictions" in result
    assert len(result["predictions"]) > 0

