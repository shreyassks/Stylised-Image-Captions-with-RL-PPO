from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_case_3(test_body, test_response):
    response = client.post(
        "/v1/generate_captions",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json"},
        json=test_body,
    )
    assert response.status_code == 200
    result_response = response.json()
    assert result_response["source_id"] == test_response["source_id"]
    assert len(result_response["images"][0]["generated_captions"]) == len(
        test_response["images"][0]["generated_captions"])
