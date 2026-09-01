"""
Test suite for the SIH26093 Stress & Trauma Assessment API.

Note: the previously uploaded test_app.py tested a generic
telemetry/audit/dispatch API that had nothing to do with this project's
actual purpose (text/voice stress & trauma risk assessment). These tests
replace it with checks against the real endpoints.
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["problem_id"] == "SIH26093"
    assert data["status"] == "OPERATIONAL"
    assert "timestamp" in data


def test_text_assessment_high_distress():
    response = client.post(
        "/api/v1/assess/text",
        json={"text": "He said he will kill me, I am terrified and can't sleep"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["ps_id"] == "SIH26093"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_label"] == "HIGH_RISK"
    assert len(data["text_analysis"]["matched_keywords"]) > 0
    assert "sha256_hash" in data


def test_text_assessment_low_distress():
    response = client.post(
        "/api/v1/assess/text",
        json={"text": "I am going to the market to buy vegetables today"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["risk_label"] == "LOW_RISK"


def test_text_assessment_hindi_keywords():
    response = client.post(
        "/api/v1/assess/text",
        json={"text": "मुझे बचाओ, मार डालूंगा उसने कहा"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["risk_label"] == "HIGH_RISK"


def test_text_assessment_rejects_empty():
    response = client.post("/api/v1/assess/text", json={"text": "   "})
    assert response.status_code == 400


def test_audit_logs():
    response = client.get("/api/v1/audit/logs")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert isinstance(data["records"], list)


def test_system_status():
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["problem_id"] == "SIH26093"
    assert "components" in data
    assert "keyword_rules" in data["components"]
