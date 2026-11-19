"""
Tests for DataSentry FastAPI endpoints.
"""

from fastapi.testclient import TestClient

from datasentry.api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "datasentry"
    assert "remediation_agent" in data["active_roles"]


def test_scan_and_explain_endpoints():
    payload = {
        "dataset_name": "test_stream",
        "current_records": [
            {"sensor_a": 12.5, "sensor_b": 100.2},
            {"sensor_a": 13.1, "sensor_b": 102.5},
            {"sensor_a": 12.9, "sensor_b": 99.8},
            {"sensor_a": 99.9, "sensor_b": 500.0},  # outlier
        ],
        "reference_records": [
            {"sensor_a": 12.0, "sensor_b": 100.0},
            {"sensor_a": 12.2, "sensor_b": 101.0},
            {"sensor_a": 13.0, "sensor_b": 100.5},
        ]
    }

    # 1. Test /scan
    scan_res = client.post("/scan", json=payload)
    assert scan_res.status_code == 200
    report = scan_res.json()
    assert report["dataset_name"] == "test_stream"
    assert "quality" in report
    assert "anomaly" in report
    assert "drift" in report

    # 2. Test /explain
    explain_res = client.post("/explain", json={"report": report, "verify_faithfulness": True})
    assert explain_res.status_code == 200
    exp_data = explain_res.json()
    assert exp_data["faithfulness_score"] == 1.0
    assert exp_data["has_hallucinations"] is False
    assert isinstance(exp_data["explanation"]["cited_finding_ids"], list)

    # 3. Test /remediate
    rem_res = client.post("/remediate", json={
        "report": report,
        "role": "remediation_agent",
        "human_approved": False,
        "dry_run": True,
    })
    assert rem_res.status_code == 200
    rem_data = rem_res.json()
    assert rem_data["allowed"] is True
    assert rem_data["executed"] is True
