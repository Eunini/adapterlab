from dataclasses import replace

import pytest
from adapterlab.api import create_app
from adapterlab.data import demo_dataset
from fastapi.testclient import TestClient


def test_dataset_registration_is_versioned_and_idempotent(client):
    initial = client.get("/api/datasets").json()
    assert len(initial) == 1
    response = client.post("/api/datasets", json=demo_dataset().model_dump())
    assert response.status_code == 201
    assert response.json()["id"] == initial[0]["id"]
    assert len(client.get("/api/datasets").json()) == 1
    assert len(client.get("/api/datasets/" + initial[0]["id"]).json()["preview"]) == 12


def test_create_cancel_and_export_guards(client):
    dataset = client.get("/api/datasets").json()[0]
    response = client.post(
        "/api/runs", json={"name": "test-run", "dataset_id": dataset["id"], "config": {"epochs": 1}}
    )
    assert response.status_code == 201
    row = response.json()
    assert row["status"] == "queued"
    assert client.get(f"/api/runs/{row['id']}/export").status_code == 409
    assert (
        client.post(f"/api/runs/{row['id']}/predict", json={"text": "please help"}).status_code
        == 409
    )
    assert client.post(f"/api/runs/{row['id']}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/runs/{row['id']}/cancel").json()["status"] == "cancelled"
    assert client.get("/api/runs/missing").status_code == 404


@pytest.mark.parametrize(
    "config",
    [{"epochs": 13}, {"rank": 3}, {"learning_rate": -1}, {"model_id": "untrusted/custom-model"}],
)
def test_invalid_configs_are_rejected(client, config):
    dataset = client.get("/api/datasets").json()[0]
    assert (
        client.post(
            "/api/runs", json={"name": "invalid", "dataset_id": dataset["id"], "config": config}
        ).status_code
        == 422
    )


def test_protected_mode_keeps_health_public_and_requires_operator_access(settings):
    token = "unit-test-operator-" + "x" * 32
    with TestClient(create_app(replace(settings, protected=True, operator_token=token))) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/datasets").status_code == 401
        assert (
            client.get("/api/datasets", headers={"Authorization": "Bearer " + token}).status_code
            == 200
        )
    with pytest.raises(ValueError, match="OPERATOR_TOKEN"):
        create_app(replace(settings, protected=True, operator_token="weak"))


def test_oversized_dataset_body_is_rejected(client):
    response = client.post(
        "/api/datasets", content=b" " * 8_000_001, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
