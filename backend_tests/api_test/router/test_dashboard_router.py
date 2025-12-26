from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.api.routers.dashboard_router import router


def create_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_dashboard_stats_hospital(monkeypatch):
    app = create_test_app()
    client = TestClient(app)

    # -------------------------
    # Mock service layer
    # -------------------------
    def fake_service(**kwargs):
        return {"metrics": {"totalIncidents": 10}}

    monkeypatch.setattr(
        "routers.dashboard_router.get_dashboard_stats",
        fake_service,
    )

    response = client.get("/api/dashboard/stats?scope=hospital")

    assert response.status_code == 200
    assert response.json()["metrics"]["totalIncidents"] == 10


def test_dashboard_stats_missing_admin_id():
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/api/dashboard/stats?scope=administration")

    assert response.status_code == 400
    assert "administration_id required" in response.text


def test_dashboard_stats_invalid_scope():
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/api/dashboard/stats?scope=invalid")

    assert response.status_code == 400


def test_dashboard_hierarchy(monkeypatch):
    app = create_test_app()
    client = TestClient(app)

    monkeypatch.setattr(
        "routers.dashboard_router.get_dashboard_hierarchy",
        lambda: {"idarat": []},
    )

    response = client.get("/api/dashboard/hierarchy")

    assert response.status_code == 200
    assert "idarat" in response.json()
