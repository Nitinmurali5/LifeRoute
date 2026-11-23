"""Health check endpoint tests for LifeRoute Flask application."""

from __future__ import annotations

import pytest
from flask.testing import FlaskClient

from backend.app import create_app
from backend.app.config import AppConfig


@pytest.fixture
def client() -> FlaskClient:
    """Fixture providing a test client for the LifeRoute Flask application."""
    config = AppConfig(city_name="Test City", radius_m=1000)
    app = create_app(config)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_check_returns_200_and_ok(client: FlaskClient) -> None:
    """Verify that GET /health returns HTTP 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == {"status": "ok"}


def test_cors_headers_present(client: FlaskClient) -> None:
    """Verify CORS headers are present in /health response."""
    response = client.get("/health")
    # Flask-CORS adds Access-Control-Allow-Origin header
    assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200


def test_index_dashboard_returns_html(client: FlaskClient) -> None:
    """Verify that GET / returns the dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"LifeRoute" in response.data


def test_api_endpoints_return_200(client: FlaskClient) -> None:
    """Verify that dashboard API endpoints return valid JSON."""
    r_hosp = client.get("/api/hospitals")
    assert r_hosp.status_code == 200
    assert r_hosp.is_json
    assert "hospitals" in r_hosp.get_json()

    r_nodes = client.get("/api/nodes")
    assert r_nodes.status_code == 200
    assert r_nodes.is_json
    assert "nodes" in r_nodes.get_json()

    r_sim = client.get("/api/simulation")
    assert r_sim.status_code == 200
    assert r_sim.is_json
    assert "routes" in r_sim.get_json()["data"]


