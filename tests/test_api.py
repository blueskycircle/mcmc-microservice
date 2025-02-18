from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_mh_endpoint_default_distribution():
    """Test Metropolis-Hastings endpoint with default distribution."""
    response = client.post(
        "/mcmc/mh",
        json={
            "iterations": 100,
            "burn_in": 10,
            "thin": 1,
            "seed": 42
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "elapsed_time" in data
    assert "acceptance_rate" in data
    assert len(data["samples"]) > 0
    assert 0 <= data["acceptance_rate"] <= 1

def test_amh_endpoint_default_distribution():
    """Test Adaptive Metropolis-Hastings endpoint with default distribution."""
    response = client.post(
        "/mcmc/amh",
        json={
            "iterations": 100,
            "burn_in": 10,
            "thin": 1,
            "initial_variance": 1.0,
            "check_interval": 20,
            "seed": 42
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "elapsed_time" in data
    assert "acceptance_rate" in data
    assert "acceptance_rates" in data
    assert len(data["samples"]) > 0
    assert len(data["acceptance_rates"]) > 0
    assert 0 <= data["acceptance_rate"] <= 1

def test_mh_endpoint_custom_distribution():
    """Test Metropolis-Hastings endpoint with custom distribution."""
    response = client.post(
        "/mcmc/mh",
        json={
            "expression": "exp(-0.5 * (x - 2)**2)",
            "iterations": 100,
            "burn_in": 10,
            "seed": 42
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["samples"]) > 0

def test_seed_reproducibility():
    """Test that using the same seed produces the same results."""
    response1 = client.post(
        "/mcmc/mh",
        json={
            "iterations": 100,
            "seed": 42
        }
    )
    response2 = client.post(
        "/mcmc/mh",
        json={
            "iterations": 100,
            "seed": 42
        }
    )
    assert response1.json()["samples"] == response2.json()["samples"]