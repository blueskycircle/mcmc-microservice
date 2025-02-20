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
            "seed": 42,
            "credible_interval": 0.95,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "elapsed_time" in data
    assert "acceptance_rate" in data
    assert "mean" in data
    assert "median" in data
    assert "credible_interval" in data
    assert len(data["samples"]) > 0
    assert 0 <= data["acceptance_rate"] <= 1
    assert isinstance(data["mean"], float)
    assert isinstance(data["median"], float)
    assert isinstance(data["credible_interval"], list)
    assert len(data["credible_interval"]) == 2
    assert data["credible_interval"][0] < data["credible_interval"][1]


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
            "seed": 42,
            "credible_interval": 0.95,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "elapsed_time" in data
    assert "acceptance_rate" in data
    assert "acceptance_rates" in data
    assert "mean" in data
    assert "median" in data
    assert "credible_interval" in data
    assert len(data["samples"]) > 0
    assert len(data["acceptance_rates"]) > 0
    assert 0 <= data["acceptance_rate"] <= 1
    assert isinstance(data["mean"], float)
    assert isinstance(data["median"], float)
    assert isinstance(data["credible_interval"], list)
    assert len(data["credible_interval"]) == 2
    assert data["credible_interval"][0] < data["credible_interval"][1]


def test_seed_reproducibility_statistics():
    """Test that using the same seed produces the same statistical results."""
    response1 = client.post(
        "/mcmc/mh", json={"iterations": 100, "seed": 42, "credible_interval": 0.95}
    )
    response2 = client.post(
        "/mcmc/mh", json={"iterations": 100, "seed": 42, "credible_interval": 0.95}
    )

    data1 = response1.json()
    data2 = response2.json()

    assert data1["samples"] == data2["samples"]
    assert data1["mean"] == data2["mean"]
    assert data1["median"] == data2["median"]
    assert data1["credible_interval"] == data2["credible_interval"]


def test_invalid_credible_interval():
    """Test error handling for invalid credible interval values."""
    # Test value greater than 1
    response = client.post(
        "/mcmc/mh", json={"iterations": 100, "credible_interval": 1.5}
    )
    assert response.status_code == 422  # Pydantic validation error code
    error_data = response.json()
    assert "credible_interval" in str(error_data["detail"]).lower()

    # Test value less than or equal to 0
    response = client.post("/mcmc/mh", json={"iterations": 100, "credible_interval": 0})
    assert response.status_code == 422
    error_data = response.json()
    assert "credible_interval" in str(error_data["detail"]).lower()

    # Test value equal to 1
    response = client.post("/mcmc/mh", json={"iterations": 100, "credible_interval": 1})
    assert response.status_code == 422
    error_data = response.json()
    assert "credible_interval" in str(error_data["detail"]).lower()


def test_different_credible_intervals():
    """Test different credible interval levels produce different bounds."""
    response95 = client.post(
        "/mcmc/mh", json={"iterations": 1000, "seed": 42, "credible_interval": 0.95}
    )
    response99 = client.post(
        "/mcmc/mh", json={"iterations": 1000, "seed": 42, "credible_interval": 0.99}
    )

    ci95 = response95.json()["credible_interval"]
    ci99 = response99.json()["credible_interval"]

    # 99% CI should be wider than 95% CI
    assert ci99[1] - ci99[0] > ci95[1] - ci95[0]
