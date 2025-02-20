import os
import pytest
from click.testing import CliRunner
from cli import mh, amh

# Safely ignore the pylint error: Redefining name 'runner' from outer scope
# pylint: disable=redefined-outer-name


@pytest.fixture
def runner():
    """Fixture that creates a CLI runner."""
    return CliRunner()


def test_mh_basic(runner):
    """Test basic Metropolis-Hastings command."""
    with runner.isolated_filesystem():
        result = runner.invoke(mh, ["--iterations", "100", "--no-plot"])
        assert result.exit_code == 0
        assert "Running Metropolis-Hastings sampler" in result.output
        assert "Time taken:" in result.output
        assert "Acceptance rate:" in result.output


def test_amh_basic(runner):
    """Test basic Adaptive Metropolis-Hastings command."""
    with runner.isolated_filesystem():
        result = runner.invoke(amh, ["--iterations", "100", "--no-plot"])
        assert result.exit_code == 0
        assert "Running Adaptive Metropolis-Hastings sampler" in result.output
        assert "Time taken:" in result.output
        assert "Acceptance rate:" in result.output


def test_mh_with_custom_expression(runner):
    """Test MH with custom target distribution."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            mh,
            [
                "-e",
                "exp(-0.5 * (x - 2)**2) / sqrt(2 * pi)",
                "--iterations",
                "100",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0


def test_save_samples(runner):
    """Test saving samples to file."""
    with runner.isolated_filesystem():
        # Create output directories
        os.makedirs("output/samples", exist_ok=True)

        result = runner.invoke(
            mh, ["--iterations", "100", "--no-plot", "--save", "-o", "test_samples.txt"]
        )
        assert result.exit_code == 0
        assert os.path.exists("output/samples/test_samples.txt")


def test_plot_generation(runner):
    """Test plot generation."""
    with runner.isolated_filesystem():
        # Create output directories
        os.makedirs("output/plots", exist_ok=True)

        result = runner.invoke(mh, ["--iterations", "100"])
        assert result.exit_code == 0
        # Check if at least one plot file was created
        assert len(os.listdir("output/plots")) > 0


def test_adaptive_parameters(runner):
    """Test adaptive MH with custom parameters."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            amh,
            [
                "--iterations",
                "100",
                "--initial-variance",
                "2.0",
                "--check-interval",
                "20",
                "--increase-factor",
                "1.2",
                "--decrease-factor",
                "0.8",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0


def test_burn_in_and_thin(runner):
    """Test burn-in and thinning parameters."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            mh, ["--iterations", "100", "--burn-in", "20", "--thin", "2", "--no-plot"]
        )
        assert result.exit_code == 0


def test_mh_statistics_output(runner):
    """Test that MH outputs statistical summaries."""
    with runner.isolated_filesystem():
        result = runner.invoke(mh, ["--iterations", "100", "--no-plot"])
        assert result.exit_code == 0
        assert "Sample mean:" in result.output
        assert "Sample median:" in result.output
        assert "95% Credible interval:" in result.output


def test_amh_statistics_output(runner):
    """Test that AMH outputs statistical summaries."""
    with runner.isolated_filesystem():
        result = runner.invoke(amh, ["--iterations", "100", "--no-plot"])
        assert result.exit_code == 0
        assert "Sample mean:" in result.output
        assert "Sample median:" in result.output
        assert "95% Credible interval:" in result.output


def test_custom_credible_interval(runner):
    """Test custom credible interval level."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            mh, ["--iterations", "100", "--no-plot", "--credible-interval", "0.99"]
        )
        assert result.exit_code == 0
        assert "99% Credible interval:" in result.output


def test_invalid_credible_interval(runner):
    """Test error handling for invalid credible interval."""
    with runner.isolated_filesystem():
        # Test value greater than 1
        result = runner.invoke(
            mh, ["--iterations", "100", "--credible-interval", "1.5"]
        )
        assert (
            result.exit_code == 2
        )  # Click uses exit code 2 for parameter validation errors
        assert (
            "Error: Invalid value for '--credible-interval': Credible interval must be between 0 and 1"
            in result.output
        )

        # Test negative value
        result = runner.invoke(
            mh, ["--iterations", "100", "--credible-interval", "-0.5"]
        )
        assert result.exit_code == 2
        assert (
            "Error: Invalid value for '--credible-interval': Credible interval must be between 0 and 1"
            in result.output
        )

        # Test value of zero
        result = runner.invoke(mh, ["--iterations", "100", "--credible-interval", "0"])
        assert result.exit_code == 2
        assert (
            "Error: Invalid value for '--credible-interval': Credible interval must be between 0 and 1"
            in result.output
        )

        # Test value of one
        result = runner.invoke(mh, ["--iterations", "100", "--credible-interval", "1"])
        assert result.exit_code == 2
        assert (
            "Error: Invalid value for '--credible-interval': Credible interval must be between 0 and 1"
            in result.output
        )


def test_seed_reproducibility_statistics(runner):
    """Test that using same seed produces same statistics."""
    with runner.isolated_filesystem():
        result1 = runner.invoke(
            mh, ["--iterations", "100", "--seed", "42", "--no-plot"]
        )
        result2 = runner.invoke(
            mh, ["--iterations", "100", "--seed", "42", "--no-plot"]
        )

        # Extract statistics from output
        mean1 = [line for line in result1.output.split("\n") if "Sample mean:" in line][
            0
        ]
        mean2 = [line for line in result2.output.split("\n") if "Sample mean:" in line][
            0
        ]
        assert mean1 == mean2

        median1 = [
            line for line in result1.output.split("\n") if "Sample median:" in line
        ][0]
        median2 = [
            line for line in result2.output.split("\n") if "Sample median:" in line
        ][0]
        assert median1 == median2

        ci1 = [
            line for line in result1.output.split("\n") if "Credible interval:" in line
        ][0]
        ci2 = [
            line for line in result2.output.split("\n") if "Credible interval:" in line
        ][0]
        assert ci1 == ci2
