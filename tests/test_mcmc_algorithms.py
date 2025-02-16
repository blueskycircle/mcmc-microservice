import numpy as np
from library.mcmc_utils import target_distribution, proposal_distribution
from library.mcmc_algorithms import metropolis_hastings


def test_metropolis_hastings():
    initial_value = 0.0
    num_iterations = 1000

    samples, elapsed_time, acceptance_rate = metropolis_hastings(
        target_distribution, proposal_distribution, initial_value, num_iterations
    )

    # Check that the returned samples array has the correct length
    assert len(samples) == num_iterations + 1

    # Check that the elapsed time is a positive number
    assert elapsed_time > 0

    # Check that the acceptance rate is between 0 and 1
    assert 0 <= acceptance_rate <= 1

    # Check that the samples are not all the same (indicating some acceptance)
    assert np.std(samples) > 0
