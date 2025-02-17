import numpy as np
from library.mcmc_utils import target_distribution, proposal_distribution
from library.mcmc_algorithms import metropolis_hastings, adaptive_metropolis_hastings


def test_metropolis_hastings():
    initial_value = 0.0
    num_iterations = 1000

    # Use the default target distribution (standard normal distribution)
    target_dist = target_distribution()

    samples, elapsed_time, acceptance_rate = metropolis_hastings(
        target_dist, proposal_distribution, initial_value, num_iterations
    )

    # Check that the returned samples array has the correct length
    assert len(samples) == num_iterations + 1

    # Check that the elapsed time is a positive number
    assert elapsed_time > 0

    # Check that the acceptance rate is between 0 and 1
    assert 0 < acceptance_rate < 1

    # Check that the samples are not all the same (indicating some acceptance)
    assert np.std(samples) > 0


def test_adaptive_metropolis_hastings():
    initial_value = 0.0
    num_iterations = 1000
    check_interval_num = 200

    # Use the default target distribution (standard normal distribution)
    target_dist = target_distribution()

    samples, elapsed_time, overall_acceptance_rate, acceptance_rates = (
        adaptive_metropolis_hastings(
            target_dist,
            initial_value,
            num_iterations,
            check_interval=check_interval_num,
        )
    )

    # Check that the returned samples array has the correct length
    assert len(samples) == num_iterations + 1

    # Check that the elapsed time is a positive number
    assert elapsed_time > 0

    # Check that the overall acceptance rate is between 0 and 1
    assert 0 < overall_acceptance_rate < 1

    # Check that the acceptance rates list is not empty
    assert len(acceptance_rates) == num_iterations / check_interval_num

    # Check that all acceptance rates in the list are between 0 and 1
    assert all(0 < rate < 1 for rate in acceptance_rates)

    # Check that the samples are not all the same (indicating some acceptance)
    assert np.std(samples) > 0
