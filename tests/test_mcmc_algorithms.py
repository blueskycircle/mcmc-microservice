import numpy as np
from library.mcmc_utils import target_distribution, proposal_distribution
from library.mcmc_algorithms import metropolis_hastings, adaptive_metropolis_hastings


def test_metropolis_hastings():
    initial_value = 0.0
    num_iterations = 1000
    burn_in = 200
    thin = 2
    seed = 42

    # Use the default target distribution (standard normal distribution)
    target_dist = target_distribution()

    # Run the sampler twice with the same seed
    samples1, elapsed_time1, acceptance_rate1 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )

    samples2, _, acceptance_rate2 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )

    # Check that using the same seed produces identical results
    assert np.array_equal(samples1, samples2)
    assert acceptance_rate1 == acceptance_rate2

    # Run with a different seed
    samples3, _, _ = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        burn_in=burn_in,
        thin=thin,
        seed=seed + 1,
    )

    # Check that different seeds produce different results
    assert not np.array_equal(samples1, samples3)

    # Original checks
    expected_length = (num_iterations // thin) + 1
    assert len(samples1) == expected_length
    assert elapsed_time1 > 0
    assert 0 < acceptance_rate1 < 1
    assert np.std(samples1) > 0
    assert abs(np.mean(samples1)) < 0.5  # Mean should be close to 0
    assert 0.5 < np.std(samples1) < 1.5  # Std should be close to 1


def test_adaptive_metropolis_hastings():
    initial_value = 0.0
    num_iterations = 1000
    burn_in = 200
    thin = 2
    check_interval = 100
    seed = 42

    # Use the default target distribution (standard normal distribution)
    target_dist = target_distribution()

    # Run the sampler twice with the same seed
    samples1, time1, acc_rate1, acc_rates1 = adaptive_metropolis_hastings(
        target_dist,
        initial_value,
        num_iterations,
        check_interval=check_interval,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )

    samples2, _, acc_rate2, acc_rates2 = adaptive_metropolis_hastings(
        target_dist,
        initial_value,
        num_iterations,
        check_interval=check_interval,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )

    # Run with a different seed
    samples3, _, _, acc_rates3 = adaptive_metropolis_hastings(
        target_dist,
        initial_value,
        num_iterations,
        check_interval=check_interval,
        burn_in=burn_in,
        thin=thin,
        seed=seed + 1,
    )

    # Check that using the same seed produces identical results
    assert np.array_equal(samples1, samples2)
    assert acc_rate1 == acc_rate2
    assert np.array_equal(acc_rates1, acc_rates2)

    # Check that different seeds produce different results
    assert not np.array_equal(samples1, samples3)
    assert not np.array_equal(acc_rates1, acc_rates3)

    # Original checks
    expected_length = (num_iterations // thin) + 1
    assert len(samples1) == expected_length
    assert time1 > 0
    assert 0 < acc_rate1 < 1
    assert np.std(samples1) > 0
    assert abs(np.mean(samples1)) < 0.5  # Mean should be close to 0
    assert 0.5 < np.std(samples1) < 1.5  # Std should be close to 1

    # Check acceptance rates list
    expected_rate_checks = (num_iterations + burn_in) // check_interval
    assert len(acc_rates1) == expected_rate_checks
    assert all(0 <= rate <= 1 for rate in acc_rates1)
