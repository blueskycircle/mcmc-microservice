import numpy as np
from library.mcmc_utils import target_distribution, proposal_distribution
from library.mcmc_algorithms import metropolis_hastings, adaptive_metropolis_hastings


def test_metropolis_hastings():
    """Test Metropolis-Hastings algorithm with seed reproducibility."""
    initial_value = 0.0
    num_iterations = 1000
    burn_in = 200
    thin = 2
    seed = 42
    credible_interval = 0.95

    # Use the default target distribution (standard normal)
    target_dist = target_distribution()

    # Run sampler twice with same seed
    samples1, time1, acc_rate1, mean1, median1, ci1 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        credible_interval=credible_interval,
    )

    samples2, _, acc_rate2, mean2, median2, ci2 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        credible_interval=credible_interval,
    )

    # Run with different seed
    samples3, _, _, mean3, median3, ci3 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        burn_in=burn_in,
        thin=thin,
        seed=seed + 1,
        credible_interval=credible_interval,
    )

    # Test reproducibility with same seed
    assert np.array_equal(samples1, samples2)
    assert acc_rate1 == acc_rate2
    assert mean1 == mean2
    assert median1 == median2
    assert ci1 == ci2

    # Test different results with different seed
    assert not np.array_equal(samples1, samples3)
    assert mean1 != mean3
    assert median1 != median3
    assert ci1 != ci3

    # Statistical checks
    expected_length = num_iterations // thin
    assert len(samples1) == expected_length
    assert time1 > 0
    assert 0 < acc_rate1 < 1
    assert -1 < mean1 < 1  # Mean should be close to 0
    assert -1 < median1 < 1  # Median should be close to 0
    assert 0.5 < np.std(samples1) < 1.5  # Std should be close to 1

    # Credible interval checks
    ci_lower, ci_upper = ci1
    assert ci_lower < ci_upper
    assert -3 < ci_lower < 0  # For standard normal, 95% CI should be roughly (-2, 2)
    assert 0 < ci_upper < 3
    assert ci_upper - ci_lower < 5  # Width should be reasonable


def test_adaptive_metropolis_hastings():
    """Test Adaptive Metropolis-Hastings algorithm with seed reproducibility."""
    initial_value = 0.0
    num_iterations = 1000
    burn_in = 200
    thin = 2
    check_interval = 100
    seed = 42
    credible_interval = 0.95

    # Use the default target distribution (standard normal)
    target_dist = target_distribution()

    # Run sampler twice with same seed
    samples1, time1, acc_rate1, acc_rates1, mean1, median1, ci1 = (
        adaptive_metropolis_hastings(
            target_dist,
            initial_value,
            num_iterations,
            check_interval=check_interval,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            credible_interval=credible_interval,
        )
    )

    samples2, _, acc_rate2, acc_rates2, mean2, median2, ci2 = (
        adaptive_metropolis_hastings(
            target_dist,
            initial_value,
            num_iterations,
            check_interval=check_interval,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            credible_interval=credible_interval,
        )
    )

    # Run with different seed
    samples3, _, _, acc_rates3, mean3, median3, ci3 = adaptive_metropolis_hastings(
        target_dist,
        initial_value,
        num_iterations,
        check_interval=check_interval,
        burn_in=burn_in,
        thin=thin,
        seed=seed + 1,
        credible_interval=credible_interval,
    )

    # Test reproducibility with same seed
    assert np.array_equal(samples1, samples2)
    assert acc_rate1 == acc_rate2
    assert np.array_equal(acc_rates1, acc_rates2)
    assert mean1 == mean2
    assert median1 == median2
    assert ci1 == ci2

    # Test different results with different seed
    assert not np.array_equal(samples1, samples3)
    assert not np.array_equal(acc_rates1, acc_rates3)
    assert mean1 != mean3
    assert median1 != median3
    assert ci1 != ci3

    # Statistical checks
    expected_length = num_iterations // thin
    assert len(samples1) == expected_length
    assert time1 > 0
    assert 0 < acc_rate1 < 1
    assert -1 < mean1 < 1  # Mean should be close to 0
    assert -1 < median1 < 1  # Median should be close to 0
    assert 0.5 < np.std(samples1) < 1.5  # Std should be close to 1

    # Credible interval checks
    ci_lower, ci_upper = ci1
    assert ci_lower < ci_upper
    assert -3 < ci_lower < 0  # For standard normal, 95% CI should be roughly (-2, 2)
    assert 0 < ci_upper < 3
    assert ci_upper - ci_lower < 5  # Width should be reasonable

    # Check acceptance rates list
    expected_rate_checks = (num_iterations + burn_in) // check_interval
    assert len(acc_rates1) == expected_rate_checks
    assert all(0 <= rate <= 1 for rate in acc_rates1)


def test_different_credible_intervals():
    """Test different credible interval levels."""
    target_dist = target_distribution()
    initial_value = 0.0
    num_iterations = 5000
    seed = 42

    # Test 95% CI
    _, _, _, _, _, ci_95 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        seed=seed,
        credible_interval=0.95,
    )

    # Test 99% CI
    _, _, _, _, _, ci_99 = metropolis_hastings(
        target_dist,
        proposal_distribution,
        initial_value,
        num_iterations,
        seed=seed,
        credible_interval=0.99,
    )

    # 99% CI should be wider than 95% CI
    ci_95_width = ci_95[1] - ci_95[0]
    ci_99_width = ci_99[1] - ci_99[0]
    assert ci_99_width > ci_95_width
