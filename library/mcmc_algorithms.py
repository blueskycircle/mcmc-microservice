import numpy as np
from tqdm import tqdm
import time


def adaptive_metropolis_hastings(
    target,
    initial,
    iterations,
    initial_variance=1.0,
    check_interval=200,
    increase_factor=1.1,
    decrease_factor=0.9,
    burn_in=1000,
    thin=1,
    seed=None,
    credible_interval=0.95,
):
    """
    Adaptive Metropolis-Hastings algorithm with burn-in and thinning.

    Args:
        target (Callable[[float], float]): Target distribution function that takes a float and returns a float
        initial (float): Initial value to start the chain
        iterations (int): Number of iterations to run
        initial_variance (float, optional): Initial proposal variance. Defaults to 1.0
        check_interval (int, optional): Interval for checking acceptance rate. Defaults to 200
        increase_factor (float, optional): Factor to increase variance. Defaults to 1.1
        decrease_factor (float, optional): Factor to decrease variance. Defaults to 0.9
        burn_in (int, optional): Number of initial samples to discard. Defaults to 1000
        thin (int, optional): Keep every nth sample. Defaults to 1
        seed (int, optional): Random seed for reproducibility. Defaults to None
        credible_interval (float, optional): Credible interval level (0 to 1). Defaults to 0.95


    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Array of samples from the target distribution
            - float: Elapsed time in seconds
            - float: Overall acceptance rate between 0 and 1
            - list[float]: List of acceptance rates at each check interval
            - float: Mean of the samples
            - float: Median of the samples
            - tuple: Credible interval (lower, upper) bounds

    Example:
        >>> target_dist = target_distribution('exp(-0.5 * x**2) / sqrt(2 * pi)')
        >>> samples, time, acc_rate, acc_rates = adaptive_metropolis_hastings(target_dist, 0.0, 10000, seed=42)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    total_iterations = iterations + burn_in
    samples = []
    current = initial
    variance = initial_variance
    acceptance_rates = []
    interval_accepted = 0
    interval_count = 0
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Sampling", unit="iteration") as pbar:
        for i in range(total_iterations):
            # Propose new value
            proposed = np.random.normal(current, np.sqrt(variance))
            acceptance_ratio = target(proposed) / target(current)

            if np.random.rand() < acceptance_ratio:
                current = proposed
                interval_accepted += 1

            # Store sample if past burn-in and meets thinning criteria
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current)

            # Check acceptance rate at each interval
            interval_count += 1
            if interval_count == check_interval:
                acceptance_rate = interval_accepted / check_interval
                variance = adaptive_proposal_distribution(
                    variance, acceptance_rate, increase_factor, decrease_factor
                )
                acceptance_rates.append(acceptance_rate)
                interval_accepted = 0
                interval_count = 0

            pbar.update(1)
            current_acceptance = interval_accepted / max(1, interval_count)
            pbar.set_postfix(acceptance_rate=current_acceptance)

    end_time = time.time()
    elapsed_time = end_time - start_time
    overall_acceptance_rate = np.mean(acceptance_rates) if acceptance_rates else 0

    samples_array = np.array(samples)
    sample_mean = np.mean(samples_array)
    sample_median = np.median(samples_array)

    # Calculate credible interval
    alpha = (1 - credible_interval) / 2
    ci_lower = np.percentile(samples_array, 100 * alpha)
    ci_upper = np.percentile(samples_array, 100 * (1 - alpha))

    return (
        samples_array,
        elapsed_time,
        overall_acceptance_rate,
        acceptance_rates,
        sample_mean,
        sample_median,
        (ci_lower, ci_upper),
    )


def adaptive_proposal_distribution(
    variance, acceptance_rate, increase_factor=1.1, decrease_factor=0.9
):
    if acceptance_rate > 0.5:
        variance *= increase_factor
    elif acceptance_rate < 0.3:
        variance *= decrease_factor
    return variance


def metropolis_hastings(
    target,
    proposal,
    initial,
    iterations,
    burn_in=1000,
    thin=1,
    seed=None,
    credible_interval=0.95,
):
    """
    Metropolis-Hastings algorithm with burn-in and thinning.

    Args:
        target (Callable[[float], float]): Target distribution function that takes a float and returns a float
        proposal (Callable[[float], float]): Proposal distribution function that takes a float and returns a float
        initial (float): Initial value to start the chain
        iterations (int): Number of iterations to run
        burn_in (int, optional): Number of initial samples to discard. Defaults to 1000
        thin (int, optional): Keep every nth sample. Defaults to 1
        seed (int, optional): Random seed for reproducibility. Defaults to None
        credible_interval (float, optional): Credible interval level (0 to 1). Defaults to 0.95

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Array of samples from the target distribution
            - float: Elapsed time in seconds
            - float: Acceptance rate between 0 and 1
            - float: Mean of the samples
            - float: Median of the
            - tuple: Credible interval (lower, upper) bounds

    Example:
        >>> target_dist = target_distribution('exp(-0.5 * x**2) / sqrt(2 * pi)')
        >>> samples, time, acc_rate = metropolis_hastings(target_dist, proposal_distribution, 0.0, 10000, seed=42)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    total_iterations = iterations + burn_in
    samples = []
    current = initial
    accepted = 0
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Sampling", unit="iteration") as pbar:
        for i in range(total_iterations):
            proposed = proposal(current)
            acceptance_ratio = target(proposed) / target(current)

            if np.random.rand() < acceptance_ratio:
                current = proposed
                if i >= burn_in:  # Only count acceptance after burn-in
                    accepted += 1

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current)

            pbar.update(1)
            pbar.set_postfix(acceptance_rate=accepted / (max(1, len(samples) - 1)))

    end_time = time.time()
    elapsed_time = end_time - start_time
    acceptance_rate = accepted / iterations

    samples_array = np.array(samples)
    sample_mean = np.mean(samples_array)
    sample_median = np.median(samples_array)

    # Calculate credible interval
    alpha = (1 - credible_interval) / 2
    ci_lower = np.percentile(samples_array, 100 * alpha)
    ci_upper = np.percentile(samples_array, 100 * (1 - alpha))

    return (
        samples_array,
        elapsed_time,
        acceptance_rate,
        sample_mean,
        sample_median,
        (ci_lower, ci_upper),
    )
