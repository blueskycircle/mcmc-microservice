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
):
    samples = [initial]
    current = initial
    accepted = 0
    variance = initial_variance
    acceptance_rates = []
    start_time = time.time()

    with tqdm(total=iterations, desc="Sampling", unit="iteration") as pbar:
        for i in range(iterations):
            proposed = np.random.normal(current, np.sqrt(variance))
            acceptance_ratio = target(proposed) / target(current)

            if np.random.rand() < acceptance_ratio:
                current = proposed
                accepted += 1

            samples.append(current)
            pbar.update(1)
            pbar.set_postfix(acceptance_rate=accepted / (len(samples) - 1))

            # Adjust variance every check_interval iterations
            if (i + 1) % check_interval == 0:
                acceptance_rate = accepted / check_interval
                variance = adaptive_proposal_distribution(
                    variance, acceptance_rate, increase_factor, decrease_factor
                )
                acceptance_rates.append(acceptance_rate)
                accepted = 0  # Reset accepted count for the next interval

    end_time = time.time()
    elapsed_time = end_time - start_time
    overall_acceptance_rate = np.mean(acceptance_rates)

    return np.array(samples), elapsed_time, overall_acceptance_rate, acceptance_rates


def adaptive_proposal_distribution(
    variance, acceptance_rate, increase_factor=1.1, decrease_factor=0.9
):
    if acceptance_rate > 0.5:
        variance *= increase_factor
    elif acceptance_rate < 0.3:
        variance *= decrease_factor
    return variance


def metropolis_hastings(target, proposal, initial, iterations):
    samples = [initial]
    current = initial
    accepted = 0
    start_time = time.time()

    with tqdm(total=iterations, desc="Sampling", unit="iteration") as pbar:
        for _ in range(iterations):
            proposed = proposal(current)
            acceptance_ratio = target(proposed) / target(current)

            if np.random.rand() < acceptance_ratio:
                current = proposed
                accepted += 1

            samples.append(current)
            pbar.update(1)
            pbar.set_postfix(acceptance_rate=accepted / (len(samples) - 1))

    end_time = time.time()
    elapsed_time = end_time - start_time
    acceptance_rate = accepted / iterations

    return np.array(samples), elapsed_time, acceptance_rate
