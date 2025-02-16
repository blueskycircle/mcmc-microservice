import numpy as np


def target_distribution(x):
    # Example target distribution: standard normal distribution
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def proposal_distribution(x):
    # Example proposal distribution: normal distribution centered at x
    return np.random.normal(x, 1.0)
