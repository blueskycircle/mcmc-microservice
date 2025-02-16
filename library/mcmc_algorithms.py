import numpy as np
from tqdm import tqdm
import time


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
