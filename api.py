from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from library.mcmc_utils import target_distribution
from library.mcmc_algorithms import metropolis_hastings, adaptive_metropolis_hastings
from library.mcmc_utils import proposal_distribution
from typing import List, Optional

# Default distribution (standard normal)
DEFAULT_DISTRIBUTION = "exp(-0.5 * x**2) / sqrt(2 * pi)"

app = FastAPI(
    title="MCMC Sampling API",
    description="API for Metropolis-Hastings and Adaptive Metropolis-Hastings MCMC sampling",
    version="1.0.0",
)


# Update the response models to include new statistics
class MCMCRequest(BaseModel):
    expression: Optional[str] = DEFAULT_DISTRIBUTION
    initial: float = 0.0
    iterations: int = 10000
    burn_in: int = 1000
    thin: int = 1
    seed: Optional[int] = None
    credible_interval: float = 0.95

    @field_validator("credible_interval")
    @classmethod
    def validate_credible_interval(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError("Credible interval must be between 0 and 1")
        return v


class MCMCResponse(BaseModel):
    samples: List[float]
    elapsed_time: float
    acceptance_rate: float
    mean: float
    median: float
    credible_interval: tuple[float, float]


class AdaptiveMCMCResponse(MCMCResponse):
    acceptance_rates: List[float]


class AdaptiveMCMCRequest(MCMCRequest):
    initial_variance: float = 1.0
    check_interval: int = 200
    increase_factor: float = 1.1
    decrease_factor: float = 0.9


@app.post("/mcmc/mh", response_model=MCMCResponse)
async def run_metropolis_hastings(request: MCMCRequest):
    """Run standard Metropolis-Hastings MCMC sampler."""
    try:
        target_dist = target_distribution(request.expression)

        samples, elapsed_time, acceptance_rate, mean, median, ci = metropolis_hastings(
            target_dist,
            proposal_distribution,
            request.initial,
            request.iterations,
            burn_in=request.burn_in,
            thin=request.thin,
            seed=request.seed,
            credible_interval=request.credible_interval,
        )

        return {
            "samples": samples.tolist(),
            "elapsed_time": elapsed_time,
            "acceptance_rate": acceptance_rate,
            "mean": mean,
            "median": median,
            "credible_interval": ci,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/mcmc/amh", response_model=AdaptiveMCMCResponse)
async def run_adaptive_metropolis_hastings(request: AdaptiveMCMCRequest):
    """Run adaptive Metropolis-Hastings MCMC sampler."""
    try:
        target_dist = target_distribution(request.expression)

        samples, elapsed_time, acceptance_rate, acceptance_rates, mean, median, ci = (
            adaptive_metropolis_hastings(
                target_dist,
                request.initial,
                request.iterations,
                initial_variance=request.initial_variance,
                check_interval=request.check_interval,
                increase_factor=request.increase_factor,
                decrease_factor=request.decrease_factor,
                burn_in=request.burn_in,
                thin=request.thin,
                seed=request.seed,
                credible_interval=request.credible_interval,
            )
        )

        return {
            "samples": samples.tolist(),
            "elapsed_time": elapsed_time,
            "acceptance_rate": acceptance_rate,
            "acceptance_rates": acceptance_rates,
            "mean": mean,
            "median": median,
            "credible_interval": ci,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
