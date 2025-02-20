import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os
import time
import click
import numpy as np
import matplotlib.pyplot as plt
from library.mcmc_utils import target_distribution, proposal_distribution
from library.mcmc_algorithms import metropolis_hastings, adaptive_metropolis_hastings


def validate_credible_interval(_ctx, _param, value):
    """
    Validate credible interval is between 0 and 1 exclusive.

    Args:
        _ctx: Click context (unused)
        _param: Click parameter (unused)
        value: The credible interval value to validate

    Returns:
        float: The validated credible interval value

    Raises:
        click.BadParameter: If value is not between 0 and 1 or if value is invalid
    """
    try:
        if value <= 0 or value >= 1:
            raise click.BadParameter("Credible interval must be between 0 and 1")
        return value
    except ValueError as exc:
        raise click.BadParameter("Credible interval must be a valid number") from exc


@click.group()
def cli():
    """MCMC sampling command line interface."""


@cli.command()
@click.option(
    "--expression",
    "-e",
    default=None,
    help="Mathematical expression for target distribution. Default is standard normal.",
)
@click.option(
    "--initial", "-i", default=0.0, type=float, help="Initial value to start the chain."
)
@click.option(
    "--iterations", "-n", default=10000, type=int, help="Number of iterations to run."
)
@click.option(
    "--burn-in",
    "-b",
    default=1000,
    type=int,
    help="Number of initial samples to discard.",
)
@click.option("--thin", "-t", default=1, type=int, help="Keep every nth sample.")
@click.option(
    "--seed", "-s", default=None, type=int, help="Random seed for reproducibility."
)
@click.option("--plot/--no-plot", default=True, help="Whether to display plots.")
@click.option(
    "--save/--no-save", default=False, help="Whether to save the samples to a file."
)
@click.option(
    "--output", "-o", default="samples.txt", help="Output file name for saving samples."
)
@click.option(
    "--credible-interval",
    default=0.95,
    type=float,
    help="Credible interval level (0 to 1).",
    callback=validate_credible_interval,
)
def mh(
    expression,
    initial,
    iterations,
    burn_in,
    thin,
    seed,
    plot,
    save,
    output,
    credible_interval,
):
    """Run standard Metropolis-Hastings MCMC sampler."""
    try:
        target_dist = target_distribution(expression)

        click.echo("Running Metropolis-Hastings sampler...")
        samples, elapsed_time, acceptance_rate, mean, median, ci = metropolis_hastings(
            target_dist,
            proposal_distribution,
            initial,
            iterations,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            credible_interval=credible_interval,
        )

        process_results(
            samples,
            elapsed_time,
            acceptance_rate,
            target_dist,
            plot,
            save,
            output,
            mean=mean,
            median=median,
            credible_interval=ci,
            ci_level=credible_interval,
        )
        return 0

    except (ValueError, TypeError, SyntaxError) as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1
    except (RuntimeError, OverflowError, ZeroDivisionError) as e:
        click.echo(f"Error: Computation failed - {str(e)}", err=True)
        return 1
    except MemoryError as e:
        click.echo("Error: Not enough memory to complete operation", err=True)
        return 1


@cli.command()
@click.option(
    "--expression",
    "-e",
    default=None,
    help="Mathematical expression for target distribution. Default is standard normal.",
)
@click.option(
    "--initial", "-i", default=0.0, type=float, help="Initial value to start the chain."
)
@click.option(
    "--iterations", "-n", default=10000, type=int, help="Number of iterations to run."
)
@click.option(
    "--initial-variance", default=1.0, type=float, help="Initial proposal variance."
)
@click.option(
    "--check-interval",
    default=200,
    type=int,
    help="Interval for checking acceptance rate.",
)
@click.option(
    "--increase-factor", default=1.1, type=float, help="Factor to increase variance."
)
@click.option(
    "--decrease-factor", default=0.9, type=float, help="Factor to decrease variance."
)
@click.option(
    "--burn-in",
    "-b",
    default=1000,
    type=int,
    help="Number of initial samples to discard.",
)
@click.option("--thin", "-t", default=1, type=int, help="Keep every nth sample.")
@click.option(
    "--seed", "-s", default=None, type=int, help="Random seed for reproducibility."
)
@click.option("--plot/--no-plot", default=True, help="Whether to display plots.")
@click.option(
    "--save/--no-save", default=False, help="Whether to save the samples to a file."
)
@click.option(
    "--output", "-o", default="samples.txt", help="Output file name for saving samples."
)
@click.option(
    "--credible-interval",
    default=0.95,
    type=float,
    help="Credible interval level (0 to 1).",
    callback=validate_credible_interval,
)
def amh(
    expression,
    initial,
    iterations,
    initial_variance,
    check_interval,
    increase_factor,
    decrease_factor,
    burn_in,
    thin,
    seed,
    plot,
    save,
    output,
    credible_interval,
):
    """Run adaptive Metropolis-Hastings MCMC sampler."""
    try:
        target_dist = target_distribution(expression)

        click.echo("Running Adaptive Metropolis-Hastings sampler...")
        samples, elapsed_time, acceptance_rate, acceptance_rates, mean, median, ci = (
            adaptive_metropolis_hastings(
                target_dist,
                initial,
                iterations,
                initial_variance=initial_variance,
                check_interval=check_interval,
                increase_factor=increase_factor,
                decrease_factor=decrease_factor,
                burn_in=burn_in,
                thin=thin,
                seed=seed,
                credible_interval=credible_interval,
            )
        )

        process_results(
            samples,
            elapsed_time,
            acceptance_rate,
            target_dist,
            plot,
            save,
            output,
            acceptance_rates=acceptance_rates,
            mean=mean,
            median=median,
            credible_interval=ci,
            ci_level=credible_interval,
        )
        return 0

    except (ValueError, TypeError, SyntaxError) as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1
    except (RuntimeError, OverflowError, ZeroDivisionError) as e:
        click.echo(f"Error: Computation failed - {str(e)}", err=True)
        return 1
    except MemoryError as e:
        click.echo("Error: Not enough memory to complete operation", err=True)
        return 1


def process_results(
    samples,
    elapsed_time,
    acceptance_rate,
    target_dist,
    plot,
    save,
    output,
    acceptance_rates=None,
    mean=None,
    median=None,
    credible_interval=None,
    ci_level=0.95,
):
    """Process and display MCMC results."""

    click.echo(f"Time taken: {elapsed_time:.2f} seconds")
    click.echo(f"Acceptance rate: {acceptance_rate:.2f}")
    click.echo(f"Number of samples: {len(samples)}")

    if mean is not None:
        click.echo(f"Sample mean: {mean:.4f}")
    if median is not None:
        click.echo(f"Sample median: {median:.4f}")
    if credible_interval is not None:
        ci_lower, ci_upper = credible_interval
        ci_level_percent = int(ci_level * 100)
        click.echo(
            f"Sample {ci_level_percent}% Credible interval: ({ci_lower:.4f}, {ci_upper:.4f})"
        )

    # Create output directories if they don't exist
    output_dir = "output"
    plots_dir = os.path.join(output_dir, "plots")
    samples_dir = os.path.join(output_dir, "samples")

    for directory in [output_dir, plots_dir, samples_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if save:
        # Save samples to the samples directory
        sample_path = os.path.join(samples_dir, output)
        np.savetxt(sample_path, samples)
        click.echo(f"Samples saved to {sample_path}")

    if plot:
        n_plots = 3 if acceptance_rates is not None else 2
        _, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        # Trace plot
        axes[0].plot(samples, color="blue")
        axes[0].set_title("Trace Plot")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Sample Value")

        # Histogram
        axes[1].hist(samples, bins=50, density=True, alpha=0.6, color="g")
        x = np.linspace(min(samples), max(samples), 1000)
        axes[1].plot(x, [target_dist(xi) for xi in x], "r", lw=2)
        axes[1].set_title("Histogram of MCMC samples and target distribution")
        axes[1].set_xlabel("Sample Value")
        axes[1].set_ylabel("Density")

        # Acceptance rates (adaptive only)
        if acceptance_rates is not None:
            axes[2].plot(acceptance_rates, color="purple")
            axes[2].set_title("Acceptance Rate Over Time")
            axes[2].set_xlabel("Check Interval")
            axes[2].set_ylabel("Acceptance Rate")

        plt.tight_layout()

        # Save plot with timestamp in plots directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = f"mcmc_plots_{timestamp}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path)
        click.echo(f"Plots saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    cli()
