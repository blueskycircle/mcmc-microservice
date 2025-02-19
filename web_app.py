import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from library.mcmc_utils import target_distribution
from library.mcmc_algorithms import metropolis_hastings, adaptive_metropolis_hastings
from library.mcmc_utils import proposal_distribution
from time import sleep

# Set page configuration
st.set_page_config(page_title="MCMC Sampling App", page_icon="📊", layout="wide")

# Add custom CSS
st.markdown(
    """
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .css-1v0mbdj.ebxwdo61 {
        width: 100%;
        max-width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title("📊 MCMC Sampling Application")
st.markdown(
    """
This application provides an interface for running Metropolis-Hastings (MH) and 
Adaptive Metropolis-Hastings (AMH) MCMC samplers. Choose your sampler, set your parameters, 
and visualize the results!
"""
)

# Initialize AMH parameters with default values
initial_variance = 1.0
check_interval = 200
increase_factor = 1.1
decrease_factor = 0.9

# Sidebar for selecting sampler and parameters
with st.sidebar:
    st.header("Sampler Configuration")

    # Select sampler
    sampler_type = st.radio(
        "Select MCMC Sampler", ["Metropolis-Hastings", "Adaptive Metropolis-Hastings"]
    )

    # Common parameters
    st.subheader("Common Parameters")
    expression = st.text_input(
        "Target Distribution",
        value="exp(-0.5 * x**2) / sqrt(2 * pi)",
        help="Mathematical expression for the target distribution",
    )

    col1, col2 = st.columns(2)
    with col1:
        iterations = st.number_input("Iterations", min_value=100, value=10000, step=100)
        burn_in = st.number_input("Burn-in", min_value=0, value=1000, step=100)
    with col2:
        initial = st.number_input("Initial Value", value=0.0)
        thin = st.number_input("Thinning", min_value=1, value=1)

    seed = st.number_input("Random Seed", min_value=0, value=42)

    # AMH specific parameters
    if sampler_type == "Adaptive Metropolis-Hastings":
        st.subheader("AMH Parameters")
        col3, col4 = st.columns(2)
        with col3:
            initial_variance = st.number_input(
                "Initial Variance", min_value=0.1, value=1.0, step=0.1
            )
            increase_factor = st.number_input(
                "Increase Factor", min_value=1.0, value=1.1, step=0.1
            )
        with col4:
            check_interval = st.number_input(
                "Check Interval", min_value=10, value=200, step=10
            )
            decrease_factor = st.number_input(
                "Decrease Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.1
            )

# Main content
try:
    if st.button("Run Sampler", type="primary"):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize sampler
        status_text.text("Initializing sampler...")
        progress_bar.progress(10)

        # Get target distribution
        target_dist = target_distribution(expression)
        progress_bar.progress(20)

        # Run selected sampler
        if sampler_type == "Metropolis-Hastings":
            status_text.text("Running Metropolis-Hastings sampler...")
            samples, elapsed_time, acceptance_rate = metropolis_hastings(
                target_dist,
                proposal_distribution,
                initial,
                iterations,
                burn_in=burn_in,
                thin=thin,
                seed=seed,
            )
            acceptance_rates = None
        else:
            status_text.text("Running Adaptive Metropolis-Hastings sampler...")
            samples, elapsed_time, acceptance_rate, acceptance_rates = (
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
                )
            )

        progress_bar.progress(70)
        status_text.text("Generating visualizations...")

        # Display results in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Elapsed Time", f"{elapsed_time:.2f} seconds")
        with col2:
            st.metric("Acceptance Rate", f"{acceptance_rate:.2%}")
        with col3:
            st.metric("Number of Samples", len(samples))

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Trace Plot", "📊 Histogram", "📉 Diagnostics"])

        with tab1:
            # Trace plot using Plotly
            fig_trace = go.Figure()
            fig_trace.add_trace(go.Scatter(y=samples, mode="lines", name="Samples"))
            fig_trace.update_layout(
                title="MCMC Trace Plot",
                xaxis_title="Iteration",
                yaxis_title="Sample Value",
                height=500,
            )
            st.plotly_chart(fig_trace, use_container_width=True)

        with tab2:
            # Histogram with target distribution
            x = np.linspace(min(samples), max(samples), 1000)
            target_values = [target_dist(xi) for xi in x]

            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=samples, name="Samples", nbinsx=50, histnorm="probability density"
                )
            )
            fig_hist.add_trace(
                go.Scatter(
                    x=x,
                    y=target_values,
                    name="Target Distribution",
                    line=dict(color="red"),
                )
            )
            fig_hist.update_layout(
                title="Sample Distribution vs Target",
                xaxis_title="Value",
                yaxis_title="Density",
                height=500,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab3:
            if acceptance_rates is not None:
                # Acceptance rate plot for AMH
                fig_acc = px.line(
                    y=acceptance_rates,
                    title="Acceptance Rate Over Time",
                    labels={"x": "Check Interval", "y": "Acceptance Rate"},
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            # Add autocorrelation plot
            autocorr = pd.Series(samples).autocorr()
            st.metric("Autocorrelation (lag 1)", f"{autocorr:.3f}")

        progress_bar.progress(90)
        status_text.text("Preparing download options...")

        # Download buttons
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            df_samples = pd.DataFrame(samples, columns=["value"])
            csv = df_samples.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Samples (CSV)",
                data=csv,
                file_name="mcmc_samples.csv",
                mime="text/csv",
            )

        with col2:
            results_dict = {
                "sampler_type": sampler_type,
                "expression": expression,
                "iterations": iterations,
                "burn_in": burn_in,
                "thin": thin,
                "seed": seed,
                "elapsed_time": elapsed_time,
                "acceptance_rate": acceptance_rate,
            }
            if acceptance_rates is not None:
                results_dict["acceptance_rates"] = acceptance_rates

            import json

            results_json = json.dumps(results_dict)
            st.download_button(
                label="Download Configuration (JSON)",
                data=results_json,
                file_name="mcmc_config.json",
                mime="application/json",
            )

        progress_bar.progress(100)
        status_text.text("Sampling completed successfully!")
        sleep(1)  # Give users time to see the completion
        progress_bar.empty()
        status_text.empty()

except ValueError as e:
    st.error(f"Value Error: {str(e)}")
except TypeError as e:
    st.error(f"Type Error: {str(e)}")
except (RuntimeError, OverflowError, ZeroDivisionError) as e:
    st.error(f"Computation Error: {str(e)}")
except MemoryError:
    st.error("Not enough memory to complete operation")
except KeyboardInterrupt:
    st.error("Operation cancelled by user")
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"Module import error: {str(e)}")
except OSError as e:
    st.error(f"System error: {str(e)}")
