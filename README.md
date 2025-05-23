# mcmc-microservice

[![Python application test with Github Actions](https://github.com/blueskycircle/mcmc-microservice/actions/workflows/main.yml/badge.svg)](https://github.com/blueskycircle/mcmc-microservice/actions/workflows/main.yml)

The `mcmc-microservice` is a versatile toolkit for Markov Chain Monte Carlo (MCMC) sampling. It provides a command-line interface (CLI), a RESTful API, and an interactive web application for running MCMC algorithms. This project enables users to sample from custom target distributions, visualize results in real-time, and export samples for further analysis.

## Table of Contents

1.  [Command Line Interface (CLI)](#command-line-interface-cli)
    *   [Installation](#installation)
    *   [Basic Usage](#basic-usage)
    *   [Standard Metropolis-Hastings (mh)](#standard-metropolis-hastings-mh)
    *   [Adaptive Metropolis-Hastings (amh)](#adaptive-metropolis-hastings-amh)
    *   [Examples](#examples)
    *   [Output](#output)
    *   [File Structure](#file-structure)
2.  [Application Programming Interface (API)](#application-programming-interface-api)
    *   [Running the API Server](#running-the-api-server)
    *   [Endpoints](#endpoints)
        *   [Standard Metropolis-Hastings (/mcmc/mh)](#standard-metropolis-hastings-mcmcmh)
        *   [Adaptive Metropolis-Hastings (/mcmc/amh)](#adaptive-metropolis-hastings-mcmcamh)
    *   [Response Format](#response-format)
    *   [Examples](#examples-1)
3.  [Web Application (Streamlit)](#web-application-streamlit)
    *   [Running the Web Application](#running-the-web-application)
    *   [Features](#features-1)
    *   [Example Usage](#example-usage)
    *   [Tips for Best Experience](#tips-for-best-experience)
4.  [Project Structure](#project-structure)
    *   [Key Components](#key-components)
    *   [File Descriptions](#file-descriptions)
    *   [Generated Files](#generated-files)
5.  [Further Work](#further-work)


## Command Line Interface (CLI)

The MCMC microservice provides two command-line tools for MCMC sampling: standard Metropolis-Hastings (`mh`) and adaptive Metropolis-Hastings (`amh`).

![CLI Demo](assets/cli-demo.gif)

### Installation

```cmd
# Clone the repository
git clone https://github.com/blueskycircle/mcmc-microservice.git
cd mcmc-microservice

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Get help and view all available commands:
```cmd
python cli.py --help

# View options for specific command
python cli.py mh --help
python cli.py amh --help
```

### Standard Metropolis-Hastings (mh)

Basic sampling from standard normal distribution:
```cmd
python cli.py mh --iterations 10000
```

**Key Parameters:**
- `-e, --expression`: Mathematical expression for target distribution (default: standard normal)
- `-i, --initial`: Initial value to start the chain (default: 0.0)
- `-n, --iterations`: Number of iterations to run (default: 10000)
- `-b, --burn-in`: Number of initial samples to discard (default: 1000)
- `-t, --thin`: Keep every nth sample (default: 1)
- `-s, --seed`: Random seed for reproducibility (optional)
- `--plot/--no-plot`: Enable/disable plotting (default: enabled)
- `--save/--no-save`: Save samples to file (default: disabled)
- `-o, --output`: Output filename for saving samples (default: "samples.txt")
- `--credible-interval`: Credible interval level between 0 and 1 (default: 0.95)

**Example with all parameters:**
```cmd
python cli.py mh  ^
    --expression "exp(-0.5 * x**2) / sqrt(2 * pi)" ^
    --initial 0.0 ^
    --iterations 10000 ^
    --burn-in 1000 ^
    --thin 1 ^
    --seed 42 ^
    --plot ^
    --save ^
    --output "samples.txt" ^
    --credible-interval 0.95
```

### Adaptive Metropolis-Hastings (amh)

Sample from a custom distribution with adaptive proposal:
```cmd
python cli.py amh ^ 
    -e "exp(-0.5 * (x - 2)**2) / sqrt(2 * pi)" ^
    -n 20000 ^
    --initial-variance 2.0
```

**Additional Parameters:**
- `--initial-variance`: Initial proposal variance (default: 1.0)
- `--check-interval`: Interval for checking acceptance rate (default: 200)
- `--increase-factor`: Factor to increase variance (default: 1.1)
- `--decrease-factor`: Factor to decrease variance (default: 0.9)

### Examples

1. **Save samples without plotting:**
```cmd
python cli.py mh --no-plot --save -o "mh_samples.txt"
```

2. **Reproducible sampling with seed:**
```cmd
python cli.py mh --iterations 1000 --seed 42
```

3. **Sample from Gumbel distribution:**
```cmd
python cli.py amh ^
    -e "(1/3) * exp(-((x - 2)/3) - exp(-((x - 2)/3)))" ^
    --initial 0 ^
    --iterations 50000 ^
    --initial-variance 1 ^
    --increase-factor 1.1 ^
    --decrease-factor 0.9 ^
    --burn-in 10000 ^
    --thin 1 ^
    --seed 42 ^
    --plot ^
    --save ^
    --output gumbel_samples.txt
```

### Output

The CLI tools generate:
1. **Console output:**
   - Time taken
   - Acceptance rate
   - Number of samples

2. **Plots (if enabled):**
   - Trace plot
   - Histogram with target distribution
   - Acceptance rate over time (AMH only)

3. **Sample file (if saving enabled):**
   - Text file with MCMC samples
   - Stored in `output/samples/` directory

### File Structure
```
output/
├── plots/          # Generated plots
│   └── *.png
└── samples/        # Saved samples
    └── *.txt
```

## Application Programming Interface (API)

The MCMC microservice provides a REST API for running MCMC samplers. The API is built using FastAPI and provides two main endpoints.

![API Demo 1](assets/api-demo-a.gif)

![API Demo 2](assets/api-demo-b.gif)

### Running the API Server

```cmd
# Start the API server
python api.py
```

The server runs at `http://localhost:8000` by default. Access the interactive API documentation at `http://localhost:8000/docs`.

### Endpoints

#### 1. Standard Metropolis-Hastings (`/mcmc/mh`)

Runs the standard Metropolis-Hastings algorithm.

**Example Request:**
```cmd
curl -X "POST" ^
  "http://localhost:8000/mcmc/mh" ^
  -H "accept: application/json" ^
  -H "Content-Type: application/json" ^
  -d "{\"expression\": \"exp(-0.5 * x**2) / sqrt(2 * pi)\", \"iterations\": 1000, \"burn_in\": 100, \"thin\": 1, \"seed\": 42, \"credible_interval\": 0.95}"
```

**Parameters:**
- `expression` (optional): Target distribution expression (default: standard normal)
- `initial` (float, default: 0.0): Initial value for the chain
- `iterations` (int, default: 10000): Number of iterations
- `burn_in` (int, default: 1000): Number of initial samples to discard
- `thin` (int, default: 1): Keep every nth sample
- `seed` (int, optional): Random seed for reproducibility

#### 2. Adaptive Metropolis-Hastings (`/mcmc/amh`)

Runs the adaptive Metropolis-Hastings algorithm with automatic proposal tuning.

**Example Request:**
```cmd
curl -X "POST" ^
  "http://localhost:8000/mcmc/amh" ^
  -H "accept: application/json" ^
  -H "Content-Type: application/json" ^
  -d "{\"expression\": \"exp(-0.5 * x**2) / sqrt(2 * pi)\", \"iterations\": 1000, \"initial_variance\": 1.0, \"check_interval\": 100, \"increase_factor\": 1.1, \"decrease_factor\": 0.9, \"seed\": 42}"
```

**Additional Parameters:**
- `initial_variance` (float, default: 1.0): Initial proposal variance
- `check_interval` (int, default: 200): Interval for checking acceptance rate
- `increase_factor` (float, default: 1.1): Factor to increase variance
- `decrease_factor` (float, default: 0.9): Factor to decrease variance

### Response Format

Both endpoints return JSON responses with the following structure:

```json
{
  "samples": [...],
  "elapsed_time": 1.23,
  "acceptance_rate": 0.45,
  "mean": 0.123,
  "median": 0.456, 
  "credible_interval": [-1.96, 1.96]
}
```

The AMH endpoint additionally returns:
```json
{
  "acceptance_rates": [...] 
}
```

**Example Response:**
```json
{
  "samples": [0.123, -0.456, 0.789, ...],
  "elapsed_time": 1.23,
  "acceptance_rate": 0.45,
  "mean": 0.157,
  "median": 0.162,
  "credible_interval": [-1.89, 2.14],
  "acceptance_rates": [0.48, 0.46, 0.44]
}
```

### Examples

#### Sampling from a Gumbel Distribution
```cmd
curl -X "POST" ^
  "http://localhost:8000/mcmc/amh" ^
  -H "accept: application/json" ^
  -H "Content-Type: application/json" ^
  -d "{\"expression\": \"(1/3) * exp(-((x - 2)/3) - exp(-((x - 2)/3)))\", \"iterations\": 50000, \"initial_variance\": 1.0, \"check_interval\": 200, \"burn_in\": 10000, \"seed\": 42}"
```

## Web Application (Streamlit)

The MCMC microservice provides an interactive web interface built with Streamlit for running and visualizing MCMC samplers.

![WEB Demo](assets/web-app-demo.gif)

### Running the Web Application

```cmd
# Start the Streamlit app
streamlit run web_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Features

#### Interactive Controls
- Choose between standard MH and adaptive MH samplers
- Adjust sampling parameters in real-time:
  - Target distribution expression
  - Number of iterations
  - Burn-in period
  - Thinning interval
  - Random seed
- AMH-specific parameters:
  - Initial variance
  - Check interval
  - Increase/decrease factors

#### Visualization Options
- Interactive plots:
  - Trace plot
  - Histogram with target distribution overlay
  - Acceptance rate over time (AMH only)
- Plotly-powered interactive charts with zoom and pan capabilities

#### Results and Downloads
- Key metrics displayed:
  - Elapsed time
  - Acceptance rate
  - Number of samples
  - Autocorrelation statistics
- Download options:
  - Save samples as CSV
  - Export configuration as JSON

### Example Usage

1. **Basic Sampling:**
   - Select sampler type (MH or AMH)
   - Use default standard normal distribution
   - Click "Run Sampler"

2. **Custom Distribution:**
   - Enter your target distribution expression
   - Example: `exp(-0.5 * (x - 2)**2) / sqrt(2 * pi)`
   - Adjust parameters as needed
   - Click "Run Sampler"

3. **Visualization Customization:**
   - Interact with plots (zoom, pan, hover)
   - Download results for further analysis

### Tips for Best Experience

1. **Performance:**
   - Start with smaller iterations for testing
   - Increase iterations gradually for better sampling

2. **Visualization:**
   - Use plot interactions for detailed inspection

3. **Downloads:**
   - Save configuration for reproducibility
   - Export samples for external analysis

## Project Structure

```
mcmc-microservice/
├── library/                      # Core MCMC implementation
│   ├── __init__.py
│   ├── mcmc_algorithms.py       # MCMC sampling algorithms
│   └── mcmc_utils.py           # Utility functions and distributions
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_api.py             # API endpoint tests
│   ├── test_cli.py             # CLI functionality tests
│   └── test_mcmc.py            # Core MCMC algorithm tests
│
├── api.py                      # FastAPI implementation
├── cli.py                      # Command-line interface
├── web_app.py                 # Streamlit web application
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

### Key Components

#### Core Library (`/library`)
- `mcmc_algorithms.py`: Implements both standard and adaptive Metropolis-Hastings
- `mcmc_utils.py`: Contains target distribution handling and proposal functions

#### Interfaces
- `cli.py`: Command-line interface using Click
- `api.py`: RESTful API using FastAPI
- `web_app.py`: Interactive web interface using Streamlit

#### Tests (`/tests`)
- Unit tests for all components
- Integration tests for endpoints
- MCMC algorithm validation

### File Descriptions

1. **Core Implementation**
   - `mcmc_algorithms.py`: Contains `metropolis_hastings()` and `adaptive_metropolis_hastings()`
   - `mcmc_utils.py`: Includes `target_distribution()` and `proposal_distribution()`

2. **Interface Files**
   - `cli.py`: Implements `mh` and `amh` commands
   - `api.py`: Provides `/mcmc/mh` and `/mcmc/amh` endpoints
   - `web_app.py`: Interactive dashboard with real-time visualization

3. **Configuration Files**
   - `requirements.txt`: Lists all Python dependencies
   - `README.md`: Project documentation and usage guides

4. **Test Files**
   - `test_mcmc.py`: Core algorithm validation
   - `test_cli.py`: Command-line interface testing
   - `test_api.py`: API endpoint validation

### Generated Files

1. **Plots**
   - Trace plots
   - Histograms
   - Acceptance rate plots (AMH)

2. **Samples**
   - Raw MCMC samples
   - Configuration files
   - CSV exports

## Further Work

1. Add more MCMC methods.
2. Allow for 2-D target distributions.
3. Turn images into 2-D target distributions.
4. Add MCMC diagnostics.
5. Containerize the tool with Docker.
6. Add a test_invalid_expression test. This is very important.
7. In the web application, allow the user to see the live updating progress bar.
8. Create tests for the web application.


