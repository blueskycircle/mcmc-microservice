# mcmc-microservice

[![Python application test with Github Actions](https://github.com/blueskycircle/mcmc-microservice/actions/workflows/main.yml/badge.svg)](https://github.com/blueskycircle/mcmc-microservice/actions/workflows/main.yml)

## Command Line Interface (CLI)

The MCMC microservice provides two command-line tools for MCMC sampling: standard Metropolis-Hastings (`mh`) and adaptive Metropolis-Hastings (`amh`).

https://github.com/user-attachments/assets/6dc543e9-7ade-4728-a7ce-668886a39597

### Installation

```bash
# Clone the repository
git clone https://github.com/blueskycircle/mcmc-microservice.git
cd mcmc-microservice

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Get help and view all available commands:
```bash
python cli.py --help

# View options for specific command
python cli.py mh --help
python cli.py amh --help
```

### Standard Metropolis-Hastings (mh)

Basic sampling from standard normal distribution:
```bash
python cli.py mh --iterations 10000
```

**Key Parameters:**
- `-e, --expression`: Mathematical expression for target distribution
- `-i, --initial`: Initial value (default: 0.0)
- `-n, --iterations`: Number of iterations (default: 10000)
- `-b, --burn-in`: Number of initial samples to discard (default: 1000)
- `-t, --thin`: Keep every nth sample (default: 1)
- `-s, --seed`: Random seed for reproducibility
- `--plot/--no-plot`: Enable/disable plotting (default: enabled)
- `--save/--no-save`: Save samples to file (default: disabled)
- `-o, --output`: Output filename (default: "samples.txt")

### Adaptive Metropolis-Hastings (amh)

Sample from a custom distribution with adaptive proposal:
```bash
python cli.py amh \
    -e "exp(-0.5 * (x - 2)**2) / sqrt(2 * pi)" \
    -n 20000 \
    --initial-variance 2.0
```

**Additional Parameters:**
- `--initial-variance`: Initial proposal variance (default: 1.0)
- `--check-interval`: Interval for checking acceptance rate (default: 200)
- `--increase-factor`: Factor to increase variance (default: 1.1)
- `--decrease-factor`: Factor to decrease variance (default: 0.9)

### Examples

1. **Save samples without plotting:**
```bash
python cli.py mh --no-plot --save -o "mh_samples.txt"
```

2. **Reproducible sampling with seed:**
```bash
python cli.py mh --iterations 1000 --seed 42
```

3. **Sample from Gumbel distribution:**
```bash
python cli.py amh \
    -e "(1/3) * exp(-((x - 2)/3) - exp(-((x - 2)/3)))" \
    --initial 0 \
    --iterations 50000 \
    --initial-variance 1 \
    --increase-factor 1.1 \
    --decrease-factor 0.9 \
    --burn-in 10000 \
    --thin 1 \
    --seed 42 \
    --plot \
    --save \
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

https://github.com/user-attachments/assets/43ca2ad7-a915-414d-be68-b884a6a4cbef

### Running the API Server

```bash
# Start the API server
python api.py
```

The server runs at `http://localhost:8000` by default. Access the interactive API documentation at `http://localhost:8000/docs`.

### Endpoints

#### 1. Standard Metropolis-Hastings (`/mcmc/mh`)

Runs the standard Metropolis-Hastings algorithm.

**Example Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/mcmc/mh' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "expression": "exp(-0.5 * x**2) / sqrt(2 * pi)",
    "iterations": 1000,
    "burn_in": 100,
    "thin": 1,
    "seed": 42
  }'
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
```bash
curl -X 'POST' \
  'http://localhost:8000/mcmc/amh' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "expression": "exp(-0.5 * x**2) / sqrt(2 * pi)",
    "iterations": 1000,
    "initial_variance": 1.0,
    "check_interval": 100,
    "increase_factor": 1.1,
    "decrease_factor": 0.9,
    "seed": 42
  }'
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
  "samples": [...],              // Array of MCMC samples
  "elapsed_time": 1.23,         // Time taken in seconds
  "acceptance_rate": 0.45,      // Overall acceptance rate
  "acceptance_rates": [...]     // (AMH only) Array of acceptance rates over time
}
```

### Examples

#### Sampling from a Gumbel Distribution
```bash
curl -X 'POST' \
  'http://localhost:8000/mcmc/amh' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "expression": "(1/3) * exp(-((x - 2)/3) - exp(-((x - 2)/3)))",
    "iterations": 50000,
    "initial_variance": 1.0,
    "check_interval": 200,
    "burn_in": 10000,
    "seed": 42
  }'
```

## Metropolis-Hastings Example

![image](https://github.com/user-attachments/assets/a44d6601-537f-427c-add8-46a6a48e91c4)

## Adaptive Metropolis-Hastings Example

![image](https://github.com/user-attachments/assets/b27eb69c-b137-40f6-a177-f9f4eb01d64e)

## Things To Do

1. Add more MCMC methods.
2. Allow for 2-D target distributions.
3. Turn images into 2-D target distributions.
4. Add MCMC diagnostics.
5. Containerize the tool with Docker.
6. Add a test_invalid_expression test. This is very important.

## Done 

1. Allow the target distribution to be defined by the user. :heavy_check_mark:
2. Turn into a CLI tool. :heavy_check_mark:
3. Add burn-in capabilities. :heavy_check_mark:
4. Add thinning capabilities. :heavy_check_mark:



