# mcmc-microservice

[![Python application test with Github Actions](https://github.com/blueskycircle/mcmc-microservice/actions/workflows/main.yml/badge.svg)](https://github.com/blueskycircle/mcmc-microservice/actions/workflows/main.yml)

## Metropolis-Hastings Example

![image](https://github.com/user-attachments/assets/a44d6601-537f-427c-add8-46a6a48e91c4)

## Adaptive Metropolis-Hastings Example

![image](https://github.com/user-attachments/assets/b27eb69c-b137-40f6-a177-f9f4eb01d64e)

## Command Line Interface (CLI)

### Get help
`python cli.py --help`

### Run standard Metropolis-Hastings
`python cli.py mh --iterations 10000`

### Run adaptive Metropolis-Hastings with custom parameters
`python cli.py amh -e "exp(-0.5 * (x - 2)**2) / sqrt(2 * pi)" -n 20000 --initial-variance 2.0`

###  Save samples without plotting
`python cli.py mh --no-plot --save -o "mh_samples.txt"`

### Run with a specific seed for reproducibility
`python cli.py mh --iterations 1000 --seed 42`

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



