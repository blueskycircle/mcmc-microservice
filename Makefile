install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv tests/test_*.py

format:
	black lib/*.py tests/*.py *.py

lint:
	pylint --disable=R,C lib/mcmc_algorithms.py lib/mcmc_utils.py

all: install lint test