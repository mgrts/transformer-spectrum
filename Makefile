#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = transformer-spectrum
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 transformer_spectrum tests
	isort --check --diff transformer_spectrum tests
	black --check transformer_spectrum tests


## Format source code with black and isort
.PHONY: format
format:
	isort transformer_spectrum tests
	black transformer_spectrum tests


## Run tests
.PHONY: test
test:
	pytest tests/ -v


## Run tests with coverage
.PHONY: test-cov
test-cov:
	pytest tests/ -v --cov=transformer_spectrum --cov-report=html --cov-report=term


## Run fast tests only
.PHONY: test-fast
test-fast:
	pytest tests/ -v -m "not slow"


## Type check with mypy
.PHONY: typecheck
typecheck:
	mypy transformer_spectrum --ignore-missing-imports


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	python -m venv venv
	@echo ">>> Virtual environment created. Activate with: source venv/bin/activate"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Generate synthetic dataset
.PHONY: data-synthetic
data-synthetic:
	$(PYTHON_INTERPRETER) -m transformer_spectrum.cli generate-synthetic \
		-o data/processed/synthetic_dataset.npy


## Download COVID data
.PHONY: data-covid
data-covid:
	$(PYTHON_INTERPRETER) -m transformer_spectrum.cli download-covid
	$(PYTHON_INTERPRETER) -m transformer_spectrum.cli process-covid


## Run training on synthetic data
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m transformer_spectrum.cli train \
		-d data/processed/synthetic_dataset.npy \
		-n makefile-experiment


## Build Docker image
.PHONY: docker-build
docker-build:
	docker build -t transformer-spectrum .


## Run Docker container
.PHONY: docker-run
docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/mlruns:/app/mlruns \
		transformer-spectrum --help


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
