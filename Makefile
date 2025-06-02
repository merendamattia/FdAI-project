# Makefile for setting up Python requirements and pre-commit hook
# Works on Linux, macOS, and Windows (via MSYS/MinGW or Git Bash)

## Please use `make target' where target is one of

.DEFAULT_GOAL := help
.PHONY: help setup deps hook test test-coverage clean run

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

##  setup			to set up the environment by installing dependencies and pre-commit hook
setup: deps hook
	@echo "✅ Environment is ready!"

##  deps			to install Python dependencies
deps:
	@echo "🔄 Upgrading pip and installing Python requirements..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .

##  hook			to install pre-commit commit-msg hook
hook:
	@echo "🔨 Installing pre-commit commit-msg hook..."
	pre-commit install -t commit-msg

##  test			to run unit tests
test:
	@echo "🧪 Running unit tests..."
	python -m pytest src/tests/ -v

##  test-coverage		to run unit tests with coverage report
test-coverage:
	@echo "📊 Running tests with coverage..."
	python -m pytest src/tests/ --cov=src/neural_network --cov-report=term-missing -v

##  clean			to remove Python cache and build files
clean:
	@echo "🧹 Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type f -name ".coverage" -delete

##  run-classify		to run classification experiments
run-classify:
	@echo "🔄 Running classification experiments..."
	python src/main.py --task-type classification --dataset census_income --target salary --epochs 100
	python src/main.py --task-type classification --dataset bank_marketing --target subscribe --epochs 100

##  run-regress		to run regression experiments
run-regress:
	@echo "📈 Running regression experiments..."
	python src/main.py --task-type regression --dataset bike_sharing --target cnt --epochs 100
	python src/main.py --task-type regression --dataset house_price --target price --epochs 100
