# Makefile for setting up Python requirements and pre-commit hook
# Works on Linux, macOS, and Windows (via MSYS/MinGW or Git Bash)

## Please use `make target' where target is one of

.DEFAULT_GOAL := help
.PHONY: help setup deps hook

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

##  setup	to set up the environment by installing dependencies and pre-commit hook
setup: deps hook
	@echo "✅ Environment is ready!"

##  deps	to install Python dependencies
deps:
	@echo "🔄 Upgrading pip and installing Python requirements..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

##  hook	to install pre-commit commit-msg hook
hook:
	@echo "🔨 Installing pre-commit commit-msg hook..."
	pre-commit install -t commit-msg
