#!/bin/bash

# This script helps run Jupyter notebooks with the correct Poetry environment

echo "Starting Jupyter notebook with Poetry environment..."
poetry run jupyter notebook

echo "Jupyter notebook session ended."