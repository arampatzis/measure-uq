# Jupyter Notebook Setup Guide

## Setting Up Jupyter with Poetry Environment

This guide will help you set up Jupyter notebooks to work with the Poetry environment for this project.

### Prerequisites

- Poetry installed
- Project dependencies installed via `poetry install`

### Steps to Set Up Jupyter with Poetry

1. **Install ipykernel in your Poetry environment**

   ```bash
   poetry add ipykernel
   ```

2. **Register the Poetry environment with Jupyter**

   ```bash
   poetry run python -m ipykernel install --user --name=measure-uq
   ```

3. **Verify the kernel is installed**

   ```bash
   jupyter kernelspec list
   ```

   You should see `measure-uq` in the list of available kernels.

### Running Notebooks

1. **Start Jupyter from the Poetry environment**

   ```bash
   poetry run jupyter notebook
   ```

   or

   ```bash
   poetry run jupyter lab
   ```
   
   This ensures that Jupyter is running from the correct environment with access to all the required packages.
   
   **Alternatively, use the provided script:**
   
   ```bash
   ./run_notebook.sh
   ```
   
   This script automatically runs Jupyter notebook with the correct Poetry environment.

2. **Select the correct kernel**

   When opening a notebook, select the `measure-uq` kernel from the kernel menu:
   - In Jupyter Notebook: Kernel → Change kernel → measure-uq
   - In Jupyter Lab: Select kernel from the top right dropdown menu

### Troubleshooting

If you encounter the error `ModuleNotFoundError: No module named 'matplotlib'` or similar errors:

1. Make sure you're using the `measure-uq` kernel in your notebook
2. Check that the package is installed in your Poetry environment with `poetry show`
3. If the issue persists, try restarting the kernel: Kernel → Restart

### Note on Python Version

This project uses Python 3.11. If you're using pyenv to manage Python versions, make sure you have Python 3.11 installed:

```bash
pyenv install 3.11-dev
```