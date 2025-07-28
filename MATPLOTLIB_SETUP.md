# Matplotlib Setup Guide for measure-uq

## Issue Resolution

If you're encountering a `ModuleNotFoundError: No module named 'matplotlib'` or related errors when running Jupyter notebooks, follow these steps to resolve the issue.

## System Dependencies

1. Install the required system packages:

```bash
sudo apt-get install python3-tk -y
```

This installs the Tkinter library which is required by matplotlib for certain GUI operations.

## Verify Matplotlib Installation

1. Check if matplotlib is installed in your Poetry environment:

```bash
poetry run pip list | grep matplotlib
```

You should see output similar to:
```
matplotlib                    3.10.3
matplotlib-inline             0.1.7
```

2. If matplotlib is not installed, add it to your project:

```bash
poetry add matplotlib ipympl
```

3. Run a simple test to verify matplotlib works:

```bash
poetry run python -c "import matplotlib; print(matplotlib.__version__)"
```

## Running Jupyter Notebooks

To ensure Jupyter uses the correct environment with matplotlib:

1. Always launch Jupyter from within the Poetry environment:

```bash
poetry run jupyter notebook
```

2. When opening a notebook, make sure to select the correct kernel that corresponds to your Poetry environment.

3. For notebooks using the `%matplotlib widget` magic command, ensure `ipympl` is installed:

```bash
poetry add ipympl
```

## Troubleshooting

If you still encounter issues:

1. Try restarting the kernel from the Jupyter interface (Kernel > Restart)

2. Verify that your notebook is using the correct kernel by checking the top right corner of the notebook interface

3. If using VS Code or another IDE with Jupyter integration, ensure it's configured to use your Poetry environment

4. For persistent issues, try creating a fresh kernel for your Poetry environment:

```bash
poetry run python -m ipykernel install --user --name=measure-uq
```

## Testing Matplotlib

You can use this simple script to test if matplotlib is working correctly in your environment:

```python
#!/usr/bin/env python

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    
    import matplotlib.pyplot as plt
    print("matplotlib.pyplot imported successfully")
    
    # Try to create a simple plot
    plt.figure()
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.savefig('test_plot.png')
    print("Plot created and saved as test_plot.png")
    
    # Check if ipympl is available
    try:
        import ipympl
        print(f"ipympl version: {ipympl.__version__}")
    except ImportError as e:
        print(f"Error importing ipympl: {e}")
        
except ImportError as e:
    print(f"Error importing matplotlib: {e}")
```

Save this as `test_matplotlib.py` and run it with `poetry run python test_matplotlib.py`.