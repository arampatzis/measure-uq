#!/usr/bin/env python

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

try:
    notebook_path = 'examples/equations/heat_1d/plot.ipynb'
    print(f"Loading notebook from {notebook_path}")
    notebook = nbformat.read(notebook_path, as_version=4)
    
    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(notebook)
    
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error executing notebook: {e}")