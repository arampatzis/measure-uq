#!/usr/bin/env python

try:
    import tkinter
    print("tkinter is successfully imported!")
except ImportError as e:
    print(f"Error importing tkinter: {e}")