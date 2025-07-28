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