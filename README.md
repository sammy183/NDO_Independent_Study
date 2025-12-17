**Brief Overview**
- main.py calls the optimzation
- RunOpt.py contains the optimization function and outputs
- aerodynamicsOpt.py contains functions for drag calculation
- propulsionsOpt.py contains functions for parsing the propeller data and thrust/efficiency calculations

- PlotNpax.py contains functions that run the optimization for a range of fixed npax; uses multiprocessing for speedups



Required packages:
- The standard: numpy, scipy, matplotlib
- Gekko: https://gekko.readthedocs.io/en/latest/
- Aerosandbox: https://github.com/peterdsharpe/AeroSandbox
- tqdm: https://pypi.org/project/tqdm/
- numba: https://numba.pydata.org/
- pyxdsm (if running XDSM.py for the SAND xdsm graph): https://github.com/mdolab/pyXDSM
