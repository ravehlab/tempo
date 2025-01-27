# tempo
The TEMPO (Temporally-multiscale prediction) integrator for accelerating molecular simulations

Below is a summary of the main packages needed. Please take a look at the requirements or environment files for a more detailed list of dependencies (including version numbers).
conda create --name py39 "python=3.9" --channel conda-forge

# Basic scientific stack
conda install -n py39 numpy matplotlib pandas

conda install ambertools=22 compilers graph-tool imp

# Jupyter
conda install -n py39 -c conda jupyter

# Additional libraries
conda install -n py39 scikit-learn
conda install -n py39 seaborn
