ATM
==========================
An experimental Python package to run free energy calculation using AToM-OpenMM

Installation
============

Step 1 Install [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)\
Step 2 Install conda-lock and mamba by:
````
  conda install -c conda-forge conda-lock
  conda install -c conda-forge mamba
````
Step 3 Install AToM-OpenMM by following the insructions from Dr. Emilio Gallicchio's lab [here](https://github.com/Gallicchio-Lab/AToM-OpenMM).

Step 4 Install the ATM package by:
````
  git clone https://github.com/EricChen521/atm.git)https://github.com/EricChen521/atm.git
  cd atm
  PYTHONPATH=. python dev/install.py
````
Step 5 Install UWHAT packagge in the atm-dev env by:
```
conda activate atm-dev
Rscript -e 'install.packages("UWHAM", repos = "http://cran.us.r-project.org")' 
```
Step 5 Update the system settings in **atm_config.yaml**:
```
atm_pythonpathname
atom_build_pathname
is_slurm
gpu_devices 
```
  
