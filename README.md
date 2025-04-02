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

Step 3 Install the ATM package by:
````
  git clone https://github.com/EricChen521/atm.git
  cd atm
  PYTHONPATH=. python dev/install.py
````


Step 4 Install UWHAT packagge in the atm-dev env by:
```
conda activate atm-dev
Rscript -e 'install.packages("UWHAM", repos = "http://cran.us.r-project.org")' 
```

Step 5 Install AToM-OpenMM by following the insructions from Dr. Emilio Gallicchio's lab [here](https://github.com/Gallicchio-Lab/AToM-OpenMM). Do not need to install the virtual env, use the atm-dev env instead.

Step 6 Update the system settings and input file paths such as protien, ligands and cofactor in **atm_config.yaml**:
```
atm_pythonpathname
atom_build_pathname
is_slurm
gpu_devices 
```
Step 7 Activate the env and run the ATM calculation
```
conda activate atm-dev
atm -c atm_config.yaml
```
Good luck!  
