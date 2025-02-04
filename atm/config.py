import dataclasses
import logging
import pathlib

from pathlib import Path
from typing import Dict, List, Tuple, Union

import yaml

LOGGER = logging.getLogger(__name__)

LOG_DIR = pathlib.Path("logs")


@dataclasses.dataclass
class AtmConfig:
    """A class to control all parameters running atm workflow."""

    protein_fpathname: str = None
    ligand_dpathname: str = None
    cofactor_fpathname: str = None
    forcefield_option: str = "openff"  # only support openff in the future, 
    forcefield_dpathname: str = None
    atom_build_pathname: str = "~/github/ATom_OpenMM"
    atm_pythonpathname: str = "~/miniconda3/envs/atm-dev/python" # the python pathname from atm-dev env
    atm_type: str = "abfe"  # 'abfe', 'rbfe'
    morph_fpathname: str = None  # 'Morph.in' file that has pairname(ligA~ligB) for rbfe
    morph_type: str = None  # 'star' or 'lomap', to generate Morph.in when not provided
    relaxed_res_ids: List[int] = None # # the residue id without positional restraint. such as [58,59,60](index based on pdb file)
    vsite_radius: float = 5.0  # unit in Å to define viste
    ref_ligname: str = None  # reference ligand name
    ref_alignidx: List[
        int
    ] = None  # three atom indexes for alignment, index starts from 1.
    ref_ligdG: float = 0.0  # reference ligand exp dG.
    kforce_vsite: float = 25.0
    kforce_displ: float = 2.5
    kforce_theta: float = 50.0
    kforce_psi: float = 50.0
    displ_vec: Tuple[float] = None
    sim_time: float = 5.0  # simulation time (ns) for a lambda window.
    dt: float = 0.004  # step size for remd
    exchange_interval: int = 20000  # step interval for an exchange attempt.
    print_energy_interval: int = 20000  # step interval to print energy,
    print_traj_interval: int = 60000  # step interval to print coords.
    disregard_ratio: int = 0.3  # disregard the first 30% samples for analysis.
    gpu_devices: List[int] = None  # available gpu device indexes
    gres: int = 4 # the GPU number of th node requested by atm job
    work_dir: str = "."  # atm workflow work dir
    is_slurm: bool = False # True if run on slurm cluster
    partition: str = None # the partition of cluster to run atm

    # def get(self,name:str)
    def to_dict(self) -> dict:
        """Generate the dict form of the instance."""
        config_dict = {}
        for param in vars(self):
            config_dict.update({param: self.__getattribute__(param)})
        return config_dict

    def write_to_yaml(self) -> None:
        """Write to a yaml file."""
        config_dict = self.to_dict()
        with open(f"{self.work_dir}/atm_config_standard.yaml", "w") as fh:
            yaml.dump(config_dict, fh)

    def update_param(self, input: Union[Dict, Path] = None) -> None:
        """Update config by dict or yaml file input."""
        if isinstance(input, Dict):
            for k, v in input.items():
                if k in vars(self):
                    self.__dict__[k] = v
        elif isinstance(input, Path):
            with open(input, "r") as fh:
                params = yaml.safe_load(fh)
                for k, v in params.items():
                    if k in vars(self):
                        self.__dict__[k] = v
