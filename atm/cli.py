import logging
import pickle

from pathlib import Path

import click

from atm.config import AtmConfig
from atm.forcefield import Gaff, Quickgaff
from atm.system import (
    calc_displ_vec,
    get_alignment,
    parse_protein,
    setup_atm_dir,
    submit_job,
    update_scripts,
)
from atm.utility import check_atm_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-10s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config", "config_file", help="Configuration yaml file for ATM.")
def run_atm(config_file):
    """Run ATM  workflow based on the yaml parameter file."""
    config = AtmConfig()
    config.update_param(Path(config_file).resolve())
    check_atm_input(config)

    # update relative paths to absolute paths
    for k, v in config.__dict__.items():
        if isinstance(v, str) and ("/" in v or "." in v or "~" in v):

            config.__dict__[k] = str(Path(v).expanduser().resolve())

    assert config.atm_type in ("abfe", "rbfe"), LOGGER.info(
        f"{config.atm_type} is not supported, ATM only supports abfe/rbfe."
    )

    work_dpath = Path(config.work_dir).resolve()

    with open(work_dpath / "ref.dat", "w") as fh:
        fh.write(f"{config.ref_ligname} {config.ref_ligdG}\n")

    ligand_dpath = Path(config.ligand_dpathname).resolve()  

    if config.displ_vec:
        LOGGER.info(f"Use user specified displacement vector: {config.displ_vec}.")
    else:
        LOGGER.info("calculating displacement vector.")
        displacement_vec = calc_displ_vec(config=config)
        config.displ_vec = displacement_vec.tolist()
    
    assert config.forcefield_option in ["gaff","quickgaff","openff"], f"forcefield {config.forcefield_option} is not supported."
    LOGGER.info(f"Generate forcefield with {config.forcefield_option}.")

    if config.forcefield_option =="gaff":
        # generate the gaff based forcefield for ligands/cofactor
        ff = Gaff(
            ligands_dpath=ligand_dpath,
            cofactor_dpath=Path(config.cofactor_fpathname).parent
            if config.cofactor_fpathname
            else None,
            forcefield_dpath=Path(config.work_dir) / "forcefield",
        )
        ff.produce()

    elif config.forcefield_option == "quickgaff":
        ff = Quickgaff(
            ligands_dpath=ligand_dpath,
            cofactor_dpath=Path(config.cofactor_fpathname).parent
            if config.cofactor_fpathname
            else None,
            forcefield_dpath=Path(config.work_dir) / "forcefield",
            miniconda3_pathname = str(Path(config.atm_pythonpathname).parents[3]),
        )
        ff.produce()
    
    if config.forcefield_option in ["gaff","quickgaff"]:
        assert (Path(config.work_dir) / "forcefield").is_dir(), f"forcefield generation by {config.forcefield_option} failed"
        
    config.forcefield_dpathname = str(Path(config.work_dir) / "forcefield")

    # setup the `free_energy` dir
    setup_atm_dir(config=config)
    alignment_result = get_alignment(config=config)

    # reverse the displacemnet vector if this is abfe as
    # the ligand was moved to edge already
    if config.atm_type == "abfe":
        config.displ_vec = (-displacement_vec).tolist()

    if config.is_slurm:
        LOGGER.info("Use AWS slurm system with GPU:0")
        config.gpu_devices=[0]

    config.write_to_yaml()

    complex_pdb_fpath = next(Path(f"{config.work_dir}/free_energy").iterdir())/"complex.pdb"
    protein_info = parse_protein(
        complex_pdb_fpath=complex_pdb_fpath,
        vsite_radius=config.vsite_radius,
        )

    update_scripts(
        config=config, protein_info=protein_info, alignment_result=alignment_result
    )

    stdout, stderr = submit_job(
        is_slurm=config.is_slurm,
        free_energy_dpath=Path(f"{config.work_dir}/free_energy")
    )
    if stderr:
        LOGGER.info(stderr.decode("utf-8"))
    
    if stdout:
        LOGGER.info(stdout.decode("utf-8"))

    if config.is_slurm:
        LOGGER.info("ATM workflow is successfully submitted!")
    else:
        LOGGER.info("ATM workflow is successfully completed!")


if __name__ == "__main__":
    run_atm()
