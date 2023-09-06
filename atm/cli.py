import logging
import pickle

from pathlib import Path

import click

from atm.config import AtmConfig
from atm.forcefield import Gaff
from atm.system import (
    calc_displ_vec,
    get_alignment,
    parse_protein,
    setup_atm_dir,
    submit_localjob,
    update_scripts,
)

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

    if Path("protein_info.pickle").is_file():
        LOGGER.info("Read protein information from the pickle file.")
        with open("protein_info.pickle", "rb") as fh:
            protein_info = pickle.load(fh)
    else:
        protein_info = parse_protein(config=config)

    if config.displ_vec:
        LOGGER.info(f"Use user specified displacement vector: {config.displ_vec}.")
    elif Path("displacement_vec.pickle").is_file():
        LOGGER.info("Read displacement vector from pickle file.")
        with open("displacement_vec.pickle", "rb") as fh:
            displacement_vec = pickle.load(fh)
    else:
        LOGGER.info("calculating displacement vector.")
        displacement_vec = calc_displ_vec(config=config)
        config.displ_vec = displacement_vec.tolist()
    
    if config.forcefield_dpathname and Path(config.forcefield_dpathname).is_dir():

        LOGGER.info(f"Use the provided forcefiled from {config.forcefield_dpathname}.")

    elif config.forcefield_option in ["gaff", "quickgaff"]:

        LOGGER.info(f"Generate forcefield with {config.forcefield_option}.")
        # generate the gaff based forcefield for ligands/cofactor
        ff = Gaff(
            ligands_dpath=ligand_dpath,
            cofactor_dpath=Path(config.cofactor_fpathname).parent
            if config.cofactor_fpathname
            else None,
            forcefield_dpath=Path(config.work_dir) / "forcefield",
        )
        ff.produce()
        config.forcefield_dpathname = str(Path(config.work_dir)/"forcefield")

    elif config.forcefield_option == "openff":
        pass

    else:
        raise NotImplementedError(f"{config.forcefield_option} is not implemented yet.")

    # setup the `free_energy` dir
    config.write_to_yaml()
    LOGGER.info(f"setup atm dir for {config.atm_type}.")
    setup_atm_dir(config=config)

    alignment_result = get_alignment(config=config)

    # reverse the displacemnet vector if this is abfe as
    # the ligand was moved to edge already
    if config.atm_type == "abfe":
        config.displ_vec = (-displacement_vec).tolist()

    config.write_to_yaml()

    update_scripts(
        config=config, protein_info=protein_info, alignment_result=alignment_result
    )

    _stdout, stderr = submit_localjob(
        free_energy_dpath=Path(f"{config.work_dir}/free_energy")
    )
    if stderr:
        LOGGER.info(stderr.decode("utf-8"))
    else:
        LOGGER.info("ATM workflow is successfully completed!")


if __name__ == "__main__":
    run_atm()
