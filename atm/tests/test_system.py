import pytest
import numpy as np
import shutil

from atm.config import AtmConfig
from atm.system import calc_displ_vec, calc_displacement_vec, setup_atm_dir
from atm.utility import data_dir


def test_calc_displacement_vec():
    protein_fpath = data_dir() / "001.prep" / "proteins" / "KRAS" / "protein.pdb"
    forcefield_dpath = data_dir() / "forcefield"
    displ_vec = calc_displacement_vec(
        protein_fpath=protein_fpath, forcefield_dpath=forcefield_dpath
    )
    print(displ_vec)


def test_calc_displ_vec():
    protein_fpath = data_dir() / "001.prep" / "proteins" / "KRAS" / "protein.pdb"
    ligand_dpath = data_dir() / "001.prep" / "ligands"
    config = AtmConfig()
    config.protein_fpathname = str(protein_fpath)
    config.ligand_dpathname = str(ligand_dpath)
    displ_vec = calc_displ_vec(config=config)
    assert max(displ_vec-np.array([32.56,-21.64,0.36])) < 1
   


@pytest.mark.parametrize("ff_option", ["openff","gaff"])
@pytest.mark.parametrize("atm_type", ["abfe","rbfe"])
def test_setup_atm_dir(tmp_path, atm_type, ff_option):

    config = AtmConfig(
        atm_type=atm_type,
        forcefield_option=ff_option,
        protein_fpathname=str(
            data_dir() / "001.prep" / "proteins" / "KRAS" / "protein.pdb"
        ),
        ligand_dpathname=str(data_dir() / "001.prep" / "ligands"),
        cofactor_fpathname=str(
            data_dir() / "001.prep" / "cofactors" / "GDP" / "cofactor.sdf"
        ),
        morph_fpathname=str(data_dir() / "001.prep" / "Morph.in"),
        work_dir=tmp_path,
        displ_vec=(32.99, -21.76, 0.25),
    )
    shutil.copytree(data_dir()/"forcefield",tmp_path/"forcefield")

    setup_atm_dir(config=config)

    for e in [e for e in (tmp_path / "free_energy").iterdir() if e.is_dir()]:
        print(e)
        assert (e / "complex.xml").is_file()
        assert (e / "complex.pdb").is_file()
