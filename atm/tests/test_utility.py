import os

import pytest

from atm.utility import create_amber_system, create_xml_from_openff, data_dir, tmp_cd


def test_tmp_cd(tmp_path):
    origin = tmp_path / "origin"
    workdir = tmp_path / "workdir"
    origin.mkdir(parents=True)
    workdir.mkdir(parents=True)
    os.chdir(origin)
    with tmp_cd(workdir):
        assert os.getcwd() == str(workdir)

    assert os.getcwd() == str(origin)


def test_create_amber_system(tmp_path):
    protein_fpath = data_dir() / "001.prep" / "proteins" / "KRAS" / "protein.pdb"
    lig1_fpath = data_dir() / "forcefield" / "Compd-0044102" / "vacuum.mol2"
    lig2_fpath = data_dir() / "forcefield" / "Compd-0044699" / "vacuum.mol2"
    top_fpath, crd_fpath = create_amber_system(
        output_crd_path=tmp_path / "test.inpcrd",
        output_top_path=tmp_path / "test.prmtop",
        solute_paths=[protein_fpath, lig1_fpath, lig2_fpath],
        translate=(2, (33.99, -21.76, 0.25)),
        buffer_size=10,
    )

    assert top_fpath.is_file()
    assert crd_fpath.is_file()
    assert (tmp_path / "tleap.in").is_file()


@pytest.mark.parametrize(
    "lig2_fpath",
    [None, data_dir() / "001.prep" / "ligands" / "Compd-0044699" / "ligand.sdf"],
)
def test_create_xml_from_openff(tmp_path, lig2_fpath):
    protein_fpath = data_dir() / "001.prep" / "proteins" / "KRAS" / "protein.pdb"
    lig1_fpath = data_dir() / "001.prep" / "ligands" / "Compd-0044102" / "ligand.sdf"
    cofactor_fpath = data_dir() / "001.prep" / "cofactors" / "GDP" / "cofactor.sdf"
    xml_out_fpath, pdb_out_fpath = create_xml_from_openff(
        protein_fpath=protein_fpath,
        lig1_fpath=lig1_fpath,
        lig2_fpath=lig2_fpath,
        cofactor_fpath=cofactor_fpath,
        translation_vec=(33.99, -21.76, 0.25),
        xml_out_fpath=tmp_path / "complex.xml",
        pdb_out_fpath=tmp_path / "complex.pdb",
    )
    assert xml_out_fpath.is_file()
    assert pdb_out_fpath.is_file()
