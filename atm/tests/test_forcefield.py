from unittest import mock

from atm.forcefield import Gaff
from atm.utility import data_dir


@mock.patch("subprocess.Popen")
def test_gaff_mocked(_mock_Popen, tmp_path) -> None:
    _mock_Popen.return_value.returncode = 0
    _mock_Popen.return_value.communicate.return_value = ("".encode(), "".encode())
    forcefield_dpath = tmp_path / "forcefield"
    print(f"Generate gaff forcefield at: {forcefield_dpath}")
    ligands_dpath = data_dir() / "001.prep" / "ligands"
    ff = Gaff(ligands_dpath=ligands_dpath, forcefield_dpath=forcefield_dpath)
    ff.produce()
    assert forcefield_dpath.is_dir()
    assert not (forcefield_dpath / "Compd-0044699").is_dir()


def test_gaff(tmp_path) -> None:
    forcefield_dpath = tmp_path / "forcefield"
    print(f"Generate gaff forcefield at: {forcefield_dpath}")
    ligands_dpath = data_dir() / "001.prep" / "ligands"
    ff = Gaff(ligands_dpath=ligands_dpath, forcefield_dpath=forcefield_dpath)
    ff.produce()
    assert forcefield_dpath.is_dir()
    assert (forcefield_dpath / "Compd-0044699").is_dir()
