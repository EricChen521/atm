from atm import config


def test_conf(tmp_path) -> None:
    atm_config = config.AtmConfig()
    atm_config.update_param({"work_dir": str(tmp_path)})
    atm_config.update_param({"ref_ligname": "Compd-0044102"})
    assert atm_config.work_dir == str(tmp_path)
    assert atm_config.ref_ligname == "Compd-0044102"
    atm_config.write_to_yaml()
    assert (tmp_path / "atm_config.yaml").is_file()
