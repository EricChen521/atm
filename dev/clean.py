"""Remove the development environment.

"""
import argparse
import os
import sys

from dev import utils


def _clean():
    if os.environ.get("CONDA_DEFAULT_ENV") == utils.ENV_NAME:
        print(utils.red('Run "conda deactivate" and try again.'))
        sys.exit(1)
    utils.shell(f"conda env remove --name {utils.ENV_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Removes the conda environment.")
    args = parser.parse_args()
    utils.validate_python()
    _clean()
