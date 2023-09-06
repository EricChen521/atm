"""Set up the conda development environment.

This will ensure mamba, conda-lock, pip, and python>3.9 are installed in the base
environment, and then install the core project requirements.

"""
import argparse

from dev import utils
from dev.utils import conda_run_stdout, green, red, shell


def _run_diagnostics():
    """Verify that PATH points to the right python and pip."""
    python_path = conda_run_stdout("which python", env=utils.BASE_NAME)
    pip_path = conda_run_stdout("which pip", env=utils.BASE_NAME)

    if "conda" not in python_path.lower():
        print(
            red(
                f'"which python" returned "{python_path}". Your $PATH may be'
                " superceding conda's path, which will likely cause issues"
            )
        )
    else:
        print(green("Your python installation is correct"))

    if "conda" not in pip_path.lower():
        print(
            red(
                f'"which pip" returned "{pip_path}". Your $PATH may be'
                " superceding conda's path, which will likely cause issues"
            )
        )
    else:
        print(utils.green("Your pip installation is correct"))


def _setup(*, diagnose):
    utils.validate_python()

    if diagnose:
        _run_diagnostics()
    else:
        if shell("conda --version > /dev/null", check=False).returncode != 0:
            print(red("ERROR: conda is not activated or installed."))
            print("Please activate/install conda, and try again.")
            exit(1)
        if shell("mamba --version > /dev/null", check=False).returncode != 0:
            print(
                green(
                    """
`mamba` is NOT installed.
Installing `mamba` and a few other core requirements into your base environment.
This might take a while, but it is a one-time setup command. You will never need to run
this again for your conda environment."""
                )
            )
            # Installs mamba in the base environment first.
            shell(
                "conda install --override-channels --strict-channel-priority -y"
                f' -n {utils.BASE_NAME} -c conda-forge "mamba>=1.1.0" "conda>=22.11.0"'
            )
        # Uses mamba to update conda and install conda-lock
        shell(
            "mamba install --override-channels --strict-channel-priority -y"
            f' -n {utils.BASE_NAME} -c conda-forge "conda-lock>=1.3.0" "pip>=22.0"'
            ' "python>=3.9"'
        )
        _run_diagnostics()

        # Installs the development environment.
        shell("python dev/install.py")

        # Ensures the conda-lock.yml file is added.
        shell("git add conda-lock.yml", check=False)

        print(green('Setup done! Run "source dev/activate" to finish.'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sets up the development environment and installs dependencies."
    )
    parser.add_argument(
        "-d", "--diagnose", help="Diagnose common setup issues.", action="store_true"
    )
    args = parser.parse_args()
    utils.validate_python()
    _setup(diagnose=args.diagnose)
