"""Install all conda dev dependencies from the lock file.

"""
import argparse
import os.path
import sys

from dev import utils
from dev.utils import conda_run, green, red, shell

CI_ENVAR = "GITLAB_CI"
BUILD_BINARY_ENVAR = "BUILD_BINARY"


def _install(*, skip_locking, build_binary):
    if os.environ.get("CONDA_DEFAULT_ENV") == utils.ENV_NAME:
        print(red('Run "conda deactivate" and try again'))
        sys.exit(1)

    if not skip_locking:
        shell("python dev/lock.py")
    conda_run(
        f"conda-lock install -n {utils.ENV_NAME} --micromamba conda-lock.yml",
        env=utils.BASE_NAME,
    )
    if build_binary or BUILD_BINARY_ENVAR in os.environ:
        print("Run `poetry build` to build binary module.")
        conda_run("poetry build")

    shell("python dev/editable.py . --no-deps")
    if CI_ENVAR in os.environ:
        print("Skip installing `pre-commit` in a CI environment.")
    else:
        conda_run("pre-commit install")
    print(green('Install done! Run "source dev/activate" to finish.'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Installs all requirements from the conda-lock.yml file."
    )
    parser.add_argument(
        "-s",
        "--skip-locking",
        action="store_true",
        help="Skip generating the lock file. Use the locked requirements to set up the "
        "conda environment.",
    )
    parser.add_argument(
        "-b",
        "--build-binary",
        action="store_true",
        help="Run `poetry build` to build binary module as part of installation.",
    )
    utils.validate_python()
    args = parser.parse_args()
    _install(skip_locking=args.skip_locking, build_binary=args.build_binary)
