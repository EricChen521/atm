"""Run core linting methods.

Use '--fix' to fix common formatting issues.

"""
import argparse
import os

from dev import utils
from dev.utils import conda_run

SKIP_FOOTING_ENV_VAR = "SKIP_FOOTING"
SKIP_FLAKE8_ENV_VAR = "SKIP_FLAKE8"


def _lint(*, branch, fix, skip_footing, skip_flake8):
    if fix:
        conda_run("isort .")
        conda_run("black .")
    else:
        conda_run("isort . -c")
        conda_run("black . --check")
        conda_run(f"detail lint origin/{branch}..")
        if not skip_footing and not os.environ.get(SKIP_FOOTING_ENV_VAR):
            conda_run("footing update --check")
    if not skip_flake8 and not os.environ.get(SKIP_FLAKE8_ENV_VAR):
        conda_run(f"flake8 -v {utils.MODULE_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lint python code.")
    parser.add_argument(
        "-b",
        "--branch",
        default="develop",
        help="Branch name to run `detail lint` against. (default: develop)",
    )
    parser.add_argument(
        "-f", "--fix", action="store_true", help="Run black and isort to format code."
    )
    parser.add_argument(
        "--skip-footing-check", action="store_true", help="Skip footing check."
    )
    parser.add_argument(
        "--skip-flake8-check", action="store_true", help="Skip flake8 check."
    )
    args = parser.parse_args()
    utils.validate_python()
    _lint(
        branch=args.branch,
        fix=args.fix,
        skip_footing=args.skip_footing_check,
        skip_flake8=args.skip_flake8_check,
    )
