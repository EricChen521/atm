"""Lock the development environment whenever pyproject.toml changes.

Use --force to force creation of the lock.

"""
import argparse
import contextlib
import os.path
import subprocess
import sys

from dev import utils
from dev.utils import blue, conda_run, red


def _lock(*, force):
    env_mod_time = os.path.getctime("pyproject.toml")
    try:
        lock_mod_time = os.path.getctime("conda-lock.yml")
    except FileNotFoundError:
        lock_mod_time = 0

    if (lock_mod_time < env_mod_time) or force:
        # The conda lock file is updated in place right now, causing older pip-based
        # dependencies to remain. Remove the lock file to regenerate it from scratch.
        with contextlib.suppress(FileNotFoundError):
            os.remove("conda-lock.yml")

        try:
            conda_run(
                "conda-lock -f pyproject.toml --micromamba --check-input-hash",
                env=utils.BASE_NAME,
            )
        except subprocess.CalledProcessError:
            if not utils.is_on_vpn():
                print(red("You aren't connected to the VPN. Connect and try again."))
                sys.exit(1)
            raise
    else:
        print(
            blue(
                "Lock file not generated because pyproject.toml has not changed since"
                " the last time when the lock file was created.\n"
                "Run `python dev/lock.py --force` to update lock if you have to."
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Locks all requirements to the conda-lock.yml file."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Generate the lock file even if the environment hasn't changed.",
    )
    args = parser.parse_args()

    utils.validate_python()
    _lock(force=args.force)
