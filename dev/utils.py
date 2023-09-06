"""Shared utilities for the development scripts."""

# them in a different file.

import os
import subprocess
import sys
import urllib.error
import urllib.request

from dev.spec import MODULE_NAME, REPO_NAME  # noqa: F401

BASE_NAME = "base"
ENV_NAME = f"{REPO_NAME}-dev"
VPN_CONDA_CHANNEL = "TBD"


def validate_python():
    """Ensure that we are running with Python 3.9 or above."""
    if not (sys.version_info.major >= 3 and sys.version_info.minor >= 9):
        print(red("Must be using Python >= 3.9"))
        sys.exit(1)


def green(text):
    """Returns `text` in green color."""
    return f"\033[92m{text}\033[0m"


def red(text):
    """Returns `text` in red color."""
    return f"\033[91m{text}\033[0m"


def blue(text):
    """Returns `text` in blue color."""
    return f"\033[96m{text}\033[0m"


def is_on_vpn():
    """Checks if the VPN is turned on."""
    try:
        urllib.request.urlopen(VPN_CONDA_CHANNEL)  # nosec B310
    except urllib.error.HTTPError as exc:
        if exc.status == 403:
            return False
    return True


def shell(cmd, check=True, stdin=None, stdout=None, stderr=None):
    """Runs `cmd` in a subprocess shell with check=True by default."""
    print(blue(cmd))
    try:
        return subprocess.run(
            cmd,
            shell=True,
            check=check,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.CalledProcessError as exc:
        if "SIGKILL" in str(exc):
            print(
                red(
                    "The Python interpreter was updated while running. "
                    "Try one more time."
                )
            )
        else:
            # Just print the exception instead of raising because the stack trace
            # inside the dev scripts generally distracts from the real error above.
            print(red(str(exc)))
        sys.exit(1)


def shell_echo(message):
    """Print `message` in green color to the stdout via `echo` in the shell."""
    shell(f'echo "{green(message)}"')


def conda_run(cmd, env=ENV_NAME, check=True, stdin=None, stdout=None, stderr=None):
    """Run `cmd` in the given conda environment `env`."""
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    return shell(
        f"{conda_exe} run --live-stream -n {env} {cmd}",
        check=check,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )


def conda_run_stdout(cmd, check=True, env=ENV_NAME) -> str:
    """Run `cmd` in the given conda environment `env` and return the stdout."""
    ret = conda_run(cmd, stdout=subprocess.PIPE, check=check, env=env)
    return ret.stdout.decode("utf-8").strip() if ret.stdout else ""
