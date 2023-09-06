"""Runs a command within the development environment.

This script itself has minimum dependencies so that we can use it in an CI environment
to run scripts or commands that only work in the development environment (because of
the dependencies).

This script should be run in the repository's root dir with a "dev/spec.py" file where
the repo's name is defined by a `REPO_NAME` variable. The conda development environment
is then expected to be called "{REPO_NAME}-dev`.

To run a Python script with the .py or .pyc suffix, you can use
`python dev/run.py path/to/myscript.py args...`, in other words, you don't have to add
`python` right before your script. Also, if the Python script is in the same "dev/" dir,
you can omit the path.

To run a script with the "-h" or "--help" option, which conflicts with the options of
this run.py script, you can prefix the option with a '@' to avoid the conflicts, e.g.,
`python dev/run.py lint.py @--help`.
"""

import argparse
import os
import shlex
import subprocess
import sys


def _blue(text):
    """Returns `text` in blue color."""
    return f"\033[96m{text}\033[0m"


if __name__ == "__main__":
    sys.path.insert(0, "dev")

    from spec import REPO_NAME

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument("cmd", help="The command to run")
    parser.add_argument("args", help="The arguments/options of the command", nargs="*")
    parsed, unknown = parser.parse_known_args()

    # Converts an option like "@--help" to "--help".
    args = [(e[1:] if e.startswith("@-") else e) for e in parsed.args + unknown]
    cmd = parsed.cmd
    if cmd.endswith((".py", ".pyc")):
        if not os.path.isfile(cmd):
            cmd = "dev" + os.sep + cmd
        cmd = ["python", cmd]
    else:
        cmd = [cmd]
    cmd = shlex.join(cmd + args)
    env = REPO_NAME + "-dev"
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    full_cmd = f"{conda_exe} run --live-stream -n {env} {cmd}"

    print(_blue(full_cmd))
    subprocess.run(full_cmd, shell=True, check=True)
