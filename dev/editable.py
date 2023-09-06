"""Installs a Python project in the editable mode using pip."""


import argparse

from dev import utils


def _editable(*, project, no_deps):
    cmd = f"pip install -e {project}"
    if no_deps:
        cmd += " --no-deps"
    utils.conda_run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="The directory name of the project")
    parser.add_argument(
        "--no-deps", help="Do not install project dependencies", action="store_true"
    )
    args = parser.parse_args()
    utils.validate_python()
    _editable(project=args.project, no_deps=args.no_deps)
