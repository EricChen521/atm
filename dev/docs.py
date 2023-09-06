"""Build docs.

Run with --open to open docs after building.

"""
import argparse
import os

from dev import utils
from dev.utils import conda_run, green, shell


def _docs(*, open):
    conda_run("bash -c 'cd docs && make clean && make html'")

    if open:
        os_name = os.uname().sysname
        if os_name == "Darwin":
            shell("open docs/_build/html/index.html")
        elif os_name == "Linux":
            shell("xdg-open docs/_build/html/index.html")
        else:
            print(green('Docs built. Open "docs/_build/html/index.html" to view'))
    else:
        print(green('Docs built. Run "python3 dev/docs.py --open" to open'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds docs.")
    parser.add_argument(
        "-o", "--open", action="store_true", help="Open docs after building them."
    )
    args = parser.parse_args()
    utils.validate_python()
    _docs(open=args.open)
