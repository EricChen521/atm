"""Run the test suite.

"""
import argparse

from dev import utils
from dev.utils import conda_run


def _test(skip_building):
    if not skip_building:
        conda_run("python dev/editable.py . --no-deps")
    conda_run(
        "pytest --disable-warnings --junitxml=test_report.xml --cov --color=yes "
        "--cov-report=xml --cov-report=html --cov-report=term-missing "
        "--cov-config=setup.cfg"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the test suite.")
    parser.add_argument(
        "--skip-building",
        action="store_true",
        help="Skip building this project and run the test suite directly.",
    )
    args = parser.parse_args()
    utils.validate_python()
    _test(args.skip_building)
