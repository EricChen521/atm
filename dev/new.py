"""Create a skeleton code for a new Python module or package."""

import subprocess

from pathlib import Path

import typer

typer.rich_utils.MAX_WIDTH = 88


def green(text) -> str:
    """Returns `text` in green color."""
    return f"\033[92m{text}\033[0m"


def red(text) -> str:
    """Returns `text` in red color."""
    return f"\033[91m{text}\033[0m"


def shell(command) -> str:
    """Runs `command` in a shell.

    Returns:
      The stdout contents.

    Raises:
      subprocess.CalledProcessError: The command fails.
    """
    ret = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
    return ret.stdout.decode("utf-8").strip() if ret.stdout else ""


app = typer.Typer()


@app.command()
def module(name: str):
    """Create a skeleton code for a new Python module in the current directory."""
    if not name:
        exit(red("ERROR: Please provide the new module's name"))
    root_dir = shell("git rev-parse --show-toplevel")
    module_template_path = Path(root_dir) / "dev/templates/module.py"
    if not module_template_path.is_file():
        exit(red(f"ERROR: Template code file not found: '{module_template_path}'"))
    shell(f"cp {module_template_path} {name}.py")
    tests_path = Path("tests")
    tests_path.mkdir(exist_ok=True)
    test_template_path = Path(root_dir) / "dev/templates/test_module.py"
    shell(f"cp {test_template_path} tests/test_{name}.py")
    print(green("Created the following files for the new module:"))
    print(f"  {name}.py")
    print(f"  tests/test_{name}.py")


@app.command()
def package(name: str):
    """Create a dir structure for a new Python package in the current directory."""
    if not name:
        exit(red("ERROR: Please provide the new module's name"))
    package_path = Path() / name
    package_path.mkdir(exist_ok=True)
    (package_path / "tests").mkdir(exist_ok=True)
    with open(package_path / "__init__.py", "w") as fh:
        fh.write("# noqa: D104\n")
    print(green("Created the following dir's and files for the new package:"))
    print(f"  {name}/")
    print(f"  {name}/tests/")
    print(f"  {name}/__init__.py")


if __name__ == "__main__":
    app()
