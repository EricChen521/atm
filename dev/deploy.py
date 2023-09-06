"""Deploy functions for this package.

Includes functions for automated package deployment, changelog generation, and docs
deployment.

"""
import argparse
import contextlib
import copy
import glob
import os
import os.path
import pathlib
import re
import subprocess
import sys
import tempfile

import tomlkit

from packaging import version

from dev import utils
from dev.utils import conda_run, conda_run_stdout, green, red, shell, shell_echo

try:
    from dev import plugins
except ImportError:
    plugins = None

CI_ENVAR = "GITLAB_CI"
CI_BRANCH_ENVAR = "CI_COMMIT_BRANCH"
CI_TAG_ENVAR = "CI_COMMIT_TAG"
CLOUDSMITH_API_USER_ENVAR = "CLOUDSMITH_API_USER"
CLOUDSMITH_API_KEY_ENVAR = "CLOUDSMITH_API_KEY"
DRAILAB_CONDA_REG_ENVAR = "DRAILAB_CONDA_REG"
PY_REG_CLOUDSMITH_ENVAR = "PY_REG_CLOUDSMITH"
MAIN_BRANCH = os.environ.get("MAIN_BRANCH", "main")
DEVELOP_BRANCH = os.environ.get("DEVELOP_BRANCH", "develop")
KEEP_DOCS_VERSIONS = os.environ.get("KEEP_DOCS_VERSIONS", 0)
CONDA_BUILD_NO_TEST_ENVAR = "CONDA_BUILD_NO_TEST"


class Error(Exception):
    """Base exception for this script."""


class NotOnCIError(Error):
    """Thrown when not running on CI."""


@contextlib.contextmanager
def _cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def _configure_git():
    """Configure git name/email and verify git version."""
    shell('git config --local user.email "drailab.com@gmail.com"')
    shell('git config --local user.name "Drailab Engineering"')
    shell("git config push.default current")


def _get_latest_tagged_version() -> str:
    """Finds and returns the latest version tag.

    The latest version tag is either from the remote repo or from the environment
    variable whose name is the value of `CI_TAG_ENVAR`.
    A version tag is in the format: <major>.<minor>.<patch>, where <major> and <minor>
    must be integer numbers, where <patch> can be a string such as "123a1" (no '/'
    allowed). Tags not in the specific format will be ignored.
    If not valid tags found, an empty will be returned.
    """
    with contextlib.suppress(KeyError):
        return os.environ[CI_TAG_ENVAR]

    # Higher versions first
    tags = conda_run_stdout(
        "git ls-remote -tq --refs --sort=-v:refname | cut -f 3 -d /", check=False
    )
    for t in tags.split("\n"):
        fields = t.split(".")
        if len(fields) >= 3:
            major, minor, *patch = fields
            with contextlib.suppress(ValueError):
                int(major), int(minor)
                return t
    return ""


def tag():
    """Bumps the version, generates the changelog, and pushes the tag."""
    # Gets the current versions from repo tags and from pyproject.toml file and checks
    # the consistency.
    poetry_version = conda_run_stdout("poetry version").split()[-1]
    tagged_version = _get_latest_tagged_version()
    if tagged_version and version.parse(poetry_version) != version.parse(
        tagged_version
    ):
        print(
            f"Warning! The latest version tag '{tagged_version}' in the remote repo"
            f" does not match the recorded version '{poetry_version}' in the"
            f" pyproject.toml file. This is typically caused by someone directly"
            f" tagging the '{MAIN_BRANCH}' branch, which may mess up semantic"
            " versioning."
        )

    # Cleans up tags in the local repo, and clones the entire repo.
    shell(f"git tag -d $(git tag -l) && git pull origin {MAIN_BRANCH} --unshallow")

    # Gets the commit range since the latest tag.
    commit_range = f"{tagged_version}..HEAD" if tagged_version else "HEAD"

    # Gets change types for the commits since the latest tag, and then determines the
    # version semantics for this range of changes.
    template = "(% for note in notes %)(( note.type )):(% endfor %)"
    template = template.replace("(", "{").replace(")", "}")
    cmd = f"detail log {commit_range} --template='{template}'"
    change_types = conda_run_stdout(cmd, check=False)
    if "api-break" in change_types:
        semantics = "major"
    elif "bug" in change_types or "feature" in change_types:
        semantics = "minor"
    else:
        semantics = "patch"

    # Bumps the version. The output string is something like this (without quotes):
    #   "Bumping version from 1.2.2 to 1.2.3"
    new_version = conda_run_stdout(f"poetry version {semantics}").split()[-1]
    if new_version == tagged_version:
        print(
            "Warning! The new version conflicts with the latest version tag"
            f" '{tagged_version}'. Bump the version once more."
        )
        new_version = conda_run_stdout("poetry version patch").split()[-1]

    # Updates the 'CHANGELOG.md' file.
    shell(f"git tag {new_version}")
    new_changelog = conda_run_stdout(f"detail log {commit_range}").strip()
    old_changelog = open("CHANGELOG.md").read()
    if -1 != old_changelog.find("# Changelog"):
        old_changelog = old_changelog[11:].strip()
    with open("CHANGELOG.md", "w") as fh:
        fh.write("# Changelog\n%s\n\n%s" % (new_changelog, old_changelog))

    # Configures git to enable `git commit`.
    _configure_git()

    # Commits all changes in files and tags the new commit.
    shell("git add pyproject.toml CHANGELOG.md")
    shell(f'git commit --no-verify -m "Version {new_version} [skip ci]"')
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(bytes(f"{new_version}\n{new_changelog}", "utf-8"))
        fname = tmp_file.name
        tmp_file.flush()
        shell(f"git tag -f -a {new_version} -F {fname} --cleanup=whitespace")

    # `gitlab_origin` must have been defined in the CI pipeline before this is run.
    # The way to define `gitlab_origin` is something like the following:
    #   git remote add gitlab_origin https://$TOKEN_NAME:${ACCESS_TOKEN}@gitlab.com/drailab/${CI_PROJECT_NAME}.git  # noqa: E501
    shell(f"git push --follow-tags gitlab_origin HEAD:{MAIN_BRANCH}")
    print(green(f"Tag pushed. Latest version is {new_version}"))


# We do NOT use this with Gitlab for now.
def _upload_to_gcs(directory_path, bucket_name, prefix="/"):
    """Upload to GCS bucket from a directory.

    This uses a prefix in front of every uploaded file to the bucket.

    It is assumed a `GOOGLE_APPLICATION_CREDENTIALS` env var points to the file that
    stores the JSON Google Cloud auth key.
    """
    from google.cloud import storage

    print(green(f"Uploading docs to {os.path.join(bucket_name, prefix)}"))

    client = storage.Client()

    with _cwd(directory_path):
        rel_paths = glob.glob("./**", recursive=True)
        bucket = client.bucket(bucket_name)
        for local_file in rel_paths:
            remote_path = f'{prefix}{"/".join(local_file.split(os.sep)[1:])}'
            if os.path.isfile(local_file):
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(local_file)


def _set_package_version_in_pyproject(version: str):
    """Updates "tool.poetry.version" with the latest version."""
    with open("pyproject.toml", "r+") as fh:
        pyproject = tomlkit.load(fh)
        pyproject["tool"]["poetry"]["version"] = version
        fh.seek(0)
        fh.write(tomlkit.dumps(pyproject))


def _gen_conda_recipe(tag: str):
    """Create a conda recipe in the recipe folder.

    This script does the following:
    1. Edits pyproject.toml so that conda-only dependencies are in the package metadata.
    2. Builds a pip distribution of the package.
    3. Uses grayskull to generate a recipe from the pip distribution.
    4. Edits the recipe.

    At the end of this script, a recipe/meta.yaml file is generated.
    """
    # Includes the conda-only dependencies in the package metadata, and sets the package
    # version to be the value of `tag`.
    with open("pyproject.toml", "r+") as fh:
        pyproject = tomlkit.load(fh)
        orig_pyproject = copy.deepcopy(pyproject)

        pyproject["tool"]["poetry"]["version"] = tag
        pyproject["tool"]["poetry"]["dependencies"].update(
            pyproject["tool"]["conda-lock"]["dependencies"]
        )

        fh.seek(0)
        fh.write(tomlkit.dumps(pyproject))

    # Build the distribution so we can generate a recipe.
    conda_run("poetry build")

    # Revert the original pyproject file for people running this locally.
    with open("pyproject.toml", "w") as fh:
        fh.write(tomlkit.dumps(orig_pyproject))

    gz_path = next(pathlib.Path("dist").glob("%s*.gz" % utils.MODULE_NAME))
    alt_gz_path = pathlib.Path("dist") / gz_path.name.replace(
        utils.MODULE_NAME, utils.REPO_NAME
    )
    gz_path.rename(alt_gz_path)

    # Calls grayskull to create a Conda build recipe.
    with tempfile.TemporaryDirectory() as tmp_dir:
        conda_run(
            f"grayskull pypi {alt_gz_path} -s package source build requirements test "
            f"about -o {tmp_dir}"
        )
        meta_path = pathlib.Path(tmp_dir) / utils.REPO_NAME / "meta.yaml"
        pathlib.Path("recipe").mkdir(exist_ok=True)

        # Creates a build number distinguishing alpha/non-alpha versions to work around
        # an issue that Cloudsmith treats 0.0.x the same as 0.0.xa0.
        build_num = re.sub(r"[\D]", "", tag)

        # Edits the recipe and saves it as "recipe/meta.yaml".
        shell(
            "sed"
            # pip check will fail for conda dependencies, so remove it from tests.
            " -e '/pip check/d'"
            # Deletes this line of wrong syntax.
            " -e '/importlib-metadata >=4python/d'"
            f" -e 's/number: 0/number: {build_num}/g'"
            # Adds `poetry-core` to the `build` section.
            " -e 's/host:/host:\\n    - poetry-core/g'"
            # Adds license information.
            " -e 's/license_file: PLEASE_ADD_LICENSE_FILE/license: None/g'"
            f" {meta_path} > recipe/meta.yaml"
        )

        try:
            edit_conda_recipe = plugins.edit_conda_recipe
        except AttributeError:
            pass
        else:
            edit_conda_recipe("recipe/meta.yaml")

    print(green("Generated recipe/meta.yaml file."))
    shell_echo(">>>>>>>>>>>>>>>")
    shell("cat recipe/meta.yaml")
    shell_echo(">>>>>>>>>>>>>>>")


def conda_package(py_vers=("",)):
    """Build the package using boa (conda mambabuild).

    After successfully building the package, this function will automatically upload
    the built package to CloudSmith.
    """
    tag = _get_latest_tagged_version()

    if not tag:
        print(red("ERROR: At least one version tag is needed for conda packaging."))
        sys.exit(1)

    # Automatically generate a conda recipe if there is no recipe folder
    _gen_conda_recipe(tag)
    if isinstance(py_vers, (list, tuple)):
        py_vers = "".join(py_vers)
    py_vers = py_vers and " ".join(
        [("--python=" + ver.strip()) for ver in py_vers.split(",") if ver.strip()]
    )

    pathlib.Path("dist").mkdir(exist_ok=True)
    print("Running conda build...")
    shell_echo("Running conda build...")
    private_channel = os.environ.get(DRAILAB_CONDA_REG_ENVAR)
    no_test = "" if os.environ.get(CONDA_BUILD_NO_TEST_ENVAR) is None else " --no-test"
    conda_run(
        f"conda mambabuild --output-folder=dist/conda{no_test}"
        f" --override-channels -c {private_channel} -c conda-forge recipe {py_vers}",
    )
    print("Packages built.")
    conda_run(
        "find dist/conda -name '*.tar.bz2' "
        "| xargs -n 1 cloudsmith push conda drailab/reg"
    )
    print(green(f"Conda package deployed. Version is {tag}."))


def docs():
    """Builds docs and publishes them to Gitlab pages."""
    tag = _get_latest_tagged_version()

    # Updates "tool.poetry.version" with the latest tag from the remote repo.
    # This will save us one more `git pull`.
    if tag:
        _set_package_version_in_pyproject(tag)
    else:
        print(red("ERROR: At least one version tag is needed for building docs."))
        sys.exit(1)

    shell("python dev/docs.py")
    shell("rm -rf public && mkdir -p public")

    # `KEEP_DOCS_VERSIONS`'s default value is 0, which means by default we only keeps
    # the most recent version of docs. If we really need to keep more versions, set the
    # value of `KEEP_DOCS_VERSIONS` to a number greater than one (1). NOTE: Value 1
    # only results in preserving a tar ball of the latest docs build in the repo, it
    # doesn't deploy more than 1 version of the docs onto the website.
    if KEEP_DOCS_VERSIONS:
        # Uses `base` environment to run because `pigz` is only installed in there.
        conda_run(
            f'sh -c "mkdir -p docs_builds/ && cd docs/_build/ && mv -f html {tag}'
            f' && tar -I pigz -cf {tag}.tar.gz {tag}"',
            env=utils.BASE_NAME,
        )
        shell(f"mv -f docs/_build/{tag}.tar.gz docs_builds")

        # We only keep `KEEP_DOCS_VERSIONS` number of most recent docs versions.
        all_docs_fnames = list(glob.glob("docs_builds/*.tar.gz"))
        all_docs_fnames.sort(key=os.path.getmtime, reverse=True)
        for docs_fname in all_docs_fnames[int(KEEP_DOCS_VERSIONS) :]:
            shell(f"git rm {docs_fname}")

        shell("git add docs_builds")
        shell(f"git commit --no-verify -m 'Added docs version {tag}. [skip ci]'")
        shell(f"git push gitlab_origin HEAD:{MAIN_BRANCH}")
        for gz_fname in glob.glob("docs_builds/*.tar.gz"):
            conda_run(f"tar -I pigz -xf {gz_fname} -C public/", env=utils.BASE_NAME)
    else:
        shell(f"mv -f docs/_build/html public/{tag}")

    shell(f"mv -f public/{tag} public/latest")

    # If we're doing an official release, update the index to the latest version.
    if "a" not in tag:
        with open("public/index.html", "w") as fh:
            fh.write('<meta http-equiv="Refresh" content="0; url=\'latest\'" />')

    print(green(f"Docs uploaded. Version is {tag}"))


def pip_package():
    """Use poetry to publish to the pip-installable package index."""
    tag = _get_latest_tagged_version()

    # Updates "tool.poetry.version" with the latest tag from the remote repo.
    # This will save us one more `git pull`.
    if tag:
        _set_package_version_in_pyproject(tag)
    else:
        print(red("ERROR: At least one version tag is needed for pip packaging."))
        sys.exit(1)

    conda_run("poetry build")
    conda_run(
        "poetry config repositories.cloudsmith %s" % os.environ[PY_REG_CLOUDSMITH_ENVAR]
    )
    conda_run(
        "poetry config http-basic.cloudsmith %s %s"
        % (os.environ[CLOUDSMITH_API_USER_ENVAR], os.environ[CLOUDSMITH_API_KEY_ENVAR])
    )
    conda_run("poetry publish --repository cloudsmith -vvv -n", stdout=subprocess.PIPE)

    print(green("Pip package deployed."))


class _FixedWidthFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        kwargs["width"] = 88
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Core CI deployment commands.", formatter_class=_FixedWidthFormatter
    )
    subparsers = parser.add_subparsers(help="sub-command help", required=True)

    tag_parser = subparsers.add_parser(
        "tag",
        help="Bumps the version, generates the changelog, and tags the package.",
        formatter_class=_FixedWidthFormatter,
    )
    tag_parser.set_defaults(func=tag)

    docs_parser = subparsers.add_parser(
        "docs", help="Builds and uploads docs.", formatter_class=_FixedWidthFormatter
    )
    docs_parser.set_defaults(func=docs)

    pip_package_parser = subparsers.add_parser(
        "pip-package",
        help="Builds package and deploys to pip.",
        formatter_class=_FixedWidthFormatter,
    )
    pip_package_parser.set_defaults(func=pip_package)

    conda_package_parser = subparsers.add_parser(
        "conda-package",
        help="Builds package and deploys to conda.",
        formatter_class=_FixedWidthFormatter,
    )
    conda_package_parser.add_argument(
        "py_vers",
        type=str,
        nargs="*",
        default="",
        help="Python versions to build for. "
        "If not specified or an empty string is given, the version will default to the "
        "one currently used. If multiple versions are specified, they are separated "
        "by commas, exampleh 3.9,3.10. (default: '')",
    )
    conda_package_parser.set_defaults(func=conda_package)

    args = parser.parse_args()

    utils.validate_python()
    if CI_ENVAR not in os.environ:
        print(
            red(
                "This script can only be executed on Gitlab's CI/CD. If you want to"
                " deploy a package from your branch, tag your branch with an alpha"
                " version (e.g. `git tag 0.0.1a1`) and then push the tag (e.g."
                " `git push --tags`)."
            )
        )
        sys.exit(1)

    kwargs = vars(args)
    func = kwargs.pop("func")
    func(**kwargs)
