"""A brief description of this module.

More detailed description of this module.
The brief description must be a SINGLE line after the triple-quotes, and it must end
with a period. The detailed description is optional and can be multi-lined.
"""

########################################################################################
# Use 88-char line wdith

import contextlib
import os

from pathlib import Path
from typing import List, Optional, Tuple

#####
# The following is template code for creating a library. Follow the rules for coding
# styles and quality control. Write succinct and correct code. Happy coding.


def use_snake_name_for_functions(a: int, b: str) -> float:
    """Does some calculation.

    The docstring of a function must start with a SINGLE line brief description after
    the triple-quotes, and it should end with a period. The docstring follows the
    Google style.

    Docstring should document what the function does, including its expected input and
    output data. It should NOT document how it does the computation unless such
    information is critical for understanding the output data.

    Args:
      a: Parameter description
      b: Parameter description

    Returns:
      Returned object description

    Yields:
      For generator

    Raises:
      KeyError: If this function raises exceptions, describe them here.
    """
    # Implementation details should be put into comments instead of docstrings.
    # FIXME: Add a `FIXME:` comment when there is a defect that cannot be fixed for the
    # time being. There is no need to distinguish between a `FIXME` and a `TODO`, and so
    # for simplicity always use `FIXME`s.
    return float(a + b)


class UseCamelNamesForClasses:
    """A base class."""

    ...


class MyClass(UseCamelNamesForClasses):
    """Brief SINGLE line description.

    More detailed description is optional and can be multi-lined.

    Attributes:
      attribute: Description of this public attribute
    """

    def __init__(self, *args, **kwargs):
        """Initialize an instance of `MyClass`.

        Args:
          ...
        """
        super().__init__(*args, **kwargs)

    def _private_method(self):
        # Docstring for a private method is optional.
        ...

    def method(self, *args, **kwargs) -> list:
        """Brief SINGLE line description.

        Args:
          ...

        Returns:
          ...

        Raises:
          ...
        """
        return [1, 2, 3]

    @property
    def attribute(self):
        """A cool data."""
        # Do NOT use """Returns a cool data.""" for the docstring.
        return 1234


#####
# The following is extended template code for creating a Python application.
# Use `typer` library to quickly and beautifully define the command line user interface.
# Update: Prefer using `ama.cli` module to define the command line user interface.


def main():
    """Define a `main` fuction if there is only one command."""
    ...


if __name__ == "__main__":
    typer.run(main)
