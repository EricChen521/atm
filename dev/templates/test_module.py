########################################################################################
# Use 88-char line wdith

import os

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest


@patch("os.remove", return_value=1)
@patch("os.makedirs")
def test_foo(mock_makedirs, mock_remove):
    """All tests start with `test_` prefix."""
    a = Path()
    b = MagicMock()
    c = MagicMock()
    b.assert_called_with(Path("dname") / "subdname")
    c.assert_not_called()
    b.reset_mock()
    b.assert_has_calls([call(a), call("EOB")])
    assert False
