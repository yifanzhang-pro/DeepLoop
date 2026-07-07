from __future__ import annotations

import errno
from collections.abc import Iterable
from typing import Final, cast

import _pytest.pathlib as pytest_pathlib

_EXTRA_SCANDIR_IGNORED_ERRNOS: Final[tuple[int, ...]] = (
    errno.EACCES,
    errno.EPERM,
)


def _extend_pytest_scandir_ignored_errors() -> None:
    """Treat unreadable symlink targets as ignorable collection entries."""
    ignored_errors_obj = getattr(pytest_pathlib, "_IGNORED_ERRORS", ())
    ignored_errors = tuple(int(code) for code in cast(Iterable[int], ignored_errors_obj))
    missing_errors = tuple(code for code in _EXTRA_SCANDIR_IGNORED_ERRNOS if code not in ignored_errors)
    if not missing_errors:
        return
    setattr(pytest_pathlib, "_IGNORED_ERRORS", ignored_errors + missing_errors)


_extend_pytest_scandir_ignored_errors()
