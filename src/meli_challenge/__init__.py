"""Top-level package for the Mercado Libre Data Analytics challenge.

This module exposes convenience imports and shared constants so that the
rest of the codebase can simply do:

```
from meli_challenge import __version__
```

or import the high-level pipeline helpers once they are implemented in
`data_prep`, `segmentation`, or `performance`.
"""

from __future__ import annotations

import importlib

__all__ = [
    "__version__",
    "lazy_import",
]


def lazy_import(module_name: str):
    """Import a module on demand and return it.

    This helper prevents circular imports when notebooks/scripts only need
    specific utilities at runtime.  It is intentionally simple so it can be
    reused across the package.
    """

    return importlib.import_module(module_name)


# Lightweight version tag – updated automatically if the package is installed
# as a wheel, otherwise defaults to a dev label.
try:  # pragma: no cover - best-effort metadata lookup
    from importlib.metadata import version

    __version__ = version("meli_challenge")
except Exception:  # pragma: no cover
    __version__ = "0.0.0-dev"

