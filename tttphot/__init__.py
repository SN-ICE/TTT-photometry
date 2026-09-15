"""tttphot -- a photometry pipeline for TTT (Teide Observatory) transient imaging.

Adapted from tjophot, the TJO pipeline. It is instrument-agnostic at the
edges: everything that knows about a TTT camera lives in
``config/instruments/<name>.yaml``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
