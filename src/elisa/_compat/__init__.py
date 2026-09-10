"""Temporary compatibility fixes for upstream dependencies."""

from .lineax import apply_lineax_workaround as _apply_lineax_workaround

_apply_lineax_workaround()
