"""Helper functions for inference module."""

from __future__ import annotations


def order_composite(sample: dict, composite: dict) -> dict:
    """Return ordered composite parameters."""
    ordered = {}

    remains = list(composite.items())
    while remains:
        site = sample | ordered
        i = remains.pop(0)
        name, (arg_names, func) = i
        if all(arg_name in site for arg_name in arg_names):
            ordered[name] = (arg_names, func)
        else:
            remains.append(i)

    return ordered
