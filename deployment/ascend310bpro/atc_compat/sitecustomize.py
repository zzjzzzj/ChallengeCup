"""Compatibility aliases for CANN/ATC modules under NumPy 2.x."""

from __future__ import annotations

import numpy as np


def _alias(name: str, value: object) -> None:
    if name not in np.__dict__:
        setattr(np, name, value)


_alias("float_", np.float64)
_alias("complex_", np.complex128)
_alias("unicode_", np.str_)
_alias("string_", np.bytes_)
_alias("float", float)
_alias("int", int)
_alias("complex", complex)
_alias("bool", bool)
_alias("object", object)
_alias("str", str)
_alias("Inf", np.inf)
_alias("Infinity", np.inf)
_alias("NaN", np.nan)
_alias("NINF", -np.inf)
