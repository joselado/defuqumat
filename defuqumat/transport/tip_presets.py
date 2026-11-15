"""Preset tips: the weight of each orbital from the real orbital of the apex atom.

``PLAN.md`` P179, ``plans/tip-orbitals.md`` section 12. Filled in step 6.
"""

from __future__ import annotations

__all__ = ["PRESETS", "preset"]

#: name -> the function that builds the preset tip
PRESETS: dict = {}


def preset(name: str):
    """The preset tip ``name`` as a :class:`~defuqumat.transport.tip.TipOrbital`."""
    if name not in PRESETS:
        raise ValueError(f"no preset tip {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]()
