"""A uniform orbital magnetic field: the conventions, Peierls on Wannier hoppings, the references.

:mod:`~defuqumat.magnetic.conventions` fixes the gauge, the sign of the charge, the
magnetic translations and the k labels once for every route; :mod:`~defuqumat.magnetic.
peierls` is the weak-field route on a torus and a ribbon; :mod:`~defuqumat.magnetic.landau`
holds the exact references (free Landau levels, the Dirac ladder); :mod:`~defuqumat.
magnetic.gipaw` the gauge-including dressing of a projector. The plan is
``notes/ORBITAL-FIELD-NEXT.md``.
"""

from defuqumat.magnetic.conventions import MagneticCell, minimum_image, refuse_regime, signed_area

__all__ = ["MagneticCell", "minimum_image", "refuse_regime", "signed_area"]
