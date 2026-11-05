"""The selected columns of the density matrix (SCDM): trial orbitals without a guess.

Damle, Lin and Ying, J. Chem. Theory Comput. 11, 1463 (2015), arXiv:1408.4926
(isolated bands); Damle and Lin, Multiscale Model. Simul. 16, 1392 (2018),
arXiv:1703.06958 (entangled bands, the weight ``f(e)``); transcribed from QE 7.5
``pw2wannier90.f90:compute_amn_with_scdm`` (lines 5279-5800) and its input checks
(lines 705-880). The automatic ``mu`` and ``sigma`` are not QE's: they are the
protocol of Vitale et al., npj Comput. Mater. 6, 66 (2020), arXiv:1909.00433,
Eqs. 16, 22, 23.

**What QE does.** At the Gamma point the states of the window are put on the FFT
grid of the wavefunctions (``dffts``) and weighted,

    X_nr = f(e_n) conj(psi_n(r)) / ||psi_n||,        (num_bands, nr1 nr2 nr3)

``||psi_n||`` the norm on the grid. The QR factorisation with column pivoting of
``X`` (LAPACK ``ZGEQP3``) selects ``num_wann`` grid points ``r_j``, the first
pivots. At every k-point of the mesh

    A_mn(k) = f(e_mk) conj(psi_mk(r_j)) / ||c_mk||,
    psi_mk(r) = sum_G c_mk(G) e^{i (k + G) . r},

computed as a slow Fourier sum at the ``num_wann`` points, and replaced by its
polar factor ``U V^H`` (``ZGESVD``). ``f`` is ``1`` (``isolated``),
``erfc((e - mu)/sigma)/2`` (``erfc``) or ``exp(-(e - mu)^2/sigma^2)``
(``gaussian``), ``e``, ``mu``, ``sigma`` in eV.

**Layout and ties.** The columns are taken in QE's order, the first grid index
fastest (``psic`` is Fortran-ordered; this code's boxes are C-ordered), so a tie
between two columns of equal norm, which the symmetry of a crystal makes common
at Gamma, is broken the same way as far as the two LAPACKs agree. The point of a
pivot is ``(i/nr1, j/nr2, k/nr3) - ANINT(...)``, the image nearest the origin,
with Fortran's ``ANINT`` (half away from zero): that image is the cell the
function is centred in.

**Ultrasoft and PAW.** QE uses the pseudo-wavefunction on the grid with no
augmentation and no ``S``, and so does this module (Vitale et al., supplement:
"no adaptation has been applied for the ultrasoft case").

**The automatic parameters** (Vitale et al.): the projectability ``p_nk`` onto
the Loewdin-orthonormalised pseudo-atomic orbitals
(:mod:`defuqumat.wannier.pao`), block-averaged over each degenerate multiplet
(rule D4), is fitted against the band energy with ``erfc((e - mu_fit) /
sigma_fit) / 2`` (least squares, every solved band at every k), and
``mu = mu_fit - 3 sigma_fit``, ``sigma = sigma_fit``, ``num_wann = N_PAO``.

**Where it runs.** The values at the pivots are one contraction per k-point on
the device (:func:`~defuqumat.batching.map_k`, dtypes those of the states); the
Gamma transform is :func:`~defuqumat.basis.fft.g_to_r`. The QR with pivoting
and the ``num_bands x num_wann`` SVDs are host LAPACK (``scipy.linalg.qr(...,
pivoting=True)``, ``numpy.linalg.svd``), as in QE, which runs ``ZGEQP3`` on one
process.

**Memory.** ``X`` is ``num_bands x nr1 nr2 nr3`` complex on the host, QE's
``psi_gamma``, plus the factor LAPACK builds in place of it: for silicon at
``ecutwfc = 12`` (``16^3``, 16 bands) 1 MB; for a 64-atom cell with 300 bands on
``100^3`` 4.8 GB. The per-k part holds ``npwx x num_wann`` phases at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["SCDMChoice", "ENTANGLEMENTS", "check_parameters", "occupation", "fit_erfc",
           "select_columns", "scdm_projections"]

#: ``scdm_entanglement``'s three values (``pw2wannier90.f90:868-874``).
ENTANGLEMENTS = ("isolated", "erfc", "gaussian")

#: Vitale et al.'s ``kappa`` in ``mu = mu_fit - kappa sigma_fit`` (their Eq. 23).
KAPPA = 3.0


@dataclass(frozen=True)
class SCDMChoice:
    """What the column selection used and found.

    ``mu`` and ``sigma`` in eV (``None`` for ``isolated``); ``mu_fit`` and
    ``sigma_fit`` the erfc fit of the projectability when the parameters were
    chosen automatically; ``pivots`` the 0-based column indices in QE's flat
    order (first index fastest) on ``grid``; ``points`` ``(num_wann, 3)`` their
    crystal coordinates, the image nearest the origin.
    """

    entanglement: str
    mu: float | None
    sigma: float | None
    mu_fit: float | None
    sigma_fit: float | None
    grid: tuple
    pivots: np.ndarray
    points: np.ndarray
    singular_min: float


def check_parameters(entanglement, mu, sigma) -> None:
    """QE's input checks, by name, before any state is solved."""
    if entanglement not in ENTANGLEMENTS:
        raise ValueError(f"scdm_entanglement = {entanglement!r} is not one of "
                         f"{', '.join(ENTANGLEMENTS)}")
    if sigma is not None and not float(sigma) > 0.0:
        raise ValueError(f"scdm_sigma = {sigma} must be positive (eV)")
    if (mu is None) != (sigma is None):
        raise ValueError("give scdm_mu and scdm_sigma together, or neither for the automatic "
                         "choice from the projectability")
    if entanglement == "isolated" and mu is not None:
        raise ValueError("scdm_mu and scdm_sigma are for scdm_entanglement = 'erfc' or "
                         "'gaussian'; 'isolated' weighs every band by 1")


def occupation(energies_ev: np.ndarray, entanglement: str, mu, sigma) -> np.ndarray:
    """``scdm_occupation``: the weight of each state, energies in eV."""
    e = np.asarray(energies_ev, dtype=float)
    if entanglement == "isolated":
        return np.ones_like(e)
    x = (e - float(mu)) / float(sigma)
    if entanglement == "erfc":
        from scipy.special import erfc
        return 0.5 * erfc(x)
    return np.exp(-x * x)


def fit_erfc(energies_ev: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """``(mu_fit, sigma_fit)`` of ``p = erfc((e - mu)/sigma)/2``, least squares.

    The start is the energy at which the sorted ``p`` first drops below 1/2 and
    a sigma of 1 eV. A fit that fails, or a ``sigma_fit`` that is not positive,
    is refused: the projectability then does not fall off within the solved
    bands.
    """
    from scipy.optimize import curve_fit
    from scipy.special import erfc

    e = np.asarray(energies_ev, dtype=float).ravel()
    y = np.asarray(p, dtype=float).ravel()
    order = np.argsort(e)
    below = np.nonzero(y[order] < 0.5)[0]
    if len(below) == 0:
        raise ValueError("the projectability never drops below 1/2 in the solved bands, so its "
                         "erfc fit has no edge; raise num_bands")
    mu0 = float(e[order][below[0]])

    def model(x, mu, sigma):
        return 0.5 * erfc((x - mu) / sigma)

    try:
        (mu_fit, sigma_fit), _ = curve_fit(model, e, y, p0=(mu0, 1.0), maxfev=10000)
    except RuntimeError as error:  # scipy's "Optimal parameters not found"
        raise ValueError(f"the erfc fit of the projectability failed ({error}); raise "
                         "num_bands or give scdm_mu and scdm_sigma") from None
    if not (np.isfinite(mu_fit) and np.isfinite(sigma_fit)) or sigma_fit <= 0.0:
        raise ValueError(f"the erfc fit of the projectability gave sigma = {sigma_fit:.4g} eV; "
                         "raise num_bands or give scdm_mu and scdm_sigma")
    return float(mu_fit), float(sigma_fit)


def _anint(x: np.ndarray) -> np.ndarray:
    """Fortran's ``ANINT``: the nearest integer, half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _grid_of(states) -> tuple:
    basis = states.calculation.basis
    gvectors = basis.smooth if hasattr(basis, "smooth") else basis.dense
    return tuple(int(n) for n in gvectors.grid)


def select_columns(states, bands, weights_gamma: np.ndarray, num_wann: int, gamma: int):
    """The QR with column pivoting at Gamma: ``(pivots, points, grid)``.

    ``bands`` the solved bands taken (after ``exclude_bands``), ``weights_gamma``
    their ``f(e)`` at the Gamma point ``gamma`` of the mesh.
    """
    from scipy.linalg import qr

    from defuqumat.basis.fft import g_to_r
    from defuqumat.basis.gvectors import _fft_index

    grid = _grid_of(states)
    n1, n2, n3 = grid
    coefficients = jnp.asarray(states.coefficients[gamma])[jnp.asarray(bands)]
    valid = jnp.asarray(states.valid[gamma])
    coefficients = jnp.where(valid[None, :], coefficients, jnp.zeros((), coefficients.dtype))
    index = _fft_index(jnp.asarray(states.miller[gamma]), grid)
    box = np.asarray(g_to_r(coefficients, index, grid))  # (nb, n1, n2, n3), C order
    # QE's psic: the first index fastest.
    psi = box.transpose(0, 3, 2, 1).reshape(len(bands), n1 * n2 * n3)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2, axis=1))
    x = psi.conj() * (np.asarray(weights_gamma, dtype=float) / norm)[:, None]
    _, pivots = qr(x, mode="r", pivoting=True)
    pivots = np.asarray(pivots[:num_wann], dtype=np.int64)
    i = pivots % n1
    j = (pivots // n1) % n2
    k = pivots // (n1 * n2)
    points = np.stack([i / n1, j / n2, k / n3], axis=1)
    return pivots, points - _anint(points), grid


def scdm_projections(states, bands, energies_ev: np.ndarray, num_wann: int, *,
                     entanglement: str = "isolated", mu=None, sigma=None, points=None,
                     k_batch=None, mu_fit=None, sigma_fit=None):
    """``A`` ``(num_bands, num_wann, nk)`` in Wannier90's layout, and the choice.

    ``states`` on the uniform mesh (Gamma in it), ``bands`` the solved bands
    taken, ``energies_ev`` ``(nk, num_bands)`` their energies. ``points``
    (crystal, ``(num_wann, 3)``) skips the selection and uses those points, which
    is how QE's own pivots are fed in for a comparison.
    """
    from defuqumat.batching import map_k

    if states.npol != 1:
        raise NotImplementedError("SCDM projections of spinor states")
    bands = np.asarray(bands, dtype=np.int64)
    kcrystal = np.asarray(states.kcrystal, dtype=float)
    gammas = np.nonzero(np.all(np.abs(kcrystal - np.rint(kcrystal)) < 1.0e-8, axis=1)
                        & np.all(np.abs(kcrystal) < 1.0e-8, axis=1))[0]
    if len(gammas) == 0:
        raise ValueError("SCDM selects its columns at the Gamma point, which is not on the mesh")
    gamma = int(gammas[0])
    weights = occupation(energies_ev, entanglement, mu, sigma)  # (nk, nb)
    if points is None:
        pivots, points, grid = select_columns(states, bands, weights[gamma], num_wann, gamma)
    else:
        points = np.asarray(points, dtype=float).reshape(num_wann, 3)
        pivots, grid = np.full(num_wann, -1, dtype=np.int64), _grid_of(states)

    coefficients = jnp.asarray(states.coefficients)[:, jnp.asarray(bands)]
    rdtype = jnp.real(coefficients[:1, :1, :1]).dtype
    r = jnp.asarray(points, dtype=rdtype)  # the same for every k-point
    two_pi = 2.0 * math.pi

    def at_points(one):
        c, miller, kvec, valid = one
        c = jnp.where(valid[None, :], c, jnp.zeros((), c.dtype))
        angle = two_pi * (jnp.asarray(miller, dtype=rdtype) @ r.T + (kvec @ r.T)[None, :])
        phase = jax.lax.complex(jnp.cos(angle), jnp.sin(angle)).astype(c.dtype)
        values = c @ phase  # (nb, num_wann): psi_mk(r_j)
        norm = jnp.sqrt(jnp.sum(jnp.real(jnp.conj(c) * c), axis=1))
        return values / norm[:, None]

    values = np.asarray(map_k(at_points, (coefficients, jnp.asarray(states.miller),
                                          jnp.asarray(kcrystal, dtype=rdtype),
                                          jnp.asarray(states.valid)), batch=k_batch))
    nk, nb = weights.shape
    amn = np.zeros((nb, num_wann, nk), dtype=values.dtype)
    smallest = np.inf
    for k in range(nk):
        x = (values[k] * weights[k][:, None]).conj()  # (nb, num_wann) = nowfc^H
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        smallest = min(smallest, float(s.min() / s.max()) if s.max() > 0 else 0.0)
        amn[:, :, k] = u @ vh
    choice = SCDMChoice(entanglement=entanglement,
                        mu=None if mu is None else float(mu),
                        sigma=None if sigma is None else float(sigma),
                        mu_fit=mu_fit, sigma_fit=sigma_fit, grid=grid, pivots=pivots,
                        points=points, singular_min=smallest)
    return amn, choice
