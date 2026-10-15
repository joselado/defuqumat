"""``mixing_mode = 'robust'``: a supervisor around the mixers, for convergence first.

The plan and its review are ``plans/robust-mixing.md``. Three things make an SCF
fail, and the supervisor treats each by name rather than by a smaller ``beta``:
ill-conditioning (a preconditioner, chosen at setup from the cell), the
nonlinearity far from the solution (a line search on a merit function), and
Anderson's own stagnation (a restart with a tight bound on the conditioning of
its least-squares problem). This module is the decision logic and nothing else;
it holds no array that is not a state the driver hands it, and every decision it
takes is written into the iteration's ``history`` entry, so a run is
reproducible from its record.

**The merit function** is Herbst and Levitt's (arXiv:2109.14018, eq. 10),

    I(rho_in) = E_KS[P_out],

the Kohn-Sham free energy of the states and occupations the input's Hamiltonian
produces, every term at the *output* density, ``becsum`` and ``ns``. It is the
energy of an actual density matrix, so ``I >= E_min`` for every input, equal at
a self-consistent minimum. ``run_scf`` assembles it from the terms of its
converged branch, evaluated every iteration (:func:`merit_terms`); the mid-run
total the loop reports is not it, being ``E_H`` and ``E_xc`` at the *input*
density, a Harris-like value with no lower bound.

**The safeguard** (their Algorithm 1, in density mixing): a trial input is kept
when ``I`` fell by more than the eigensolver's noise, ``nelec * ethr``, or the
residual ``accuracy`` (``rho_ddot``, the number the run converges on) fell;
otherwise the step from the last kept input is halved and tried again. The step
is the one the inner mixer proposed at the kept input, ``mixed - x``, rescaled:
the Anderson coefficients depend on the residuals only (eq. 39), so a rescale is
exactly the step at a smaller damping, the inner mixer is called once per kept
input and never sees a rejected one, and nothing inside it changes, which is
what keeps every other mode bit-identical. Each trial is one diagonalisation and
one SCF iteration of the loop; ``history`` marks which were kept.

**The ladder.** A stall -- over the last ``window`` kept inputs the best
``accuracy`` has not halved *and* the merit has not fallen beyond its noise, or
the step has been cut to its floor ``window`` times running -- moves the run one
rung down, from the last kept input (or the best one, if the run has since
moved ten times above it in the residual). The merit clause is not optional: on
``benchmarks/fe-unstable.in`` at 4x4x4 the run leaves the nonmagnetic saddle
Anderson converges to, the energy falling by 5e-4 Ry an iteration while the
residual *rises*, which is the run going downhill to the ferromagnet, and a
residual-only test called it a stall and sent it back to the saddle:
restart the history with the conditioning bound at 1e2 (Herbst and Levitt's
Fe2CrGa); switch on the LDOS preconditioner on a smeared cell; cut the
magnetization's step to a quarter; shorten the history to four; and at the
bottom, damped **Hamiltonian** mixing with an energy-only line search
(:class:`HamiltonianDamping`), the one rung with a theorem under it. A run that
cannot lower the merit there at the smallest step stops and says so: the map is
not single-valued at the eigensolver's floor, and no mixer fixes that.

What is guaranteed and what is not is section 2 of the plan: the damped
iteration with an energy-only line search has a theorem under it (their
Theorems 1 and 2, for Hamiltonian mixing and monotone occupations); this
production scheme, with Anderson and the residual clause, has the measured
success rate of the plan's test bed, and nothing more is claimed for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np

__all__ = ["Supervisor", "HamiltonianDamping", "RUNGS", "vacuum_fraction", "ROBUST_MODES"]

#: The names ``run_scf`` reads as this mode.
ROBUST_MODES = ("robust",)

#: The ladder, in order. Each name is what ``history`` records when it is taken.
RUNGS = ("restart", "ldos", "magnetization", "short-history", "hamiltonian")

#: The bound on the Anderson least-squares conditioning: :class:`AndersonMixer`'s
#: own 1e12, and **not** Herbst and Levitt's 1e6. Measured on D22 (2026-10-16,
#: ``tools/robust_mixing``): at 1e6 the supervisor's graphene monolayer at
#: ``mixing_beta = 0.1``, which it never cut back, took 29 diagonalisations to
#: plain Anderson's 17, the residual falling by 0.82 an iteration from 4e-9 on,
#: because the trim drops the history the tail needs; the fit here is in
#: ``rho_ddot`` over unit-norm residuals, which is not their Gram matrix.
CONDITION_LIMIT = 1.0e12
#: The bound after a stall. Herbst and Levitt's 1e2 unjammed Fe2CrGa, but the
#: same measurement makes a tight bound cost the tail, so a restart here clears
#: the history and keeps the bound.
CONDITION_LIMIT_RESTART = 1.0e12


def vacuum_fraction(positions, lattice, radius: float = 3.0) -> float:
    """The largest fraction of the cell along one lattice vector that holds no atom.

    Measured on the atoms' fractional coordinates along each lattice vector: the
    widest gap between neighbouring planes of atoms, less ``2 radius`` (bohr) for
    the atoms' own extent, over the length of the cell along that vector. A bulk
    crystal reads 0, a slab with 30 bohr of vacuum in a 45 bohr cell about 0.5.
    Read on the geometry and not on a density, so it costs nothing and needs no
    grid.
    """
    lattice = np.asarray(lattice, dtype=float)
    fractional = np.asarray(positions, dtype=float) @ np.linalg.inv(lattice)
    # Columns of the inverse are the reciprocal vectors b_i, a_i . b_j = delta_ij.
    reciprocal = np.linalg.inv(lattice)
    best = 0.0
    for axis in range(3):
        # The spacing of the lattice planes normal to this reciprocal vector,
        # which is the length a fractional coordinate along it is measured in.
        height = 1.0 / np.linalg.norm(reciprocal[:, axis])
        coords = np.sort(np.mod(fractional[:, axis], 1.0))
        gaps = np.diff(np.concatenate([coords, coords[:1] + 1.0]))
        empty = max(0.0, float(gaps.max()) * height - 2.0 * radius)
        best = max(best, empty / height)
    return best


def _combine(a, b, scale):
    """``a + scale (b - a)`` over a state ``(rho, becsum, ns, tau)``, ``None`` kept."""
    return jax.tree_util.tree_map(lambda x, y: x + scale * (y - x), a, b)


@dataclass
class Supervisor:
    """The robust mode's line search and stagnation ladder.

    The driver calls :meth:`judge` once per diagonalisation with the trial's
    merit and residual, then either :meth:`kept` (with the inner mixer's
    proposal from this input) or :meth:`retry`, each returning the next input.
    """

    nelec: float
    #: Whether a magnetization exists for the magnetic rung to act on.
    magnetic: bool = False
    #: Whether the cell has states at the Fermi level for the LDOS rung.
    smeared: bool = False
    #: Whether the LDOS preconditioner is already installed.
    ldos: bool = False
    #: How many halvings a step takes before the trial is kept regardless.
    max_cuts: int = 4
    #: The window, in kept inputs, a stall is judged over.
    window: int = 8
    #: The best ``accuracy`` must fall by this factor over a window.
    progress: float = 0.5

    scale: float = 1.0
    cuts: int = 0
    rung: int = -1
    _kept: tuple | None = field(default=None, repr=False)
    _proposal: tuple | None = field(default=None, repr=False)
    _merit: float | None = None
    _accuracy: float | None = None
    _ethr: float | None = None
    _best: tuple | None = field(default=None, repr=False)
    _best_accuracy: float = np.inf
    _accuracies: list = field(default_factory=list)
    _merits: list = field(default_factory=list)
    _floored: int = 0
    _last: dict = field(default_factory=dict)

    @property
    def scale_floor(self) -> float:
        return 0.5 ** self.max_cuts

    def judge(self, merit: float, accuracy: float, ethr: float, unsettled: int) -> bool:
        """Whether the trial input just diagonalised is kept.

        The first input of a run is kept. After it, the trial is kept when the
        residual fell, or when the merit fell by more than the eigensolver's
        noise -- ``nelec`` times the looser of the two thresholds, which is
        about ``0.1 accuracy`` under ``pw.x``'s schedule, so the energy clause
        resolves after the first two iterations and the residual clause carries
        those. With bands left unsettled the merit is not trusted at all. And a
        step already cut :attr:`max_cuts` times is kept regardless, since a
        smaller one is not a different direction.
        """
        if self._kept is None:
            self._last = {"kept": True, "reason": "first"}
            return True
        noise = self.nelec * max(ethr, self._ethr or ethr)
        residual_fell = accuracy < self._accuracy
        energy_fell = unsettled == 0 and merit < self._merit - noise
        if residual_fell or energy_fell:
            reason = "residual" if residual_fell else "energy"
            keep = True
        elif self.cuts >= self.max_cuts:
            reason, keep = "floor", True
        else:
            reason, keep = "rejected", False
        self._last = {"kept": keep, "reason": reason, "scale": self.scale,
                      "merit_change": merit - self._merit,
                      "noise": noise}
        return keep

    def kept(self, state, proposal, merit, accuracy, ethr):
        """Record the kept input and the inner mixer's proposal from it; the next input.

        ``state`` and ``proposal`` are ``(rho, becsum, ns, tau)``, the input
        just diagonalised and what the inner mixer returned from it at its own
        damping. The next trial is the proposal at the current scale, which
        grows back towards the mixer's own step after a step that was not cut.
        """
        if self._last.get("reason") == "floor":
            self._floored += 1
        else:
            self._floored = 0
        if self.cuts == 0:
            self.scale = min(1.0, 2.0 * self.scale)
        self.cuts = 0
        self._kept, self._proposal = state, proposal
        self._merit, self._accuracy, self._ethr = merit, accuracy, ethr
        if accuracy < self._best_accuracy:
            self._best_accuracy, self._best = accuracy, state
        self._accuracies.append(accuracy)
        self._merits.append(merit)
        self._last["next_scale"] = self.scale
        return _combine(state, proposal, self.scale)

    def retry(self):
        """The next trial after a rejected one: the kept input's step, halved."""
        self.cuts += 1
        self.scale *= 0.5
        self._last["next_scale"] = self.scale
        return _combine(self._kept, self._proposal, self.scale)

    def stalled(self) -> bool:
        """Whether the last :attr:`window` kept inputs made too little progress."""
        if self._floored >= self.window:
            return True
        if len(self._accuracies) <= self.window:
            return False
        before = min(self._accuracies[:-self.window])
        if min(self._accuracies[-self.window:]) <= self.progress * before:
            return False
        noise = self.nelec * (self._ethr or 0.0)
        return self._merits[-1] > self._merits[-1 - self.window] - 2.0 * noise

    def next_rung(self) -> str | None:
        """The next rung that applies to this cell, taken; ``None`` past the last."""
        while self.rung + 1 < len(RUNGS):
            self.rung += 1
            name = RUNGS[self.rung]
            if name == "ldos" and (self.ldos or not self.smeared):
                continue
            if name == "magnetization" and not self.magnetic:
                continue
            self._accuracies.clear()
            self._merits.clear()
            self._floored = 0
            self.scale, self.cuts = 1.0, 0
            return name
        return None

    def restart_from_best(self):
        """Where the line search starts over: the last kept input, or the best one.

        The last kept input, unless its residual is ten times the best seen, in
        which case the run has gone somewhere worse and goes back. Not the best
        one always: the lowest residual can be a saddle the run was right to
        leave (see the module docstring).
        """
        start = (self._kept if self._accuracy is not None
                 and self._accuracy <= 10.0 * self._best_accuracy else self._best)
        self._kept = None
        self._proposal = None
        return start

    @property
    def record(self) -> dict:
        """What ``history`` records for this trial."""
        out = dict(self._last)
        out["rung"] = RUNGS[self.rung] if self.rung >= 0 else "none"
        return out


@dataclass
class HamiltonianDamping:
    """The bottom rung: ``H_{n+1} = H_n + alpha (H_KS(P_out(H_n)) - H_n)``, Armijo on ``I``.

    Herbst and Levitt's Theorems 1 and 2 (arXiv:2109.14018, section II) are for
    exactly this iteration: Hamiltonian mixing along the plain SCF direction,
    no Anderson and no preconditioner, the step accepted only when the merit
    fell. Along it the slope of ``I`` is ``<R_V, chi0 R_V> <= 0`` with no
    linearisation, which is why this rung mixes the Hamiltonian and the others
    the density (``plans/robust-mixing.md`` section 2). ``H`` is the triple the
    loop builds its Hamiltonians from, ``(v_scf, ddd_paw, v_ns)``, with ``None``
    for a piece the run does not have. ``alpha`` starts at ``alpha_max < 1``,
    which Theorem 2 needs.
    """

    alpha_max: float = 0.5
    #: Below this step a trial is kept regardless and counted towards the end.
    alpha_min: float = 1.0e-3
    #: Consecutive trials kept at ``alpha_min`` before the run is declared at
    #: the eigensolver's floor.
    patience: int = 6

    alpha: float = 0.5
    cuts: int = 0
    _h: tuple | None = field(default=None, repr=False)
    _r: tuple | None = field(default=None, repr=False)
    _merit: float | None = None
    _floored: int = 0
    _last: dict = field(default_factory=dict)

    def judge(self, merit: float, unsettled: int) -> bool:
        """Kept when the merit fell, or when the step is at its floor."""
        if self._h is None:
            self._last = {"kept": True, "reason": "first"}
            return True
        if unsettled == 0 and merit < self._merit:
            reason, keep = "energy", True
        elif self.alpha <= self.alpha_min:
            reason, keep = "floor", True
        else:
            reason, keep = "rejected", False
        self._last = {"kept": keep, "reason": reason, "scale": self.alpha,
                      "merit_change": merit - self._merit}
        return keep

    def kept(self, h_in, h_out, merit):
        """Record the kept Hamiltonian and its residual; the next trial."""
        self._floored = self._floored + 1 if self._last.get("reason") == "floor" else 0
        if self.cuts == 0 and self._h is not None:
            self.alpha = min(self.alpha_max, 2.0 * self.alpha)
        self.cuts = 0
        self._h = h_in
        self._r = jax.tree_util.tree_map(lambda o, i: o - i, h_out, h_in)
        self._merit = merit
        self._last["next_scale"] = self.alpha
        return self._step()

    def retry(self):
        self.cuts += 1
        self.alpha *= 0.5
        self._last["next_scale"] = self.alpha
        return self._step()

    def _step(self):
        return jax.tree_util.tree_map(lambda h, r: h + self.alpha * r, self._h, self._r)

    @property
    def at_floor(self) -> bool:
        """Whether the merit has not fallen at the smallest step :attr:`patience` times running."""
        return self._floored >= self.patience

    @property
    def record(self) -> dict:
        out = dict(self._last)
        out["rung"] = "hamiltonian"
        return out
