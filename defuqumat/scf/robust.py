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
otherwise the step from the last kept input is halved and tried again. "Fell" is
measured against the **worst of the last** :attr:`Supervisor.memory` **kept
inputs**, Grippo, Lampariello and Lucidi's nonmonotone rule (SIAM J. Numer. Anal.
23, 707), and not against the last one: Anderson's ordinary path is not
monotone, and a monotone test rejected its steps of a factor two or three near
convergence, each rejection costing three to six diagonalisations (measured on
the four-cell cobalt helix at ``mixing_beta = 0.7``: 38 against plain Anderson's
27). The worst of a window cannot grow, so a run that runs away still fails the
test. The step
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
restart the history; switch on the LDOS preconditioner on a smeared cell; and
at the bottom, damped **Hamiltonian** mixing with an energy-only line search
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
#: ``magnetization`` (the magnetization's step cut to a quarter) and
#: ``short-history`` (four entries) were rungs until 2026-10-16 and are gone on
#: the bed's record: neither was the rung a run converged on in any sweep, and
#: ``short-history`` drove the PAW nickel cells from 1e-8 to 1e4 in the
#: residual. ``ldos`` is the one that rescued runs: the cobalt film and the
#: nickel slab under the flat fit, started on plain Anderson, converged on it.
RUNGS = ("restart", "ldos", "hamiltonian")

#: The bound on the Anderson least-squares conditioning: :class:`AndersonMixer`'s
#: own 1e12, and not Herbst and Levitt's 1e6, so that a run the supervisor never
#: cuts back is plain Anderson's to the bit. Their value was tried on D22
#: (2026-10-16) and bought nothing on the test bed: two-atom silicon at
#: ``mixing_beta = 0.1`` took 33 diagonalisations at 1e6 and at 1e9 and 54 at
#: 1e12 under the first, monotone acceptance test, and 14 at 1e12 under the
#: nonmonotone one, so the count followed the acceptance rule and not the bound.
CONDITION_LIMIT = 1.0e12
#: The bound after a stall. Herbst and Levitt's 1e2 unjammed Fe2CrGa; here a
#: restart clears the history and keeps the bound, since no cell of the bed has
#: measured the tighter one.
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
    #: Whether the cell has states at the Fermi level for the LDOS rung.
    smeared: bool = False
    #: Whether the LDOS preconditioner is already installed.
    ldos: bool = False
    #: Whether the merit is a lower bound: false under Methfessel-Paxton and
    #: Marzari-Vanderbilt smearing, whose occupations leave ``[0, 1]``. Then
    #: the energy clause is off, the acceptance is on the residual alone, and
    #: the bottom rung, whose line search is energy-only, is not taken.
    #: Measured on D22 (2026-10-16) on the fully relativistic PAW nickel of
    #: ``ni-tetragonal-relaxed-mae-paw.in``: under its own ``smearing = 'mv'``
    #: the merit went 5.4e-3 Ry *below* the converged energy at an input whose
    #: residual was 2.0, and the energy clause kept that input; the same cell
    #: under a Gaussian of the same width stayed above it over the same
    #: fourteen iterations.
    variational: bool = True
    #: How many halvings a step takes before the trial is kept regardless.
    max_cuts: int = 4
    #: The window, in kept inputs, a stall is judged over. Calibrated on the
    #: converged plain-Anderson traces of the bed (``tools/robust_mixing``, D22,
    #: 2026-10-16): the DFT+U and spin-orbit nickel cells sit on plateaus where
    #: the best residual does not halve for 15 to 30 iterations and then
    #: converge (69 to 115 iterations), so a window of 8 at a factor of 2 fired
    #: on eight of the 31 converged runs and threw their history away. At 20
    #: with no new best at all it fires on one of them, at its iteration 40 of
    #: 81, and on every run that failed (``h4-pbe`` at 60 and 61, the nickel
    #: DFT+U cell at 0.1 at 126). Raised to 30 after the supervisor's own runs:
    #: at 20 it fired on the PAW nickel's slow tail at 1e-8 against a
    #: ``conv_thr`` of 1e-10 (iterations 29 and 56), where plain Anderson
    #: converges by waiting.
    window: int = 30
    #: The best ``accuracy`` must fall by this factor over a window; 1 is any
    #: new best at all.
    progress: float = 1.0
    #: How many kept inputs the acceptance test compares against the worst of.
    #: Eight, from the A/B on D22 (2026-10-16): the PAW nickel under a Gaussian
    #: at 0.7 took 72 at 8, 127 at 3 (plain Anderson 81); the DFT+U nickel at
    #: 0.4 77 and 136, at 0.7 83 and not converged at 200 (Anderson 77 and 69);
    #: the four-cell cobalt helix at 1.0 28 and 29 (Anderson 31). A bound of ten
    #: times the best instead matched Anderson on the DFT+U cell and took 133 on
    #: the PAW one.
    memory: int = 8
    #: The window after the first stall: the run is known to be stuck, so a
    #: second full window proves nothing.
    window_after: int = 15

    #: The inner mixer's step length, which the supervisor may raise when every
    #: trial is kept at once: Herbst and Levitt's growing trial damping, as a
    #: ``beta``. ``None`` leaves the mixer's own alone (a preconditioner bakes
    #: ``beta`` in when it is built, so the driver sets this only for plain
    #: Anderson). It never falls below the input's ``beta`` and never rises above
    #: :attr:`beta_max`.
    beta: float | None = None
    beta_max: float = 0.8
    #: How many trials kept at once, running, before ``beta`` doubles.
    grow_after: int = 3

    scale: float = 1.0
    cuts: int = 0
    rung: int = -1
    _beta_floor: float | None = None
    _clean: int = 0
    _kept: tuple | None = field(default=None, repr=False)
    _proposal: tuple | None = field(default=None, repr=False)
    _merit: float | None = None
    _accuracy: float | None = None
    _ethr: float | None = None
    _best: tuple | None = field(default=None, repr=False)
    _best_accuracy: float = np.inf
    _accuracies: list = field(default_factory=list)
    _merits: list = field(default_factory=list)
    #: The kept inputs since the line search last started, for the
    #: nonmonotone test; unlike the two lists above, not cleared by a rung.
    _window_accuracies: list = field(default_factory=list)
    _window_merits: list = field(default_factory=list)
    _floored: int = 0
    _last: dict = field(default_factory=dict)
    #: The best residual and the last merit when the ladder last acted.
    _action_accuracy: float | None = None
    _action_merit: float | None = None

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
        recent = slice(-self.memory, None)
        residual_fell = accuracy < max(self._window_accuracies[recent])
        energy_fell = (self.variational and unsettled == 0
                       and merit < max(self._window_merits[recent]) - noise)
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
        doubles back towards the mixer's own step after every kept input, and
        returns to it at once after one kept at the floor.
        """
        if self.beta is not None:
            if self._beta_floor is None:
                self._beta_floor = self.beta
                self.beta_max = max(self.beta_max, self.beta)
            self._clean = self._clean + 1 if self.cuts == 0 else 0
            if self._clean >= self.grow_after and self.beta < self.beta_max:
                self.beta = min(self.beta_max, 2.0 * self.beta)
                self._clean = 0
            self._last["beta"] = self.beta
        if self._last.get("reason") == "floor":
            self._floored += 1
            # A step cut to the floor and kept says the smaller steps bought
            # nothing, so the next trial is the mixer's own again. Without this
            # the scale never recovered: on the PAW nickel cell under a
            # Gaussian the run sat at a residual of 3.3e-8 for sixty iterations
            # with every trial a 1e-5 fraction of a step, each the kept input to
            # within its own noise and so rejected (D22, 2026-10-16).
            self.scale = 1.0
        else:
            self._floored = 0
            self.scale = min(1.0, 2.0 * self.scale)
        self.cuts = 0
        self._kept, self._proposal = state, proposal
        self._merit, self._accuracy, self._ethr = merit, accuracy, ethr
        if accuracy < self._best_accuracy:
            self._best_accuracy, self._best = accuracy, state
        self._accuracies.append(accuracy)
        self._merits.append(merit)
        self._window_accuracies.append(accuracy)
        self._window_merits.append(merit)
        self._last["next_scale"] = self.scale
        return _combine(state, proposal, self.scale)

    def retry(self, proposal=None):
        """The next trial after a rejected one: the kept input's step, halved.

        ``proposal``, when given, replaces the kept input's step: it is the
        inner mixer's proposal after the rejected pair entered its history.
        """
        if proposal is not None:
            self._proposal = proposal
        if self.beta is not None and self._beta_floor is not None:
            self.beta = max(self._beta_floor, 0.5 * self.beta)
            self._clean = 0
            self._last["beta"] = self.beta
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
        if min(self._accuracies[-self.window:]) < self.progress * before:
            return False
        noise = self.nelec * (self._ethr or 0.0)
        return self._merits[-1] > self._merits[-1 - self.window] - 2.0 * noise

    def on_stall(self) -> str | None:
        self.window = min(self.window, self.window_after)
        return self._on_stall()

    def _on_stall(self) -> str | None:
        """What a stall does: the restart again if the last action paid, else the next rung.

        A rung is not used up by one stall. If, since the ladder last acted, the
        best residual fell tenfold or the merit fell by ten times its noise, the
        last action worked and the run only needs its history cleared again;
        only a stall that follows an action that bought nothing moves down the
        ladder. Measured on the DFT+U nickel cell of ``ni-kind1-force.in`` at
        ``mixing_beta = 0.4``, where each stall used to cost a rung: the whole
        ladder was gone by iteration 85, and the run spent the rest of its 200
        on the slow bottom rung while the restart rung had been working.
        """
        noise = self.nelec * (self._ethr or 0.0)
        paid = (self._action_accuracy is not None and (
            self._best_accuracy < 0.1 * self._action_accuracy
            or (self._merit is not None
                and self._merit < self._action_merit - 10.0 * noise)))
        self._action_accuracy, self._action_merit = self._best_accuracy, self._merit
        if paid:
            self._accuracies.clear()
            self._merits.clear()
            self._floored = 0
            self.scale, self.cuts = 1.0, 0
            return "restart"
        return self.next_rung()

    def next_rung(self) -> str | None:
        """The next rung that applies to this cell, taken; ``None`` past the last."""
        while self.rung + 1 < len(RUNGS):
            self.rung += 1
            name = RUNGS[self.rung]
            if name == "ldos" and (self.ldos or not self.smeared):
                continue
            if name == "hamiltonian" and not self.variational:
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
        self._window_accuracies.clear()
        self._window_merits.clear()
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
