"""The command line reads an input file the way ``Calculator.from_file`` does.

Audit 2026-11-20b #26: ``defuqumat dos`` and ``pdos`` called ``run_scf`` with
their own ``--conv-thr`` (1e-8 whatever the input said) and dropped the rest of
``&electrons`` (``mixing_beta``, ``mixing_ndim``, ``electron_maxstep``), so the
command line and the front door converged the same file differently. After an
SCF that did not converge, both wrote a density of states from it and exited 0.
And every subcommand but ``size`` looked for the pseudopotentials beside the
input only, never in ``&control``'s ``pseudo_dir``.

No SCF runs here: ``run_scf`` and the steps after it are replaced by fakes that
record what they were handed.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat import cli

pytestmark = pytest.mark.unit

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo" / "Si.pz-vbc.UPF"

_INPUT = """ &control
    calculation = 'scf'{control}
 /
 &system
    ibrav=2, celldm(1) =10.20,
    nat=2, ntyp=1,
    ecutwfc=12.0
 /
 &electrons
    conv_thr = 1.0d-10, mixing_beta = 0.2, mixing_ndim = 4, electron_maxstep = 37
 /
ATOMIC_SPECIES
 Si  28.086  Si.pz-vbc.UPF
ATOMIC_POSITIONS (alat)
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS
  1
   0.250000  0.250000  0.250000   1.00
"""


def _input(tmp_path, beside=True, control=""):
    path = tmp_path / "si.in"
    path.write_text(_INPUT.format(control=control))
    if beside:
        shutil.copy(PSEUDO, tmp_path / PSEUDO.name)
    return path


class _Recorder:
    """A fake ``run_scf``: records its keywords, returns a result of one state."""

    def __init__(self, converged):
        self.converged = converged
        self.calls = []

    def __call__(self, system, pseudos, **options):
        self.calls.append((system, pseudos, options))
        return SimpleNamespace(converged=self.converged, accuracy=3.2e-4,
                               iterations=37, total_energy=-15.8, density=None)


def _never(*args, **options):
    raise AssertionError("a derived quantity was computed from an unconverged SCF")


@pytest.mark.parametrize("command", ["dos", "pdos"])
def test_the_input_electrons_namelist_reaches_the_scf(tmp_path, monkeypatch, command):
    path = _input(tmp_path)
    scf = _Recorder(converged=False)
    monkeypatch.setattr(cli, "run_scf", scf)
    monkeypatch.setattr(cli, "run_dos", _never)
    monkeypatch.setattr("defuqumat.workflows.pdos.run_pdos", _never)

    assert cli.main([command, str(path)]) == 1
    (_, pseudos, options), = scf.calls
    assert len(pseudos) == 1
    assert options["conv_thr"] == pytest.approx(1e-10)
    assert options["mixing_beta"] == pytest.approx(0.2)
    assert options["mixing_ndim"] == 4
    assert options["max_iterations"] == 37

    # and the command line still wins over the file
    scf.calls.clear()
    assert cli.main([command, str(path), "--conv-thr", "1e-6"]) == 1
    assert scf.calls[0][2]["conv_thr"] == pytest.approx(1e-6)


def test_a_converged_dos_runs_its_nscf_at_the_same_threshold(tmp_path, monkeypatch):
    path = _input(tmp_path)
    scf = _Recorder(converged=True)
    handed = {}

    def run_dos(system, pseudos, density, **options):
        handed.update(options)
        dos = SimpleNamespace(scheme="smearing", fermi_energy=None)
        return dos, SimpleNamespace(kpoints=SimpleNamespace(nk=1))

    monkeypatch.setattr(cli, "run_scf", scf)
    monkeypatch.setattr(cli, "run_dos", run_dos)
    monkeypatch.setattr(cli, "write_dos", lambda output, dos: None)
    assert cli.main(["dos", str(path)]) == 0
    assert handed["conv_thr"] == pytest.approx(1e-10)


def test_the_pseudopotentials_are_found_in_the_inputs_pseudo_dir(tmp_path, monkeypatch):
    """``pseudo_dir`` relative to the input file, as ``pw.x`` and ``from_file`` read it."""
    (tmp_path / "pp").mkdir()
    shutil.copy(PSEUDO, tmp_path / "pp" / PSEUDO.name)
    path = _input(tmp_path, beside=False, control="\n    pseudo_dir = 'pp'")

    scf = _Recorder(converged=False)
    monkeypatch.setattr(cli, "run_scf", scf)
    assert cli.main(["dos", str(path)]) == 1
    assert len(scf.calls[0][1]) == 1

    relaxed = []

    def run_relax(system, pseudos, **options):
        relaxed.append(pseudos)
        return SimpleNamespace(
            nsteps=0, converged=True, total_energy=-15.8, forces=np.zeros((2, 3)),
            system=system, positions=np.zeros((2, 3)))

    monkeypatch.setattr(cli, "run_relax", run_relax)
    assert cli.main(["relax", str(path)]) == 0
    assert len(relaxed[0]) == 1
