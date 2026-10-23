"""The order of the walked third derivative's loops (``OPEN.md`` Part XXIII item 6).

``walked_susceptibility_derivative`` walks the chunks outer and the tangents inner, so a
chunk's arguments (``field._arguments``, whose upload is the cost) and its stacked stores
are built once per chunk per pass and not once per tangent per chunk, and each tangent's
sums still run over the chunks in ``field.chunks`` order, which is what keeps every column
bit-identical to the tangent-outer walk. Here the compiled passes are replaced by
arithmetic with a record of the calls, so the order is pinned in milliseconds; the
numbers on real cells are in ``notes/speed/xxiii-6.md``.
"""

from types import SimpleNamespace

import numpy as np

import defuqumat.response.chunked_third as ct

NCHUNK, NTANGENT = 3, 4


def _run(monkeypatch):
    log = {"arguments": 0, "stacked": 0, "forward": [], "chunk": [], "global": []}

    class Field:
        solver = SimpleNamespace(nocc=1, occupied_counts=None, smearing=None)
        commutators = None
        bare = "bare"
        dpsi = "dpsi"
        dipole = None
        chunks = [(c, 1) for c in range(NCHUNK)]

        def _arguments(self, rows, live):
            log["arguments"] += 1
            return (rows,)

    def forward(rows, x, dx, occupied, block, stacked):
        log["forward"].append((rows, float(dx)))
        return np.array(rows + 1.0) * dx, np.array(rows + 2.0) * dx

    def global_(big, rowset, x, dx, rho, drho, sums, dsums):
        log["global"].append((float(dx), float(sums), float(dsums)))
        return (np.array(1.0), np.array(2.0), None, np.array(3.0)), (
            np.array(4.0), np.array(5.0), None, np.array(6.0))

    def chunk(rows, x, dx, occupied, block, bare, dpsi, stored, v, dv, ddd, dddd,
              dipole):
        log["chunk"].append((rows, float(dx)))
        return np.array(rows + 1.0) * dx, np.array(rows + 3.0) * dx

    passes = {"forward": forward, "global": global_, "chunk": chunk}
    monkeypatch.setattr(ct, "_third_passes", lambda *a, **k: passes)
    monkeypatch.setattr(ct, "hoisted", lambda calculation: None)
    monkeypatch.setattr(ct, "row_leaves", lambda calculation, rows: None)
    monkeypatch.setattr(ct, "_rows_of", lambda store, rows: np.zeros(2))

    def stacked(store, rows):
        log["stacked"] += 1
        return store

    monkeypatch.setattr(ct, "_stacked_rows", stacked)
    tangents = [(float(t + 1), "dpsi", None, np.zeros(1)) for t in range(NTANGENT)]
    columns = ct.walked_susceptibility_derivative(
        None, Field(), np.zeros(1), "strain", 0.0, tangents)
    return log, columns


def test_a_chunk_s_arguments_are_built_once_per_chunk_per_pass(monkeypatch):
    log, _ = _run(monkeypatch)
    # two passes over the chunks (forward and chunk walk), not two per tangent
    assert log["arguments"] == 2 * NCHUNK
    # dpsi in the forward walk, then bare, dpsi and stored in the chunk walk
    assert log["stacked"] == NCHUNK * (1 + 3)


def test_each_tangent_keeps_its_chunk_order(monkeypatch):
    log, columns = _run(monkeypatch)
    for walk in ("forward", "chunk"):
        for t in range(NTANGENT):
            rows = [r for r, dx in log[walk] if dx == t + 1.0]
            assert rows == list(range(NCHUNK))
        # the chunk is the outer loop: a chunk's calls are contiguous
        assert [r for r, _ in log[walk]] == sorted(r for r, _ in log[walk])
    assert len(columns) == NTANGENT
    for t, column in enumerate(columns):
        dx = t + 1.0
        # the sums are added in chunk order and reach the global step whole
        sums = sum(r + 1.0 for r in range(NCHUNK)) * dx
        assert log["global"][t] == (dx, sums, sum(r + 2.0 for r in range(NCHUNK)) * dx)
        energy = 1.0 + sum((r + 1.0) * dx for r in range(NCHUNK))
        denergy = 4.0 + sum((r + 3.0) * dx for r in range(NCHUNK))
        expected = -4.0 * ct.FPI * (denergy / 3.0 - energy * 6.0 / 3.0**2)
        assert column == expected
