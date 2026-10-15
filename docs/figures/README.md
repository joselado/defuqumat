# Figures of the user guide

Each figure of the user guide has its own folder here, `<id>/`. The folder has two
scripts and the input files they read. `compute.py` runs the calculation with
defuqumat and writes `data.npz`. `plot.py` reads only `data.npz` and writes
`docs/guide/img/<id>.png` (light theme) and `<id>-dark.png` (dark theme). `plot.py`
does not import defuqumat, so a figure can be replotted on any machine with
matplotlib. The website build never runs a calculation: `data.npz` is committed.

The `pw.x` input files of a figure are in its folder. The pseudopotentials are the
committed ones in `tests/data/pseudo/`, and `compute.py` builds that path from its
own location, so it runs from any directory.

The plan of the whole guide, with the list of figures, is `plans/user-guide-web.md`.

## Shared files

- `style.py`: the matplotlib style. A `plot.py` defines `draw(theme)` and ends with
  `style.main(draw, "<id>")`, which writes both PNG files.
- `common.py`: `common.start(params)` prints the parameters,
  `common.save(__file__, params, **arrays)` writes `data.npz` with the parameters,
  the git commit, the host and the wall time as JSON under `params`, and
  `common.load(path)` reads it back.
- `run.sh <id>`: runs `compute.py` with one thread (BLAS, OpenMP and
  `DEFUQUMAT_THREADS=1`), `PYTHONPATH` on this checkout, at most one job per entry
  of `FIGURE_CPUS` (each pinned to its own core), and a memory cap of 8 GB. It
  writes `<id>/compute.log`, and `<id>/DONE` or `<id>/FAILED` at the end.

## Recompute and replot

```bash
bash docs/figures/run.sh <id>          # compute, writes <id>/data.npz
python3 docs/figures/<id>/plot.py      # writes the two PNG files
```

Do not run a `compute.py` beside a test run on a shared machine. The wall time of
each figure is stored in its `data.npz` (`wall_time_s`), with the host.

## Add a figure to a page

````markdown
![One sentence that reads the physics off the plot.](img/<id>.png#only-light)
![One sentence that reads the physics off the plot.](img/<id>-dark.png#only-dark)

The paragraph that reads the figure.

??? example "Code of this figure (docs/figures/<id>/)"

    ```python title="compute.py"
    --8<-- "<id>/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "<id>/plot.py"
    ```
````

The collapsed block shows the two scripts on the website when the reader clicks on
it. `pymdownx.snippets` copies them in at build time from `docs/figures/`, so the
page always shows the code that is in git, and a wrong `<id>` fails the build. The
PDF prints the path instead of the code.

The website shows one image per theme, and the PDF (`docs/build_guide_pdf.py`)
keeps the light one. Check both with `mkdocs build` and
`python3 docs/build_guide_pdf.py /tmp/guide.pdf`.
