# defuqumat

defuqumat is a plane-wave density-functional theory code in Python, and we run
it with a normal Quantum ESPRESSO `pw.x` input file. With it we compute the
ground state, the band structure and the density of states, and from there
forces, stress, magnetism, linear and nonlinear response, topological invariants
and tunnelling images. In this guide we take one quantity at a time, and for
each we give the equation, what the quantity means, and the few lines of Python
that compute it on a real crystal.

![The band structure of silicon in LDA, computed with defuqumat.](img/bands_silicon.png#only-light)
![The band structure of silicon in LDA, computed with defuqumat.](img/bands_silicon-dark.png#only-dark)

## Installing

```bash
git clone https://github.com/joselado/defuqumat
cd defuqumat
pip install -e ".[notebooks]"
```

You need Python 3.10 or newer, and `pip` installs JAX, NumPy, SciPy, Numba and
equinox. The `[notebooks]` part adds matplotlib and Jupyter, which you need for
the plots. We run every example in this guide from the root of the repository.

## A first calculation

We start with silicon in the diamond structure, with LDA:

```python
from defuqumat import Calculator

calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                            pseudo_dir="tests/data/pseudo")
scf = calc.get_scf()

print(f"converged in {scf.iterations} iterations")
print(f"total energy {scf.total_energy:.8f} Ry")
```

```
converged in 7 iterations
total energy -15.83929992 Ry
```

The file `si.in` is a normal `pw.x` input file. We read it with
`Calculator.from_file`, and `pseudo_dir` says where the pseudopotential files
are. The object `calc` is the crystal together with its pseudopotentials, and
every quantity in this guide is a method of this object, with a name that starts
with `get_`. If there is no self-consistent ground state yet, the method computes
it first, so this is also a valid script:

```python
calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                            pseudo_dir="tests/data/pseudo")
dos = calc.get_dos(grid=(12, 12, 12))   # runs the SCF, then the DOS
```

If you have the text of an input file and not a file, you can use
`Calculator.from_text(text, pseudo_dir=...)`, which works in the same way.

## How this guide is organized

- **Getting started** is how we compute a crystal of your own: the input file,
  the pseudopotentials, and the two convergence tests you always need.
- **Ground state** is the total energy and the density, the band structure, and
  the density of states.

Every page follows the same order. We first give the equation, then what it
means and which experiment measures it, then the code, then a figure, and at the
end the limits of the code. The code under each figure is the code that made the
figure. In the plots we use moderate meshes, so that each figure computes in a
few minutes.

## Accuracy, and where to find more

When Quantum ESPRESSO or Elk compute the same number, we compared the two codes
on the same input, and the table with these comparisons is in the
[README](https://github.com/joselado/defuqumat#readme). The
[notebooks](https://github.com/joselado/defuqumat/tree/master/notebooks) show
each feature on a real crystal, with more detail than this guide, and the file
[`notebooks/README.md`](https://github.com/joselado/defuqumat/blob/master/notebooks/README.md)
lists them by the property you want to compute.
