# `h_sc_dftu`: simple cubic hydrogen with a Hubbard U on 1s

The `h_sc` run with one block added, `dft+u` with `dftu = 1` (fully localised
limit), species 1, `l = 0`, `U = 0.1469972` Ha (4 eV), `J = 0`, so that
`STATE.OUT` carries the DFT+U record `vmatmt` after the four density and
potential pairs (`writestate.f90:122-125`). Written by Elk 11.0.2 on D22
(`/l/ladovj1/review/harmonics-elk/bin/elk`, one thread, 0.3 s), converged.

`scf.in` is the defuqumat input for the same cell with `U H-1s 4.0`.

| file | what it is for |
|---|---|
| `STATE.OUT`, `GEOMETRY.OUT` | the state the reader and the seed are tested on |
| `INFO.OUT` | `chgmt = 0.6054875069`, `chgir = 0.3945124931`, DFT+U energy 5.56126e-3 Ha |
| `VMATMT.OUT` | Elk's text print of `vmatmt`: 0.3306372221e-1 Ha in the one `l = 0` entry |
| `DMATMT.OUT` | the muffin-tin occupation, 0.5501448706 (both spins), which is **not** in `STATE.OUT` |

To regenerate, run Elk in a copy of this directory holding only `elk.in` and
`H.in`, and keep the files above.
