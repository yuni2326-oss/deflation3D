# Basis-dependent deflation in sequential neural eigenanalysis of 3D elasticity

Code, committed data and figures for a study of **sequential Rayleigh-quotient neural
eigensolvers** on three-dimensional linear elasticity. A network is trained to minimize the
Rayleigh quotient of one mode at a time; each earlier mode is removed from the trial space by a
*deflation rule*. The study asks which deflation rules recover the low modes of a slender elastic
block in the correct order, why the others fail, and when the failure is an artifact of how the
Rayleigh quotient is measured.

Everything a number in the study rests on is in this repository: the finite-element references,
every neural run as one record per (arm, mode, seed) cell, the per-seed span diagnostics, and the
module that computes every interval and test from those records.

## Contents

```
eigen_benchmark/
  problems/p5_block.py      27-node hex FEM of the block: assembly, sparse/dense eigensolve,
                            Richardson ladder, tensor Gauss quadrature, audit-grid certification;
                            multi-block L-bracket and two-layer variants (composite quadrature,
                            element-wise and pointwise Young's modulus)
  neural/p5_neural.py       the sequential neural eigensolver (vector MLP, hard clamp, four
                            deflation rules, Gauss-family or Monte Carlo quadrature, storage rules)
  neural/deflation.py,      exact projection and W-orthonormalization used by the span
  neural/diagnostics.py       diagnostics
  drivers/run_p5.py         FEM references, cluster (degenerate-pair) detection, mode atlas
  drivers/run_p5_neural.py  experiment drivers: tau calibration, main grid, follow-ups
  drivers/p5_stats.py       every interval and test reported, recomputed from committed records
  drivers/p5_exploit.py     post hoc re-evaluation of stored networks: where the training grid misses
                            energy (by block and along the arms), its membrane, bending and
                            out-of-plane parts, stage shapes against FEM modes
  drivers/manifest.py       provenance (environment, git SHA at launch and at write time)
  render/figures_p5.py      figures, drawn from committed artifacts only (no FEM solve)
  quadrature.py, metrics.py shared numerics: Gauss-Legendre rules, MAC and classification
  tests/                    regression tests
docs/_generated/data/paper4/  committed artifacts (references, records, manifests, statistics)
docs/_generated/fig/paper4/   figures rendered from those artifacts
```

## Requirements

Python 3.12, NumPy 2.1, SciPy 1.17, mpmath 1.3, Matplotlib, pytest, and PyTorch 2.12 for the neural
solver. The committed artifacts were produced in an NVIDIA NGC PyTorch 26.04 container on GB10
(Grace-Blackwell) nodes; each manifest records the exact environment it was produced in. A GPU is
needed only to rerun the neural experiments; the tests, the statistics and the figures run on a CPU.

## Quick start

```bash
PYTHONPATH=. python -m pytest eigen_benchmark/tests -q          # regression tests (CPU)
PYTHONPATH=. python -m eigen_benchmark.drivers.p5_stats         # rewrite p5_stats.json from records
PYTHONPATH=. python -c "from eigen_benchmark.render import figures_p5 as f; [f.ALL[k]() for k in sorted(f.ALL)]"
```

`p5_stats` recomputes every interval from the committed records with a seed-cluster bootstrap
(B = 20,000, generator seed 12345) and should reproduce the committed `p5_stats.json` exactly.

## The four deflation rules (arms)

| Arm | Name in code | Rule |
|---|---|---|
| (a′) | `diagonal` | subtract Σ_j (⟨u, R_j⟩ / ⟨R_j, R_j⟩) R_j using the raw stored outputs R_j |
| (a″) | `reorth_diagonal` | re-orthogonalize the stored outputs immediately before use, then the diagonal formula |
| (b) | `exact` | exact mass projection onto the stored span (Gram solve) |
| (b′) | `qr` | the exact projection computed through a QR factorization |

The diagonal formula equals the exact projection only when the stored set is orthogonal in the
quadrature in which the subtraction is applied; with raw outputs the residual is Φ(I − D⁻¹G) and
depends on the stored basis. Storage conventions (`storage="raw"` or `"orthogonalized"`) and
Monte Carlo training quadrature (`mc_points=`) are options of `solve_p5_neural`.

## Classification and fixed constants

Each cell is classified, in this order, as `invalid_quadrature` (Rayleigh quotient below the first
reference eigenvalue), `non_converged` (audit-grid tail variation above τ), `lower_mode_basin`,
`spurious`, or `correct` (matched mode equals the target, MAC ≥ 0.9, relative eigenvalue error ≤ 0.05).
These rules and the constants below were fixed in code before the data they judge were produced.

| Constant | Value | Where |
|---|---|---|
| geometry | L = 1, b = 0.20, h = 0.04, ν = 0.29, E = ρ = 1, clamped at x = 0 | `p5_block.P5_GEOMETRY` |
| reference ladder | (12,4,2) → (24,8,4) → (48,16,8), Richardson | `run_p5.LADDER` |
| near-degeneracy rule | relative gap < 0.01 → scored as a cluster (subspace MAC) | `run_p5.GAP_RULE` |
| shape certification | adjacent-level subspace cos² ≥ 0.999 | `p5_block.SELF_MAC_THRESHOLD` |
| training quadrature | Gauss family nx 16–20 × ny 8–10 × nz 6–8, redrawn every 250 iterations | `p5_neural.FAMILY_*` |
| audit grid | (30, 15, 12); check grid (40, 20, 15) | `run_p5_neural.AUDIT_GRID` |
| main grid | 20 seeds, 20,000 iterations, 6 modes, width 128 × depth 5 | `run_p5_neural.GRID_*` |
| convergence gate τ | 5.0 (calibration seeds 9001–9008, 90th percentile rounded up) | `manifest_p5_tau.json` |
| ladder resolved | tail spread ≤ 1e-3 | `LADDER_RESOLVED` |

The square-section structure (b = h = 0.10) has its own reference, audit grid (30, 14, 14),
training family and τ (0.5); see `run_p5_neural.e16_settings`. Three further structures carry their
own settings in the same module:

| Structure | Geometry | Reference ladder | Audit grid | τ | Settings |
|---|---|---|---|---|---|
| near-degenerate section | b = 0.105, h = 0.10 | as the square section | (30, 14, 14) | 0.5 | `followup_settings("e18_*")` |
| L-bracket | arm 1 [0, 1] × [0, 0.2], arm 2 [0.8, 1] × [0.2, 0.7], h = 0.1 | `run_p5.L_LADDER` | (24, 12, 12) | 10 | `e21_settings`, `e21b_settings`–`e21f_settings` |
| two-layer block | E = 1 below and 0.1 above z = h/2, h = 0.10 | `run_p5.LADDER` | (30, 15, 6), points per layer | 20 | `e22_settings` |

For the L-bracket a quadrature grid (n_L, n_W, n_T) counts points per unit length along each arm, across
the width and through the thickness; for the two-layer block the third entry counts points per layer.
Uniform Monte Carlo training quadrature is available on the box and on the L-bracket (blocks drawn in
proportion to their volume), and the two-layer block refuses it. Input normalization divides the
coordinates the network sees by the bounding box of the domain (`input_scale`; the L-bracket from
`e21g_settings`); the hard clamp keeps the physical coordinate.
When the top ladder rung would not fit in node memory, the ladder is measured in point chunks
(`ladder_chunk`); point-wise states are independent, so the ladder is the same.
`oracle_fit` (the supervised-fit control) takes another structure's reference, held-out check grids
and the stored modes of another run to project against (`e21f_settings`).

## Committed artifacts

`docs/_generated/data/paper4/` holds, per run tag:

- `p5_neural_<tag>_records.jsonl`: one JSON record per (arm, condition, mode, seed) cell, with the
  outcome, matched mode, MAC, eigenvalue error, projected mass and quadrature-ladder values
- `p5_neural_<tag>.csv`: per-mode summary rows; `*_partial*` files are per-mode checkpoints
- `p5_span_per_seed_<tag>.csv`, `p5_span_diagnostics_<tag>.csv`: stored-span Gram diagnostics
- `manifest_p5_neural_<tag>.json`: environment, settings and provenance of the run
- `p5_reference*.{csv,json}`: FEM references; `p5_stats.json`: every reported interval and test

Run tags: `v2` (main grid), `e2_*`–`e14*` (follow-up experiments), `e15r`/`e15b` (storage
convention under Monte Carlo quadrature), `e16sq_*` (square section), `e17_*` (training quadrature:
placement × redraw period), `e18_*` (near-degenerate section), `e19_*` (torsion–axial inversion:
budget and input normalization), `e20r_b4` (fourfold Monte Carlo sample), `e21lb_*` (L-bracket),
`e21lb_exact_g1` (L-bracket, family redrawn at every iteration), `e21lb_exact_dense` (L-bracket,
denser family with audit grid (30, 15, 12)), `e21lb_exact_mc1` (L-bracket, uniform Monte Carlo points
redrawn at every iteration), `e21lb_exact_hires` (L-bracket, high-resolution family with audit grid
(48, 24, 20)), `e22bi_*` (two-layer block). Supervised fits of the L-bracket are in
`p5_e1_oracle_fit_e21f_m{6,8}.csv`, the post hoc exploitation analyses in `p5_exploit_*.csv` (the
membrane, bending and out-of-plane split in `p5_exploit_strain_e21.csv`), and the
mode atlases of the L-bracket and the two-layer block in `p5_mode_{shapes,atlas}_<structure>.*`.
Trained network parameters (about 60 MB per run) are not included. The post hoc analyses of
`p5_exploit` and the projection step of the supervised fits re-read them, so their outputs are
committed here but cannot be regenerated from this repository alone.

Manifests written before the launch-provenance field existed record the repository head at write
time rather than at launch. Inside the containers used here `git status` is refused (the repository
is owned by another user), so the `git_dirty` written at the top level of a manifest is not reliable;
read the `launch` field, which carries the commit and the dirty flag passed in by the runner. FEM
reference manifests carry `launch` from the L-bracket onward.

## License

To be added by the repository owner.
