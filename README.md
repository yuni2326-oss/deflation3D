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
                            Richardson ladder, tensor Gauss quadrature, audit-grid certification
  neural/p5_neural.py       the sequential neural eigensolver (vector MLP, hard clamp, four
                            deflation rules, Gauss-family or Monte Carlo quadrature, storage rules)
  neural/deflation.py,      exact projection and W-orthonormalization used by the span
  neural/diagnostics.py       diagnostics
  drivers/run_p5.py         FEM references, cluster (degenerate-pair) detection, mode atlas
  drivers/run_p5_neural.py  experiment drivers: tau calibration, main grid, follow-ups
  drivers/p5_stats.py       every interval and test reported, recomputed from committed records
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
training family and τ (0.5); see `run_p5_neural.e16_settings`.

## Committed artifacts

`docs/_generated/data/paper4/` holds, per run tag:

- `p5_neural_<tag>_records.jsonl`: one JSON record per (arm, condition, mode, seed) cell, with the
  outcome, matched mode, MAC, eigenvalue error, projected mass and quadrature-ladder values
- `p5_neural_<tag>.csv`: per-mode summary rows; `*_partial*` files are per-mode checkpoints
- `p5_span_per_seed_<tag>.csv`, `p5_span_diagnostics_<tag>.csv`: stored-span Gram diagnostics
- `manifest_p5_neural_<tag>.json`: environment, settings and provenance of the run
- `p5_reference*.{csv,json}`: FEM references; `p5_stats.json`: every reported interval and test

Run tags: `v2` (main grid), `e2_*`–`e14*` (follow-up experiments), `e15r`/`e15b` (storage
convention under Monte Carlo quadrature), `e16sq_*` (square section). Trained network parameters
(about 60 MB per run) are not included.

Manifests written before the launch-provenance field existed record the repository head at write
time rather than at launch.

## License

To be added by the repository owner.
