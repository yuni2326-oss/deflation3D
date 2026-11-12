# Reference-free verification and repair of physics-informed neural eigensolvers for three-dimensional structures

Code, committed data and figures for a study of **sequential physics-informed neural eigensolvers** on
three-dimensional linear elasticity. A network is trained to minimize the Rayleigh quotient of one mode at a
time; each earlier mode is removed from the trial space by a *deflation rule*. The study traces the failures
of the higher modes to three layers and checks each from the solver's own outputs:

- the **instrument**, the quadrature on which the Rayleigh quotient is trained and measured: a refinement
  ladder of audit grids tells whether a stage eigenvalue is resolved;
- the **algebra**, how the stored modes are subtracted: the diagonal formula equals the exact projection only
  when the stored set is orthogonal in the quadrature in which the subtraction is applied;
- the **optimization**, which mode a stage settles on: an *order check* flags stage j when a later stage with a
  stable ladder reaches an audit eigenvalue below (1 − t) times that of stage j, and a *staged allocation*
  re-solves from the earliest flagged stage along further training paths, in a second and last round.

No step of the procedure reads a reference solution; the finite-element references only score it. The
procedure was developed on seven structures, its first pass was tested prospectively on an eighth, and the
staged allocation was then run unchanged on two held-out geometries. A strong-form (residual) solver with
the same network, clamp, audit grid and scoring is compared with the variational one.

Everything a number in the study rests on is in this repository: the finite-element references, every
neural run as one record per (arm or path, mode, seed) cell, the per-seed span diagnostics, the selections
and plans of the staged allocation, the post hoc readings, and the modules that compute every interval and
test from those records.

## Contents

```
eigen_benchmark/
  problems/p5_block.py      27-node hex FEM of the block: assembly, sparse/dense eigensolve,
                            Richardson ladder, tensor Gauss quadrature, audit-grid certification;
                            multi-block L- and T-brackets and the two-layer block (composite
                            quadrature, element-wise and pointwise Young's modulus); clamping at
                            x = 0 or at both ends
  neural/p5_neural.py       the sequential variational solver (vector MLP, hard clamp, four
                            deflation rules, Gauss-family or Monte Carlo quadrature, storage rules,
                            input normalization, Fourier input features, earlier stages taken from
                            the stored networks of other runs)
  neural/p5_pinn.py         the strong-form solver: interior residual, free-surface traction,
                            normalization and orthogonality penalties, a learned eigenvalue; the same
                            network, hard clamp, audit grid, ladder and scored outputs as above
  neural/deflation.py,      exact projection and W-orthonormalization used by the span
  neural/diagnostics.py       diagnostics
  drivers/run_p5.py         FEM references, cluster (degenerate-pair) detection, mode atlas
  drivers/run_p5_neural.py  experiment drivers: tau calibration, main grid, follow-ups, re-solving
                            one stage from stored networks, the staged allocation (paths, per-seed
                            selection, order-check plan, second round), the held-out geometries,
                            supervised fits for the representability probe
  drivers/p5_stats.py       every interval and test reported, recomputed from committed records,
                            including the selection rule and the order check
  drivers/p5_posthoc.py     post hoc readings from committed records only: distance of the audit
                            eigenvalue from the top of the ladder, the order check and re-ordering of
                            the outputs, the order check at lower thresholds, and the deepest drop
                            of a later stage below each stage
  drivers/p5_exploit.py     post hoc re-evaluation of stored networks: where the training grid misses
                            energy (by block and along the arms), its membrane, bending and
                            out-of-plane parts, stage shapes against FEM modes, Rayleigh-Ritz
                            extraction over the stored stages, the Green identity of the strong-form
                            operators, orthogonality defect, drift and span error of the stored
                            modes, the quotient that training minimized, and the representability
                            probe on a selected chain
  drivers/manifest.py       provenance (environment, git SHA at launch and at write time)
  render/figures_p5.py      figures, drawn from committed artifacts only (no FEM solve)
  quadrature.py, metrics.py shared numerics: Gauss-Legendre rules, MAC and classification
  tests/                    regression tests
docs/_generated/data/paper4/  committed artifacts (references, records, manifests, statistics)
docs/_generated/fig/paper4/   figures rendered from those artifacts
```

## Requirements

Python 3.12, NumPy 2.1, SciPy 1.17, mpmath 1.3, Matplotlib, pytest, and PyTorch 2.12 for the neural
solvers. The committed artifacts were produced in an NVIDIA NGC PyTorch 26.04 container on GB10
(Grace-Blackwell) nodes; each manifest records the exact environment it was produced in. A GPU is
needed only to rerun the neural experiments; the tests, the statistics and the figures run on a CPU.

## Quick start

```bash
PYTHONPATH=. python -m pytest eigen_benchmark/tests -q          # regression tests (CPU)
PYTHONPATH=. python -m eigen_benchmark.drivers.p5_stats         # rewrite p5_stats.json from records
PYTHONPATH=. python -c "from eigen_benchmark.drivers import p5_posthoc as p; p.main(); p.threshold_main(); p.order_drops_main()"
PYTHONPATH=. python -c "from eigen_benchmark.render import figures_p5 as f; [f.ALL[k]() for k in sorted(f.ALL)]"
```

`p5_stats` recomputes every interval from the committed records with a seed-cluster bootstrap
(B = 20,000, generator seed 12345) and should reproduce the committed `p5_stats.json` exactly. The
post hoc line reproduces `p5_posthoc.json`, `p5_posthoc_threshold.json` and
`p5_posthoc_order_drops.csv` exactly. The `audit_distance` block of `p5_posthoc.json` counts the stable
ladder cells of every record file present (8,180 of the variational solver), so a record file added
later changes it. The manifests written next to these files record the environment and the commit, so
they change.

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
Monte Carlo training quadrature (`mc_points=`) are options of `solve_p5_neural`. The storage practice of
orthogonalizing the stored outputs on the audit grid and subtracting them with the diagonal formula is the
`diagonal` arm with `storage="orthogonalized"`.

## Classification and fixed constants

Each cell is classified, in this order, as `invalid_quadrature` (Rayleigh quotient below the first
reference eigenvalue), `non_converged` (audit-grid tail variation above τ), `lower_mode_basin`,
`spurious`, or `correct` (matched mode equals the target, MAC ≥ 0.9, relative eigenvalue error ≤ 0.05).
A cell is *accurate* (`p5_stats.accurate`) when it meets the same three criteria without the convergence
gate; the lower-bound gate is kept. These rules and the constants below were fixed in code before the data
they judge were produced.

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
| selection | per seed, the candidate of lowest audit eigenvalue among those with tail spread ≤ 1e-3 | `p5_stats.E27_RULE` |
| order check | stage j (1–6) is flagged when a later stage, up to the look-ahead stage 7, has tail spread ≤ 1e-3 and an audit eigenvalue below (1 − 0.05) times that of stage j | `p5_stats.E30_ORDER_RULE` |
| staged allocation | round 1: paths `first` and `norm` at stages 1–6, `first` at stage 7; round 2, for seeds with a flag: `first`, `norm`, `ff` and `long` at the earliest flagged stage and the round-1 paths at the stages after it; the check is applied to the final chain, and a flag that remains is reported, not repaired | `run_p5_neural.E30_*`, `p5_stats.e30` |

The paths of the staged allocation are the first-pass setting (`first`), input normalization (`norm`),
fixed Fourier input features (`ff`) and a held-rate schedule (`long`: 80,000 iterations with the
learning-rate decay points of the first-pass schedule, so the low rate is held longer).

The structures carry their own references, audit grids, training families and τ, set in
`run_p5_neural`:

| Structure | Geometry | Audit grid | τ | Settings |
|---|---|---|---|---|
| baseline block | L = 1, b = 0.20, h = 0.04, clamped at x = 0 | (30, 15, 12) | 5.0 | `GRID_*`, `followup_settings` |
| thick block | the baseline block with h = 0.10 | (30, 15, 12) | 5.0 | `e26_settings("thick")` |
| clamped–clamped block | the baseline block clamped at x = 0 and x = L | (30, 15, 12) | 5.0 | `e26_settings("clamped2")`, `e28_settings`, `e30_settings` |
| square section | b = h = 0.10 | (30, 14, 14) | 0.5 | `e16_settings` |
| near-degenerate section | b = 0.105, h = 0.10 | (30, 14, 14) | 0.5 | `followup_settings("e18_*")`, `e26_settings("near")` |
| L-bracket | arm 1 [0, 1] × [0, 0.2], arm 2 [0.8, 1] × [0.2, 0.7], h = 0.1 | (24, 12, 12) | 10 | `e21_settings`, `e21b_settings`–`e21h_settings` |
| two-layer block | E = 1 below and 0.1 above z = h/2, h = 0.10 | (30, 15, 6), points per layer | 20 | `e22_settings` |
| T-bracket | stem [0, 0.8] × [0, 0.2], crossbar [0.8, 1] × [−0.25, 0.45], h = 0.1 | (26, 12, 12) | 10 | `e25_settings` |
| held-out block | the clamped–clamped block with h = 0.07 | (30, 15, 12) | 5.0 | `e31_settings("cc07")` |
| held-out L-bracket | the L-bracket with arm 2 [0.8, 1] × [0.2, 0.6] | (42, 12, 12) | 10 | `e31_settings("lb04")` |

For the L- and T-brackets a quadrature grid (n_L, n_W, n_T) counts points per unit length along each arm,
across the width and through the thickness; for the two-layer block the third entry counts points per
layer. Uniform Monte Carlo training quadrature is available on every structure: on the brackets the
blocks are drawn in proportion to their volume, and on the two-layer block the points fall anywhere in the
box, with Young's modulus evaluated point by point. Input normalization divides the
coordinates the network sees by the bounding box of the domain (`input_scale`; the L-bracket from
`e21g_settings`); the hard clamp keeps the physical coordinate.
When the top ladder rung would not fit in node memory, the ladder is measured in point chunks
(`ladder_chunk`); point-wise states are independent, so the ladder is the same.
`oracle_fit` (the supervised-fit control) takes another structure's reference, held-out check grids
and the stored modes of another run to project against (`e21f_settings`, `e31p_settings`).
`grid_v2(stage_from=)` keeps the earlier modes as the networks stored by another run and re-solves
only the later stages (`solve_p5_neural(stored=)`); `p5_neural.MixedStoredStage` takes each stored stage
from the run selected for it, which is how the staged allocation passes its selections on. The random
stream is advanced by the samples the stored stages drew, so the new stage starts exactly as it did in the
sequential run (`e21h_settings`: a fourfold schedule, or width 256). `checkpoint_every=` writes
intermediate networks of the new stage and `save_hist=` its audit-grid Λ history (`p5_hist_<tag>.npz`).

## Committed artifacts

`docs/_generated/data/paper4/` holds, per run tag:

- `p5_neural_<tag>_records.jsonl`: one JSON record per (arm, condition, mode, seed) cell, with the
  outcome, matched mode, MAC, eigenvalue error, projected mass and quadrature-ladder values
- `p5_neural_<tag>.csv`: per-mode summary rows; `*_partial*` files and `p5_<tag>_progress.json` are
  per-mode checkpoints of a run in progress
- `p5_span_per_seed_<tag>.csv`, `p5_span_diagnostics_<tag>.csv`: stored-span Gram diagnostics
- `manifest_p5_neural_<tag>.json`: environment, settings and provenance of the run
- `p5_hist_<tag>.npz`: audit-grid Λ every 25 iterations of the stages a run solved (runs with `save_hist`)
- `p5_reference*.{csv,json}`: FEM references; `p5_stats.json`: every reported interval and test
- `p5_e<nn>_<structure>_selection.json`, `p5_e<nn>_<structure>_r<round>_selection.json` and
  `p5_e<nn>_<structure>_plan.json`: the per-seed selections of the staged runs and the order-check plans
- `p5_posthoc*.{json,csv}`: the post hoc readings of `p5_posthoc`

Run tags: `v2` (main grid), `e2_*`–`e14*` (follow-up experiments), `e15r`/`e15b` (storage
convention under Monte Carlo quadrature), `e16sq_*` (square section), `e17_*` (training quadrature:
placement × redraw period), `e18_*` (near-degenerate section), `e19_*` (torsion–axial inversion:
budget and input normalization), `e20r_b4` (fourfold Monte Carlo sample), `e21lb_*` (L-bracket),
`e21lb_exact_g1` (L-bracket, family redrawn at every iteration), `e21lb_exact_dense` (L-bracket,
denser family with audit grid (30, 15, 12)), `e21lb_exact_mc1` (L-bracket, uniform Monte Carlo points
redrawn at every iteration), `e21lb_exact_hires` (L-bracket, high-resolution family with audit grid
(48, 24, 20)), `e21lb_exact_mc1_norm` and `e21lb_exact_norm` (L-bracket, input normalization with
uniform Monte Carlo points redrawn at every iteration and with the Gauss family),
`e21lb_s6_norm_x4`, `e21lb_s6_raw_x4` and `e21lb_s6_norm_w256` (L-bracket, only the mode-6 stage
re-solved from the stored networks of `e21lb_exact_mc1_norm` or `e21lb_exact_mc1`: a fourfold
schedule with and without input normalization, and width 256 with normalization), `e22bi_*`
(two-layer block).

Later runs:

| Tag | Run |
|---|---|
| `e21lb_s6_norm_x4lo`, `e21lb_s6_norm_ff` | L-bracket, the mode-6 stage re-solved from `e21lb_exact_mc1_norm`: the first-pass decay points over 80,000 iterations, and Fourier input features |
| `e24_pinn_thick`, `e24_pinn_lb`, `e28_pinn_thick_w10` | strong-form solver on the thick block and the L-bracket; the thick block with free-surface traction weight 10 (pilots in `p5_pilot_e24_*.csv`) |
| `e25tb_diag`, `e25tb_exact`, `e25tb_protocol` | T-bracket: the original setting with the diagonal and the exact rule, and Monte Carlo points redrawn at every iteration with exact projection and input normalization |
| `e26_thick_mc`, `e26_cc_mc`, `e26_near_mc`, `e26_bi_mc`, `e26_tb_raw_mc`, `e27_sq_mc` | first passes with uniform Monte Carlo points redrawn at every iteration and exact projection, raw coordinates: thick block, clamped–clamped block, near-degenerate section, two-layer block, T-bracket, square section |
| `e21lb_s6_raw_ff`, `e15b_s6_raw_ff` | the mode-6 stage re-solved with Fourier input features from `e21lb_exact_mc1` and from `e15b` |
| `e26_cc_mc_s2_ff`, `e26_cc_mc_s3_ff`, `e26_near_mc_s5_ff` | one stage re-solved with Fourier input features from the first pass |
| `e28_cc_s<k>_<path>` | clamped–clamped block, every stage on the paths `first`, `ff`, `long`, `norm`, and four re-initializations at stages 2 and 3; selection in `p5_e28_cc_selection.json` |
| `e28_lb_s6_reinit<r>`, `e28_nr_s6_<path>`, `e28_tb_s<k>_ff` | L-bracket stage 6 re-initialized four times; near-degenerate section stage 6 after the selection at stage 5; T-bracket, every stage of `e26_tb_raw_mc` re-solved with Fourier input features |
| `e29_lb_s7`, `e29_nr_s7` | one stage beyond the last reported mode, from the stored stages of `e21lb_exact_mc1` and `e26_near_mc` |
| `e30_<cc,nr>_r<round>_s<k>_<path>` | staged allocation on the clamped–clamped block and the near-degenerate section (`p5_e30_throughput.json`: measured cost per iteration) |
| `e31_<cc07,lb04>_fp`, `e31_<cc07,lb04>_r1_s<k>_<path>` | held-out geometries: first pass and staged allocation |
| `e31_b3_cc_hr` | clamped–clamped block, the same budget spent evenly: the held-rate schedule with 49,400 iterations per stage, no check and no selection |
| `e31r_lb04_s6_<ff,long>` | held-out L-bracket, stage 6 re-solved post hoc on the chain selected in the first round |
| `e32_tb_orth_mc`, `e32_b283_orth_mc`, `e32_b283_exact_mc` | storage practice on the T-bracket; baseline block with 283 Monte Carlo points under the storage practice and under the exact projection |

Supervised fits are in `p5_e1_oracle_fit*.csv` (among them the L-bracket in `_e21f_m{6,8}`, and the
held-out L-bracket in `_e31p_lb04_m{6,7}` and in `_e31q_lb04_m{6,7}`, refitted with the networks kept), the post
hoc exploitation analyses in `p5_exploit_*.csv` (among them the membrane, bending and out-of-plane split
in `p5_exploit_strain_e21.csv` and `p5_exploit_strain_e21g.csv`, stage shapes against FEM modes in
`*_mode6*.csv` and `*_stage_mac.csv`, Rayleigh-Ritz extraction in `p5_exploit_rr_*`, the Green identity
of the strong-form operators in `p5_exploit_green_*.csv`, drift and its closed form in
`p5_exploit_e32_drift_*.csv`, the quotient that training minimized in `p5_exploit_e32_mechanism.csv` and
`p5_exploit_e32_decomposition.csv`, and the probe on the chain selected in the first round in
`p5_exploit_e31q_probe_chain.csv`), and the mode atlases of the L-bracket, the two-layer block and the
T-bracket in `p5_mode_{shapes,atlas}_<structure>.*`.
Trained network parameters (about 60 MB per run) and the intermediate networks of the re-solved
stages are not included. The post hoc analyses of `p5_exploit` and the projection step of the supervised
fits re-read them, so their outputs are committed here but cannot be regenerated from this repository
alone; the networks can be retrained from the committed seeds and settings.

Manifests written before the launch-provenance field existed record the repository head at write
time rather than at launch. Inside the containers used here `git status` is refused (the repository
is owned by another user), so the `git_dirty` written at the top level of a manifest is not reliable;
read the `launch` field, which carries the commit and the dirty flag passed in by the runner. FEM
reference manifests carry `launch` from the L-bracket onward.

## License

To be added by the repository owner.
