"""P5 강형식 물리정보 신경 고유해석(E24) — 자유 경계, 강형식 연산자, 순차 풀이, 드라이버 연결.

연산자는 해석 장으로 손 유도와 비교하고, 망 시행함수에서는 Green 항등식(∫ε:C:ε = −∫u·∇·σ + ∫_Γ u·σn)으로 자유면·법선·
잔차·견인력을 함께 검증한다 — 하나라도 틀리면 양변이 어긋난다.
"""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eigen_benchmark.neural import p5_neural as p5n     # noqa: E402
from eigen_benchmark.neural import p5_pinn as pin       # noqa: E402
from eigen_benchmark.quadrature import gauss_legendre   # noqa: E402

BOX = {"L": 1.0, "b": 0.2, "h": 0.04, "E": 1.0, "rho": 1.0, "nu": 0.29}
L_GEO = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
T_GEO = {"kind": "T", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.25, "E": 1.0, "rho": 1.0, "nu": 0.29}


def _boxes(g):
    L, b, h = g["L"], g["b"], g["h"]
    if g.get("kind") == "L":
        a = g["a"]
        return [(0, 0, L - b, b), (L - b, 0, L, b), (L - b, b, L, b + a)], h
    if g.get("kind") == "T":
        a = g["a"]
        return [(0, 0, L - b, b), (L - b, 0, L, b), (L - b, -a, L, 0), (L - b, b, L, b + a)], h
    return [(0, 0, L, b)], h


def _inside(g, P, tol=0.0):
    boxes, h = _boxes(g)
    P = np.atleast_2d(P)
    ok = np.zeros(len(P), bool)
    for x0, y0, x1, y1 in boxes:
        ok |= ((P[:, 0] >= x0 - tol) & (P[:, 0] <= x1 + tol) & (P[:, 1] >= y0 - tol)
               & (P[:, 1] <= y1 + tol) & (P[:, 2] >= -tol) & (P[:, 2] <= h + tol))
    return ok


@pytest.mark.parametrize("g,want", [
    (BOX, 0.2 * 0.04 + 2 * 1.0 * 0.04 + 2 * 1.0 * 0.2),
    (L_GEO, (2 * 1.0 + 0.2 + 2 * 0.5) * 0.1 + 2 * (1.0 * 0.2 + 0.2 * 0.5)),
    (T_GEO, (2 * 1.0 + 0.2 + 4 * 0.25) * 0.1 + 2 * (1.0 * 0.2 + 2 * 0.2 * 0.25)),
])
def test_free_faces_tile_the_free_boundary_with_outward_unit_normals(g, want):
    """고정면 x = 0을 뺀 모든 면이 한 번씩 — 면적의 합이 자유 경계 면적이고, 면 중심에서 법선 쪽은 밖, 반대쪽은 안이다."""
    F = pin.free_faces(g)
    assert sum(pin.face_area(f) for f in F) == pytest.approx(want, rel=1e-12)
    for o, e1, e2, n in F:
        assert np.linalg.norm(n) == pytest.approx(1.0)
        assert abs(n @ e1) < 1e-15 and abs(n @ e2) < 1e-15
        c = o + 0.5 * e1 + 0.5 * e2
        assert not _inside(g, c + 1e-6 * n)[0] and _inside(g, c - 1e-6 * n)[0]
        assert not (abs(c[0]) < 1e-12 and n[0] < 0)                # 고정면은 없다
    two = pin.free_faces(BOX, "two_face")
    assert len(two) == len(pin.free_faces(BOX)) - 1


def test_boundary_sample_lies_on_the_free_faces_and_weighs_their_area():
    rng = np.random.default_rng(3)
    P, N, w = pin.boundary_sample(3000, rng, "cpu", L_GEO)
    P, N = P.numpy(), N.numpy()
    assert float(w.sum()) == pytest.approx(sum(pin.face_area(f) for f in pin.free_faces(L_GEO)), rel=1e-12)
    assert _inside(L_GEO, P, tol=1e-12).all()
    assert (~_inside(L_GEO, P + 1e-7 * N)).all() and _inside(L_GEO, P - 1e-7 * N).all()
    top = N[:, 2] > 0.5
    assert abs(top.mean() - 0.3 / 0.92) < 0.03          # 윗면 몫 = 0.3 / 전체 0.92


def test_strong_operators_match_a_hand_derived_field():
    """u = (sin x cos 2y, x²z, yz²)의 ∇·σ와 σ·n을 손으로 유도한 값과 비교한다."""
    nu, E = 0.29, 1.0
    lam_l, mu = pin.lame(E, nu)
    f = lambda X: torch.stack([torch.sin(X[0]) * torch.cos(2 * X[1]), X[0] ** 2 * X[2], X[1] * X[2] ** 2])
    x, y, z = 0.3, 0.7, -0.4
    X = torch.tensor([x, y, z], dtype=torch.float64)
    u, ds = pin.div_sigma_of(f, X, nu, E)
    lap = np.array([-5 * math.sin(x) * math.cos(2 * y), 2 * z, 2 * y])
    gdiv = np.array([-math.sin(x) * math.cos(2 * y), -2 * math.cos(x) * math.sin(2 * y) + 2 * z, 2 * y])
    assert np.allclose(ds.numpy(), mu * lap + (lam_l + mu) * gdiv, rtol=1e-12, atol=1e-14)
    assert np.allclose(u.numpy(), [math.sin(x) * math.cos(2 * y), x ** 2 * z, y * z ** 2])
    J = np.array([[math.cos(x) * math.cos(2 * y), -2 * math.sin(x) * math.sin(2 * y), 0.0],
                  [2 * x * z, 0.0, x ** 2], [0.0, z ** 2, 2 * y * z]])
    eps = 0.5 * (J + J.T)
    sig = lam_l * np.trace(eps) * np.eye(3) + 2 * mu * eps
    n = np.array([0.6, 0.0, 0.8])
    t = pin.traction_of(f, X, torch.tensor(n), nu, E)
    assert np.allclose(t.numpy(), sig @ n, rtol=1e-12, atol=1e-14)


def _face_quadrature(g, n=14):
    P, N, W = [], [], []
    s, ws = gauss_legendre(n, 0.0, 1.0)
    for o, e1, e2, nrm in pin.free_faces(g):
        S, T = np.meshgrid(s, s, indexing="ij")
        P.append(o + S.reshape(-1, 1) * e1 + T.reshape(-1, 1) * e2)
        N.append(np.tile(nrm, (n * n, 1)))
        W.append(np.outer(ws, ws).reshape(-1) * pin.face_area((o, e1, e2, nrm)))
    return (torch.as_tensor(np.vstack(P)), torch.as_tensor(np.vstack(N)), torch.as_tensor(np.concatenate(W)))


@pytest.mark.parametrize("g,grid", [(BOX, (20, 10, 8)), (L_GEO, (20, 10, 10))])
def test_green_identity_ties_residual_traction_and_faces_on_a_network_trial_field(g, grid):
    """망 시행함수 u = x·N(x)에서 ∫ε:C:ε = −∫u·∇·σ + ∫_Γ u·σn. 고정면에서 u = 0이라 그 면의 항은 없다."""
    ens = p5n.EnsembleVecMLP3D(2, width=8, depth=2, seed=4, device="cpu")
    pts, w = p5n.quadrature(grid, "cpu", g)
    bp, bn, bw = _face_quadrature(g)
    nu = g["nu"]
    for s in range(2):
        p = {k: v[s] for k, v in ens.params.items()}
        S = p5n._state_one(p, ens.base, pts, None, None, g)
        a = p5n.energy_ip(S, S, w, nu)
        u, ds = torch.vmap(lambda x: pin.div_sigma_point(p, ens.base, x, nu))(pts)
        vol = -(w * (u * ds).sum(-1)).sum()
        ub = torch.vmap(lambda x: pin._u_vec(p, ens.base, x))(bp)
        t = torch.vmap(lambda x, n_: pin.traction_point(p, ens.base, x, n_, nu))(bp, bn)
        bnd = (bw * (ub * t).sum(-1)).sum()
        assert float(vol + bnd) == pytest.approx(float(a), rel=1e-8)


def test_strong_solver_reports_the_keys_the_scorer_reads():
    out = pin.solve_p5_pinn(n_modes=2, n_seeds=2, iters=6, width=8, depth=2, snapshot_every=2, device="cpu",
                            geometry=BOX, audit_grid=(8, 5, 4), resample_every=2,
                            family_axes=((6, 7), (4,), (3,)), boundary_points=20, check_grids=[(10, 6, 5)])
    assert out["lam"].shape == (2, 2) and np.isfinite(out["lam"]).all()
    assert out["lam_ladder"].shape == (2, 2, 2) and np.array_equal(out["lam_ladder"][..., 0], out["lam"])
    assert len(out["shapes"]) == 2 and out["shapes"][0].shape == (2, 3, 8 * 5 * 4)
    assert (out["lam_param"] > 0).all() and out["loss_terms"].shape == (2, 2, 4)
    assert np.array_equal(out["orth_train"][:, 0], np.zeros(2)) and np.array_equal(
        out["projected_mass_fraction"][:, 0], np.ones(2))
    assert (out["orth_audit"][:, 1] >= 0).all() and out["span_diagnostics"][0] is None
    assert out["span_diagnostics"][1] is not None and out["arm"] == pin.ARM_NAME
    assert len(out["hist"][0]) == 4 and out["hist_grid"] == "audit"
    assert out["n_grids_used"] >= 2


def test_strong_solver_draws_the_interior_samples_in_the_variational_order(monkeypatch):
    """내부 표본의 생성기·순서는 변분형과 같다 — 같은 시드에서 두 해석기가 같은 훈련 표본 열을 본다(경계 표본은 따로 뽑는다)."""
    seen = {"pinn": [], "var": []}
    real = p5n.mc_quadrature
    key = ["pinn"]

    def spy(*a, **k):
        out = real(*a, **k)
        seen[key[0]].append(out[0].numpy().copy())
        return out
    monkeypatch.setattr(p5n, "mc_quadrature", spy)
    kw = dict(n_modes=2, n_seeds=1, iters=4, width=4, depth=1, device="cpu", audit_grid=(6, 4, 3),
              resample_every=1, mc_points=17, geometry=L_GEO)
    pin.solve_p5_pinn(**kw, boundary_points=5)
    key[0] = "var"
    p5n.solve_p5_neural(**{**kw, "grid": (6, 4, 3)})
    assert len(seen["pinn"]) == len(seen["var"]) == 8
    assert all(np.array_equal(a, b) for a, b in zip(seen["pinn"], seen["var"]))


def test_grid_v2_runs_the_strong_solver_and_records_its_extras(tmp_path):
    from eigen_benchmark.drivers import run_p5
    from eigen_benchmark.drivers import run_p5_neural as run
    d = str(tmp_path)
    run_p5.structure_reference("lq", L_GEO, ladder=[(10, 2, 1), (20, 4, 2), (40, 8, 4)],
                               n_modes=6, cert_grid=(16, 8, 6), outdir=d)
    kw = dict(outdir=d, quick=True, arms=("penalty",), control=False, device="cpu", tau=5.0, geometry=L_GEO,
              ref_name="p5_reference_lq", ref_level=(40, 8, 4), audit_grid=(16, 8, 6), check_grid=(20, 10, 8),
              mc_points=30, mc_resample_every=1, normalize_input=True, solver="strong", boundary_points=20,
              save_params=True)
    run.grid_v2(**kw, tag="pq")
    recs = [json.loads(x) for x in open(f"{d}/p5_neural_pq_records.jsonl", encoding="utf-8")]
    assert {r["mode"] for r in recs} == {1, 2} and all(r["arm"] == pin.ARM_NAME for r in recs)
    for key in ("lam_param", "res_train", "res_audit", "bc_train", "bc_audit", "orth_train", "orth_audit"):
        assert all(math.isfinite(r[key]) for r in recs), key
    man = json.load(open(f"{d}/manifest_p5_neural_pq.json", encoding="utf-8"))
    assert man["solver"] == "strong" and man["boundary_points"] == 20
    assert man["strong_form"]["loss_term_names"] == ["residual", "traction", "normalization", "orthogonality"]
    z = np.load(f"{d}/p5_params_pq.npz")
    assert {k.split("_")[0] for k in z.files} == {"penalty"}
    with pytest.raises(ValueError):
        run.grid_v2(**{**kw, "arms": ("exact",)}, tag="bad")
