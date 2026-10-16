"""P5 3D 신경 arm — Rayleigh 몫, 하드 BC, 세 deflation 규칙.

P4 2D 판본과 같은 규약을 따른다. 다른 점은 상태가 12성분(변위 3 + 기울기 9)이고,
하드 BC가 거리함수 구성 없이 D(x) = x로 닫혀 있다는 것뿐이다.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eigen_benchmark.neural import p5_neural as p5n     # noqa: E402
from eigen_benchmark.problems import p5_block as p5     # noqa: E402


def test_quadrature_matches_the_certified_grid_and_sums_to_the_volume():
    """§2.4가 정한 격자를 그대로 써야 한다 — 다른 격자를 쓰면 인증이 무의미하다."""
    pts, w = p5n.quadrature()
    g = p5.P5_GEOMETRY
    assert pts.shape == (16 * 8 * 6, 3)
    assert w.shape == (16 * 8 * 6,)
    vol = g["L"] * g["b"] * g["h"]
    assert abs(float(w.sum()) / vol - 1.0) < 1e-12


def test_trial_function_satisfies_the_clamped_condition_exactly():
    """φ = x·N이므로 x = 0에서 세 성분 모두 정확히 0이어야 한다 — 벌점이 아니다."""
    ens = p5n.EnsembleVecMLP3D(2, width=16, depth=2, seed=0, device="cpu")
    face = torch.zeros(5, 3, dtype=torch.float64)
    face[:, 1] = torch.linspace(0, 0.2, 5)
    S = p5n.state_batch(ens, face)
    assert S.shape[:2] == (2, 12)
    assert float(S[:, :3].abs().max()) < 1e-300


def test_rayleigh_quotient_reproduces_a_manufactured_fields_energy_ratio():
    """구적과 변형에너지 식이 맞는지 — 해석 장의 a/b를 독립 계산과 비교한다."""
    pts, w = p5n.quadrature()
    u, G = p5.manufactured_field(pts.numpy())
    a_ref, b_ref = p5._forms_from_field(u, G, w.numpy())
    S = torch.zeros(12, len(w), dtype=torch.float64)
    S[:3] = torch.as_tensor(u.T)
    S[3:] = torch.as_tensor(G.reshape(len(w), 9).T)
    rq = float(p5n.rq_from_state(S, w, p5.P5_GEOMETRY["nu"]))
    assert abs(rq / (a_ref / b_ref) - 1.0) < 1e-12, (rq, a_ref / b_ref)


def test_deflation_rules_all_remove_the_stored_span_from_the_field():
    """세 규칙 모두 저장 스팬 성분을 제거해야 한다 — 제거량의 정확도만 다르다."""
    pts, w = p5n.quadrature()
    rng = np.random.default_rng(0)
    P = torch.as_tensor(rng.normal(size=(2, 12, len(w))))
    S = P[0] * 1.7 + P[1] * (-0.4) + 0.05 * torch.as_tensor(
        rng.normal(size=(12, len(w))))
    for kind in ("exact", "diagonal", "qr"):
        out = p5n.project(S, P, w, kind)
        assert out.shape == S.shape
        leak = max(abs(float(torch.einsum("ij,j,ij->", P[a, :3], w, out[:3])))
                   for a in range(2))
        base = max(abs(float(torch.einsum("ij,j,ij->", P[a, :3], w, S[:3])))
                   for a in range(2))
        assert leak < base, (kind, leak, base)
    # 정확 사영은 기계정밀도까지 뺀다
    out = p5n.project(S, P, w, "exact")
    for a in range(2):
        num = abs(float(torch.einsum("ij,j,ij->", P[a, :3], w, out[:3])))
        den = float(torch.einsum("ij,j,ij->", P[a, :3], w, P[a, :3]))
        assert num / den < 1e-10, (a, num / den)


def test_gradient_components_follow_the_same_projection_coefficients():
    """사영은 선형이므로 도함수도 같은 계수로 따라가야 한다. 어긋나면 사영된 함수의
    Rayleigh 몫이 그 함수의 것이 아니게 된다."""
    pts, w = p5n.quadrature()
    rng = np.random.default_rng(1)
    P = torch.as_tensor(rng.normal(size=(1, 12, len(w))))
    S = torch.as_tensor(rng.normal(size=(12, len(w))))
    out = p5n.project(S, P, w, "exact")
    c = (S[:3] - out[:3]).reshape(-1) @ P[0, :3].reshape(-1) / (
        P[0, :3].reshape(-1) @ P[0, :3].reshape(-1))
    assert torch.allclose(S[3:] - out[3:], c * P[0, 3:], atol=1e-8)


def test_sequential_solve_returns_ordered_eigenvalues_and_diagnostics():
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=20, width=16, depth=2,
                              device="cpu")
    assert out["lam"].shape == (2, 2)
    assert len(out["span_diagnostics"]) == 2
    assert out["span_diagnostics"][0] is None          # 모드 1은 deflation 없음
    d = out["span_diagnostics"][1]
    assert d["k"] == 1 and "kappa_gram" in d and "n_seeds_summarized" in d
    assert out["arm"].startswith("p5_neural_3d[")


@pytest.mark.parametrize("kind", ["exact", "diagonal", "qr"])
def test_every_arm_runs_and_names_itself(kind):
    out = p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=10, width=16, depth=2,
                              projection=kind, device="cpu")
    assert kind in out["arm"] or p5n.ARM_NAME[kind] == out["arm"]
    assert np.isfinite(out["lam"]).all()


def test_solver_records_how_much_mass_survives_the_projection():
    """붕괴한 시드의 Λ는 λ₁보다도 작게 나온다. 그것이 "낮은 모드를 찾았다"인지
    "사영된 함수가 소실됐다"인지는 귀속이 다르고, 남은 질량비로만 갈린다.
    비율이 없으면 표를 보고 두 경우를 구별할 수 없다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=3, iters=20, width=16, depth=2,
                              device="cpu")
    pm = out["projected_mass_fraction"]
    assert pm.shape == (3, 2)
    # 모드 1은 사영이 없으므로 정의상 1
    assert np.allclose(pm[:, 0], 1.0)
    assert ((pm[:, 1] >= 0.0) & (pm[:, 1] <= 1.0 + 1e-12)).all(), pm[:, 1]


def test_solver_returns_the_final_shapes_the_scorer_needs():
    """4분류는 MAC을 읽는다. 최종 형상을 돌려주지 않으면 채점을 솔버 밖에서 할 수 없다.
    **사영된 상태의 변위**여야 한다 — 그것이 그 스테이지가 내놓은 함수다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=3, iters=20, width=16, depth=2,
                              device="cpu")
    sh = out["shapes"]
    assert len(sh) == 2 and sh[0].shape == (3, 3, out["n_q"])   # (seed, comp, n_q)
    assert np.isfinite(sh[0]).all()
    # 모드 2의 형상은 저장된 모드 1과 질량직교해야 한다(정확 사영이므로)
    import torch
    _, w = p5n.quadrature()
    A = torch.as_tensor(sh[1][0]); B = torch.as_tensor(out["stored"][0][0][:3])
    ip = abs(float((w * (A * B).sum(0)).sum()))
    na = float((w * (A ** 2).sum(0)).sum()) ** 0.5
    nb = float((w * (B ** 2).sum(0)).sum()) ** 0.5
    assert ip / (na * nb) < 1e-8, ip / (na * nb)


def test_solver_can_recheck_every_cell_on_a_second_quadrature_order():
    """§3.6 구적분리를 신경 arm의 **상시 계기**로 둔다.

    P1은 이 검사를 고전 기저에만 적용했다. 그러나 최적화는 인증 격자가 분해하지 못하는
    변형장을 찾아낼 수 있고, 그때 Λ는 그 함수의 Rayleigh 몫이 아니라 구적 인공물이다.
    사전 인증은 '참조 모드'를 덮을 뿐 '최적화가 도달하는 함수'를 덮지 못하므로,
    셀마다 두 구적차수로 재서 값이 움직이는지 봐야 한다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=3, iters=20, width=16, depth=2,
                              device="cpu", check_grid=(24, 12, 9))
    assert out["lam_check"].shape == out["lam"].shape
    assert out["check_grid"] == (24, 12, 9)
    rel = np.abs(out["lam_check"] - out["lam"]) / np.abs(out["lam"])
    assert out["quadrature_drift"].shape == out["lam"].shape
    assert np.allclose(out["quadrature_drift"], rel, rtol=0, atol=1e-12)
    # 검사 격자를 주지 않으면 재검을 하지 않는다(기본 비용을 늘리지 않는다)
    plain = p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=10, width=16, depth=2,
                                device="cpu")
    assert plain["lam_check"] is None and plain["quadrature_drift"] is None


def test_grid_driver_records_quadrature_drift_per_cell():
    """본실험 표가 구적 인공물을 걸러낼 수 있어야 한다. 셀마다 두 구적차수의
    상대차를 싣지 않으면, 최적화가 격자를 속인 셀과 진짜로 낮은 Λ를 찾은 셀을
    표에서 구별할 수 없다."""
    import csv
    import shutil
    import tempfile

    from eigen_benchmark.drivers import run_p5_neural as rp5

    with tempfile.TemporaryDirectory() as d:
        for f in ("p5_reference.csv", "p5_stage0.csv"):
            shutil.copy(f"docs/_generated/data/paper4/{f}", d)
        rp5.grid(outdir=d, quick=True, device="cpu")
        rows = list(csv.DictReader(open(f"{d}/p5_neural.csv")))
    assert rows
    for r in rows:
        assert r["check_grid"], "재검 격자가 기록되지 않았다"
        assert 0.0 <= float(r["quad_drift_median"]) < 1e6
        assert float(r["quad_drift_max"]) >= float(r["quad_drift_median"])


# ── 재표집 구적 (§5 후속: 고정 격자를 최적화가 이기는 문제의 근본 대응) ──────

def test_quadrature_family_gives_distinct_grids_that_all_integrate_correctly():
    """가족의 모든 격자가 매끄러운 제작해를 tol 안에서 적분해야 한다. 하나라도
    부정확하면 그 반복의 목적함수가 틀린 값을 최소화한다."""
    from eigen_benchmark.problems import p5_block as p5
    fam = p5n.quadrature_family()
    assert len(fam) >= 8
    seen = set()
    for pts, w in fam:
        seen.add(len(w))
        u, G = p5.manufactured_field(pts.numpy())
        a, b = p5._forms_from_field(u, G, w.numpy())
        ref_pts, ref_w = p5._tensor_grid((96, 40, 24))
        a0, b0 = p5._forms_from_field(*p5.manufactured_field(ref_pts), ref_w)
        assert abs(a / a0 - 1) < 1e-5, (len(w), a / a0 - 1)
        assert abs(b / b0 - 1) < 1e-5
    assert len(seen) > 1, "격자 크기가 모두 같다 — 가족이 아니다"


def test_resampling_uses_more_than_one_grid_and_records_which():
    """어떤 격자를 썼는지 남지 않으면 재현할 수 없다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=60, width=16, depth=2,
                              device="cpu", resample_every=10)
    assert out["resample_every"] == 10
    assert out["n_grids_used"] > 1, out["n_grids_used"]
    # 최종 Λ는 **인증 격자**에서 잰다 — 비교 가능해야 하므로
    assert out["grid"] == p5n.CERTIFIED_GRID


def test_resampling_off_by_default_keeps_the_fixed_certified_grid():
    out = p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=10, width=16, depth=2,
                              device="cpu")
    assert out["resample_every"] is None and out["n_grids_used"] == 1


def test_stored_spans_are_re_evaluated_on_the_grid_actually_in_use():
    """저장 스팬을 옛 격자의 값으로 들고 있으면, 사영이 현재 격자에서 직교를
    만들지 못한다 — deflation이 조용히 무력해진다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=40, width=16, depth=2,
                              device="cpu", resample_every=10)
    import torch
    _, w = p5n.quadrature()
    A = torch.as_tensor(out["shapes"][1][0])          # 모드2의 최종 형상
    B = torch.as_tensor(out["stored"][0][0][:3])      # 저장된 모드1
    ip = abs(float((w * (A * B).sum(0)).sum()))
    na = float((w * (A ** 2).sum(0)).sum()) ** 0.5
    nb = float((w * (B ** 2).sum(0)).sum()) ** 0.5
    assert ip / (na * nb) < 1e-8, ip / (na * nb)


def test_grid_driver_writes_results_after_each_arm():
    """20시간짜리 실행이 종료 시에만 쓰면 중간 실패로 전부 잃는다. 실제로 검증
    실행 하나를 그렇게 잃었다. arm이 끝날 때마다 저장해야 한다."""
    import csv
    import shutil
    import tempfile

    from eigen_benchmark.drivers import run_p5_neural as rp5

    with tempfile.TemporaryDirectory() as d:
        for f in ("p5_reference.csv", "p5_stage0.csv"):
            shutil.copy(f"docs/_generated/data/paper4/{f}", d)
        seen = []

        def spy(path, rows):
            if path.endswith("p5_neural.csv"):
                seen.append(len(rows))
            return rp5.manifest.write_csv.__wrapped__(path, rows) \
                if hasattr(rp5.manifest.write_csv, "__wrapped__") else None

        rp5.grid(outdir=d, quick=True, device="cpu", arms=("exact", "qr"))
        rows = list(csv.DictReader(open(f"{d}/p5_neural.csv")))
    # 두 arm이 모두 실렸는지 — 마지막 저장이 앞 arm을 덮지 않아야 한다
    arms = {r["arm"] for r in rows}
    assert len(arms) == 2, arms


# ── v2 §2.2 훈련 격자와 감사 격자의 분리 ─────────────────────────────────────

def test_audit_grid_is_where_every_reported_quantity_is_measured():
    """v1의 실패는 학습·게이트·채점이 **같은 격자를 본 것**이다. v2는 보고되는 모든
    양을 훈련이 본 적 없는 감사 격자에서 잰다. 훈련 격자의 Λ도 함께 남겨 둘을
    비교할 수 있어야 한다 — 남기지 않으면 분리가 실제로 일어났는지 검증 불가능하다."""
    out = p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=30, width=8, depth=2,
                              audit_grid=(12, 6, 5), grid=(8, 6, 4),
                              snapshot_every=10)
    assert out["audit_grid"] == (12, 6, 5)
    assert out["n_q_audit"] == 12 * 6 * 5
    assert out["n_q"] == 8 * 6 * 4                  # 훈련 격자는 따로 보고된다
    assert out["lam"].shape == (2, 1)
    assert np.asarray(out["lam_train"]).shape == (2, 1)
    # 이력은 감사 격자에서 — 게이트가 읽는 값이다
    assert out["hist_grid"] == "audit"
    # 형상도 감사 격자에서 나와야 MAC이 같은 자로 재진다
    assert out["shapes"][0].shape == (2, 3, 12 * 6 * 5)


def test_train_grid_resampling_does_not_move_the_audit_grid():
    """재표집은 훈련 격자만 흔든다. 감사 격자가 함께 흔들리면 게이트가 무의미해진다."""
    out = p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=40, width=8, depth=2,
                              audit_grid=(12, 6, 5), resample_every=10,
                              snapshot_every=10)
    assert out["audit_grid"] == (12, 6, 5)
    assert out["n_grids_used"] > 1, "재표집이 실제로 일어나지 않았다"
    assert out["shapes"][0].shape[-1] == 12 * 6 * 5


def test_audit_grid_defaults_off_so_v1_runs_reproduce():
    out = p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=20, width=8, depth=2,
                              grid=(8, 6, 4), snapshot_every=10)
    assert out.get("audit_grid") is None
    assert out["n_q"] == 8 * 6 * 4


# ── E1 오라클 적합 (사전규정 2026-09-10) ────────────────────────────────────

def test_energy_inner_product_polarizes_the_quadratic_form_rq_already_uses():
    """적합 손실은 변형률까지 맞춰야 한다 — Rayleigh 몫은 변형률이 지배하므로
    변위만 맞추면 이 실험이 묻는 양을 재지 못한다. 그러려면 `rq_from_state`가 쓰는
    이차형식의 **쌍선형 판**이 필요하고, 대각에서 원래 형식과 같아야 한다."""
    import torch
    from eigen_benchmark.neural import p5_neural as p5n
    torch.manual_seed(0)
    n_q = 7
    S = torch.randn(12, n_q, dtype=torch.float64)
    T = torch.randn(12, n_q, dtype=torch.float64)
    w = torch.rand(n_q, dtype=torch.float64) + 0.1
    nu, E, rho = 0.29, 1.0, 1.0

    # 대각: a(u,u)는 Rayleigh 몫의 분자다
    a_uu = p5n.energy_ip(S, S, w, nu, E)
    num = p5n.rq_from_state(S, w, nu, E, rho) * (w * (S[:3] ** 2).sum(dim=0)).sum()
    assert torch.isclose(a_uu, num, rtol=1e-12)
    # 대칭
    assert torch.isclose(p5n.energy_ip(S, T, w, nu, E),
                         p5n.energy_ip(T, S, w, nu, E), rtol=1e-12)
    # 쌍선형 — 한 인자를 두 배로 하면 값도 두 배
    assert torch.isclose(p5n.energy_ip(2 * S, T, w, nu, E),
                         2 * p5n.energy_ip(S, T, w, nu, E), rtol=1e-12)
    # 강체 상수변위는 변형이 없으므로 에너지가 0이다
    R = torch.zeros(12, n_q, dtype=torch.float64)
    R[:3] = 1.0
    assert abs(float(p5n.energy_ip(R, R, w, nu, E))) < 1e-24


# ── 2026-09-12 정전 대응: 모드 단위 중간저장의 계기 ──────────────────────────

def test_solver_announces_each_mode_as_it_finishes():
    """arm 단위 저장(v1 B5)은 21–48시간짜리 실행에 너무 성기다. 2026-09-12 정전으로
    E2 (b′) 24.2시간과 E3 53.3시간을 **산출물 0바이트로** 잃었다. 모드가 끝날 때마다
    바깥에 알려야 완료된 모드만이라도 건진다."""
    seen = []
    p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=20, width=16, depth=2,
                        device="cpu", on_mode_done=seen.append)
    assert [i["mode"] for i in seen] == [1, 2], "모드 순서대로 한 번씩 와야 한다"
    last = seen[-1]
    # 채점에 필요한 것이 전부 실려야 바깥에서 그 모드를 채점할 수 있다
    assert last["lam"].shape == (2,)
    assert last["lam_train"].shape == (2,)
    assert last["shapes"].shape[0] == 2
    assert last["hist"].shape[1] == 2
    assert last["proj_mass"].shape == (2,)
    assert last["seconds"] > 0
    assert last["arm"].startswith("p5_neural_3d[")


def test_mode_callback_cannot_move_the_numbers():
    """중간저장은 **곁가지**여야 한다. 콜백이 전역 난수를 소비해도 Λ가 비트 하나
    움직이면 커밋된 실행과 비교할 수 없다 — 가중치 초기화는 모드마다 다시 심고
    (`torch.manual_seed`), 재표집은 전용 Generator를 쓰므로 견뎌야 한다."""
    kw = dict(n_modes=2, n_seeds=2, iters=20, width=16, depth=2, device="cpu",
              resample_every=10)

    def greedy(info):
        torch.rand(8)                      # 전역 torch 난수를 소비한다
        np.random.rand(8)                  # 전역 numpy 난수도 소비한다

    base = p5n.solve_p5_neural(**kw)
    withcb = p5n.solve_p5_neural(**kw, on_mode_done=greedy)
    assert np.array_equal(base["lam"], withcb["lam"]), "Λ가 움직였다"
    assert np.array_equal(base["lam_train"], withcb["lam_train"])
    assert np.array_equal(base["projected_mass_fraction"],
                          withcb["projected_mass_fraction"])
    for a, b in zip(base["shapes"], withcb["shapes"]):
        assert np.array_equal(a, b), "형상이 움직였다"


def test_training_loop_can_report_progress_inside_a_mode(capsys):
    """모드 하나가 3.5시간이다. 모드 단위 로그만으로는 QR이 왜 2배 예산에서
    diagonal의 2.00배가 아니라 그 이상 걸렸는지 볼 수 없다."""
    p5n.solve_p5_neural(n_modes=1, n_seeds=2, iters=20, width=16, depth=2,
                        device="cpu", heartbeat_every=10)
    out = capsys.readouterr().out
    assert "it=10/20" in out, out
    assert "ms/iter" in out, out


def test_progress_reporting_stays_off_by_default():
    """커밋된 실행은 이 인자를 주지 않았다 — 기본값이 침묵이어야 로그가 갈리지 않는다."""
    import inspect
    sig = inspect.signature(p5n.solve_p5_neural)
    assert sig.parameters["heartbeat_every"].default is None
    assert sig.parameters["on_mode_done"].default is None


# ── 구적 사다리 (2026-09-16) ────────────────────────────────────────────────

def test_solver_measures_every_rung_of_a_quadrature_ladder():
    """한 격자 차이로는 수렴을 못 본다(E4). 사다리는 감사 격자를 0번 칸으로 두고
    그 위 차수들을 이어 재, 셀마다 수렴 연구를 남긴다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=20, width=16, depth=2,
                              device="cpu", audit_grid=(10, 6, 5),
                              check_grids=[(12, 7, 6), (14, 8, 7)])
    assert out["lam_ladder"].shape == (2, 2, 3)      # (시드, 모드, 사다리 칸)
    assert np.array_equal(out["lam_ladder"][:, :, 0], out["lam"]), \
        "0번 칸은 채점이 쓰는 감사 격자 Λ 그대로여야 한다"
    assert out["check_grids"] == ((10, 6, 5), (12, 7, 6), (14, 8, 7))


def test_ladder_keeps_the_single_grid_drift_instrument_intact():
    """기존 `check_grid` 계기는 그대로여야 커밋된 실행과 비교가 된다 —
    사다리의 첫 칸이 곧 그 계기다."""
    kw = dict(n_modes=1, n_seeds=2, iters=20, width=16, depth=2, device="cpu",
              audit_grid=(10, 6, 5))
    one = p5n.solve_p5_neural(**kw, check_grid=(12, 7, 6))
    lad = p5n.solve_p5_neural(**kw, check_grids=[(12, 7, 6), (14, 8, 7)])
    assert np.array_equal(one["lam_check"], lad["lam_check"])
    assert np.array_equal(one["quadrature_drift"], lad["quadrature_drift"])


def test_ladder_is_off_by_default_so_committed_runs_are_unchanged():
    import inspect
    sig = inspect.signature(p5n.solve_p5_neural)
    assert sig.parameters["check_grids"].default is None


# ── 검토 반영 (2026-09-20): Gram 성분과 망 파라미터 ──────────────────────────

def test_span_diagnostics_can_name_each_gram_offdiagonal():
    """기전 진술이 "성분 공유"로 승격되면서 **어느 쌍이 큰가**가 검정 대상이 됐다
    (예측: k=3에서 C₁₃만 크고 C₁₂·C₂₃은 작다). `max_offdiag`만으로는 못 본다.
    P1·P4와 공용 함수이므로 **옵트인**이어야 한다."""
    out = p5n.solve_p5_neural(n_modes=3, n_seeds=2, iters=20, width=16, depth=2,
                              device="cpu", gram_offdiag=True)
    d = out["span_diagnostics"][2]              # 모드 3 풀이가 받는 k=2 스팬
    # 부호는 저장 모드의 부호 규약에 따라 임의다 — 시드 중앙값을 내려면 절댓값이어야
    # 하므로 |C_ij|를 남긴다. 그래서 키 이름도 `absc_`다.
    assert {"absc_12"} <= set(d)
    assert 0.0 <= d["absc_12"] <= 1.0 + 1e-12
    pairs = [v for k, v in d.items() if k.startswith("absc_")]
    assert abs(max(pairs) - d["max_offdiag"]) < 1e-12


def test_gram_offdiagonal_is_off_by_default_so_committed_runs_are_unchanged():
    """가드다 — 오늘 기준 이미 통과하며 새 코드를 끌어내지 않는다. P1·P4와 공용
    함수이므로 기본이 꺼짐으로 남아야 그 문제들의 CSV 스키마가 안 바뀐다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=20, width=16, depth=2,
                              device="cpu")
    assert not any(k.startswith("absc_") for k in out["span_diagnostics"][1])


def test_solver_can_hand_back_the_trained_parameters():
    """망을 저장하지 않아 480셀 재측정에 77 GPU-시간이 들었다(E7). 파라미터를 남기면
    같은 일이 순전파 비용이 된다 — 재현이 비트 단위임은 E4·E6·E7이 보였다."""
    out = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=20, width=16, depth=2,
                              device="cpu", return_params=True)
    assert len(out["params"]) == 2                      # 모드마다 하나
    for pm in out["params"]:
        assert pm and all(v.shape[0] == 2 for v in pm.values())   # 시드 축이 앞
    base = p5n.solve_p5_neural(n_modes=2, n_seeds=2, iters=20, width=16, depth=2,
                               device="cpu")
    assert base.get("params") is None, "기본은 꺼짐이어야 커밋된 실행이 불변이다"
    assert np.array_equal(base["lam"], out["lam"]), "파라미터 반환이 수치를 바꾸면 안 된다"


# ── E8 준비 (검토 §4.3): 입력 정규화 ────────────────────────────────────────

def test_input_normalisation_is_opt_in():
    """커밋된 480셀은 원좌표로 돌았다 — 기본이 꺼짐이어야 그 실행이 불변이다."""
    import inspect
    sig = inspect.signature(p5n.solve_p5_neural)
    assert sig.parameters["normalize_input"].default is False


def test_normalised_input_changes_the_trial_space_but_keeps_the_clamped_face_exact():
    """검토 §4.3 — `_VecMLP3D`가 원좌표를 받아 축 스케일이 25배 다르다(z ∈ (0, 0.04)).
    그것이 3차원 난이도의 원인인지 물으려면 망 입력만 바꿔야 한다. **하드 BC는 물리
    x가 곱해지는 구조라 그대로 유지**되어야 하고, 그렇지 않으면 이 실험은 경계조건
    변경과 뒤섞인다."""
    kw = dict(n_modes=1, n_seeds=2, iters=20, width=16, depth=2, device="cpu")
    base = p5n.solve_p5_neural(**kw)
    norm = p5n.solve_p5_neural(**kw, normalize_input=True)
    assert not np.array_equal(base["lam"], norm["lam"]), \
        "정규화가 아무것도 바꾸지 않으면 실험이 성립하지 않는다"

    g = p5.P5_GEOMETRY
    sc = torch.tensor([g["L"], g["b"], g["h"]], dtype=torch.float64)
    ens = p5n.EnsembleVecMLP3D(2, width=16, depth=2, seed=0, device="cpu")
    face = torch.zeros(5, 3, dtype=torch.float64)
    face[:, 1] = torch.linspace(0, 0.2, 5)
    S = p5n.state_batch(ens, face, scale=sc)
    assert float(S[:, :3].abs().max()) < 1e-300, "고정면에서 변위가 정확히 0이어야 한다"


# ── 검토 반영 (2026-09-23): 기저불변성 명제의 수치 검증 ──────────────────────

def test_exact_projection_is_invariant_to_the_stored_basis_representation():
    """명제 — 정확 사영은 저장 스팬의 **기저 표현**에 불변이다. Φ → ΦR(R 비특이)에
    대해 P_ΦR = P_Φ. 반면 대각근사는 그렇지 않다. 이것이 대각 규칙의 실패가
    '최적화 사고'가 아니라 **구조적으로 기저 의존적**임을 말한다."""
    torch.manual_seed(0)
    n_q, k = 40, 3
    w = torch.rand(n_q, dtype=torch.float64) + 0.5
    P = torch.randn(k, 12, n_q, dtype=torch.float64)          # 저장 스팬(12성분 상태)
    v = torch.randn(12, n_q, dtype=torch.float64)             # 사영당할 상태
    R = torch.randn(k, k, dtype=torch.float64)                # 비특이 기저변환
    assert abs(float(torch.linalg.det(R))) > 1e-6
    PR = torch.einsum("ij,jcn->icn", R.T, P)                  # ΦR

    ex = p5n.PROJECTIONS["exact"]
    a, b = ex(v, P, w), ex(v, PR, w)
    assert torch.allclose(a, b, atol=1e-8), \
        f"정확 사영이 기저 표현에 의존한다: 최대차 {float((a - b).abs().max()):.2e}"

    dg = p5n.PROJECTIONS["diagonal"]
    c, d = dg(v, P, w), dg(v, PR, w)
    assert not torch.allclose(c, d, atol=1e-6), \
        "대각근사가 기저 표현에 불변으로 나왔다 — 이 대비가 명제의 요지다"


def test_exact_projection_annihilates_the_stored_span_and_diagonal_does_not():
    """(I − P_Φ)Φ = 0은 정확 사영이 스팬을 실제로 제거함을 뜻한다. 대각근사는
    (I − A_D)Φ = Φ(I − D⁻¹G)이므로 비대각이 0이 아닌 한 잔차가 남는다."""
    torch.manual_seed(1)
    n_q, k = 40, 3
    w = torch.rand(n_q, dtype=torch.float64) + 0.5
    P = torch.randn(k, 12, n_q, dtype=torch.float64)
    ex, dg = p5n.PROJECTIONS["exact"], p5n.PROJECTIONS["diagonal"]
    for j in range(k):
        assert float(ex(P[j], P, w).abs().max()) < 1e-8, "정확 사영이 저장 모드를 남겼다"
    resid = max(float(dg(P[j], P, w).abs().max()) for j in range(k))
    assert resid > 1e-3, f"대각근사 잔차가 0에 가깝다({resid:.2e}) — 비대각이 작은 경우다"


# ── E11 (사전규정 2026-09-24): 사용 전 재직교화 arm ──────────────────────────

def test_reorthogonalized_diagonal_equals_the_exact_projection():
    """명제 1의 예측 — 저장 집합을 사용 전에 W-재직교화하면 D⁻¹G = I가 되어 대각근사가
    정확 사영과 **일치**한다. 이 arm이 벽을 없애는지가 E11의 질문이고, 그 대수적 전제를
    여기서 고정한다."""
    torch.manual_seed(3)
    n_q, k = 48, 3
    w = torch.rand(n_q, dtype=torch.float64) + 0.5
    P = torch.randn(k, 12, n_q, dtype=torch.float64)
    v = torch.randn(12, n_q, dtype=torch.float64)
    ro = p5n.PROJECTIONS["reorth_diagonal"](v, P, w)
    ex = p5n.PROJECTIONS["exact"](v, P, w)
    assert torch.allclose(ro, ex, atol=1e-8), \
        f"재직교화 대각이 정확 사영과 다르다: 최대차 {float((ro - ex).abs().max()):.2e}"
    # 그리고 그냥 대각과는 달라야 한다 — 그래야 이 arm이 새 정보를 준다
    dg = p5n.PROJECTIONS["diagonal"](v, P, w)
    assert not torch.allclose(ro, dg, atol=1e-6)


def test_reorthogonalized_diagonal_annihilates_the_stored_span():
    torch.manual_seed(4)
    n_q, k = 48, 3
    w = torch.rand(n_q, dtype=torch.float64) + 0.5
    P = torch.randn(k, 12, n_q, dtype=torch.float64)
    for j in range(k):
        r = float(p5n.PROJECTIONS["reorth_diagonal"](P[j], P, w).abs().max())
        assert r < 1e-8, f"저장 모드 {j}가 남았다: {r:.2e}"


def test_reorthogonalized_arm_has_its_own_name():
    assert "reorth" in p5n.ARM_NAME["reorth_diagonal"]


# ── E13-b (사전규정 2026-09-24): 양면 고정 ──────────────────────────────────

def test_two_face_clamped_trial_function_vanishes_on_both_faces():
    """E13-b는 x = 0과 x = L 두 면을 고정한다. 하드 BC를 유지하려면 거리함수가
    x(L − x)여야 하고, 두 면에서 정확히 0이어야 한다 — 벌점이 아니다."""
    g = dict(p5.P5_GEOMETRY)
    ens = p5n.EnsembleVecMLP3D(2, width=16, depth=2, seed=0, device="cpu")
    for x in (0.0, g["L"]):
        face = torch.zeros(5, 3, dtype=torch.float64)
        face[:, 0] = x
        face[:, 1] = torch.linspace(0, g["b"], 5)
        S = p5n.state_batch(ens, face, bc="two_face")
        assert float(S[:, :3].abs().max()) < 1e-300, f"x = {x} 면에서 0이 아니다"
    # 내부에서는 0이 아니어야 한다
    inner = torch.full((3, 3), 0.5, dtype=torch.float64)
    assert float(p5n.state_batch(ens, inner, bc="two_face")[:, :3].abs().max()) > 0


def test_boundary_condition_defaults_to_the_committed_one_face_clamp():
    import inspect
    assert inspect.signature(p5n.solve_p5_neural).parameters["bc"].default == "clamped_x0"


# ── E14 (2026-09-27): 시드별 스팬 진단 ──────────────────────────────────────

def test_span_diagnostics_can_return_every_seed_not_only_the_median():
    """전이 주장을 구조 수준 3점이 아니라 시드 수준 수십 점에서 세우려면 시드별
    비대각이 필요하다(검토 §3). 지금까지는 시드 중앙값만 저장했는데, 요약을 만드는
    코드가 이미 시드별 값을 계산하고 **버리고** 있었다."""
    torch.manual_seed(7)
    n_seeds, k, n_q = 5, 3, 40
    P = torch.randn(n_seeds, k, 12, n_q, dtype=torch.float64)
    w = torch.rand(n_q, dtype=torch.float64) + 0.5
    out = p5n.span_diagnostics_3d(P, w, with_offdiag=True, per_seed=True)
    assert "per_seed" in out and len(out["per_seed"]) == n_seeds
    for i, d in enumerate(out["per_seed"]):
        assert d["seed"] == i and "max_offdiag" in d and "absc_12" in d
    import numpy as _np
    med = float(_np.median([d["max_offdiag"] for d in out["per_seed"]]))
    assert abs(med - out["max_offdiag"]) < 1e-12, "요약이 시드별 값의 중앙값이어야 한다"


def test_per_seed_is_off_by_default_so_committed_schemas_are_unchanged():
    torch.manual_seed(8)
    P = torch.randn(4, 2, 12, 30, dtype=torch.float64)
    w = torch.rand(30, dtype=torch.float64) + 0.5
    assert "per_seed" not in p5n.span_diagnostics_3d(P, w)


# ── E15 — 저장 규약 × 몬테카를로 구적 ───────────────────────────────────────────
#
# Rowan 외(IJNME 2025) 식 (8)은 대각형 뺄셈이다. 그 방법이 정확한 것은 **직교화된**
# 고유함수를 저장하기 때문이다(고정 적분 격자에서 저장 집합이 정확히 직교). 몬테카를로
# 구적에서는 새 표본마다 그 직교성이 표본 오차만큼 깨진다. E15가 그것을 잰다.

import numpy as np          # noqa: E402

TINY = dict(n_seeds=2, iters=6, width=8, depth=2, device="cpu", audit_grid=(6, 4, 3),
            snapshot_every=2)


def _mass_gram(stored, s, w):
    """저장 상태 목록의 시드 s 질량 그람(변위 3성분)."""
    U = [np.asarray(P)[s, :3] for P in stored]
    return np.array([[float((w * (a * b).sum(0)).sum()) for b in U] for a in U])


def test_mc_quadrature_is_uniform_with_equal_weights_summing_to_the_volume():
    g = p5.P5_GEOMETRY
    rng = np.random.default_rng(7)
    pts, w = p5n.mc_quadrature(500, rng, "cpu", g)
    assert pts.shape == (500, 3) and w.shape == (500,)
    vol = g["L"] * g["b"] * g["h"]
    assert abs(float(w.sum()) - vol) < 1e-15 and float(w.max() - w.min()) == 0.0
    hi = torch.tensor([g["L"], g["b"], g["h"]], dtype=pts.dtype)
    assert bool((pts >= 0).all()) and bool((pts <= hi).all())
    pts2, _ = p5n.mc_quadrature(500, np.random.default_rng(7), "cpu", g)
    assert torch.equal(pts, pts2)


def test_orthogonalized_storage_is_mass_orthogonal_on_the_audit_grid():
    out = p5n.solve_p5_neural(n_modes=3, projection="diagonal", storage="orthogonalized",
                              **TINY)
    w = out["w"]
    for s in range(2):
        G = _mass_gram(out["stored"], s, w)
        C = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
        assert np.abs(C - np.eye(3)).max() < 1e-10


def test_diagonal_rule_on_an_orthogonal_stored_set_equals_the_exact_projection():
    """저장 집합이 쓰이는 구적에서 직교하면 대각형 뺄셈(식 (8))은 정확 사영이다."""
    torch.manual_seed(0)
    pts, w = p5n.quadrature((6, 4, 3), "cpu")
    P = torch.randn(3, 12, len(w), dtype=torch.float64)
    Q = []                                        # 질량 내적으로 직교화(변위 성분 기준)
    for a in range(3):
        v = P[a].clone()
        for q in Q:
            v = v - p5n._mass_ip(v, q, w) / p5n._mass_ip(q, q, w) * q
        Q.append(v)
    Q = torch.stack(Q, 0)
    S = torch.randn(12, len(w), dtype=torch.float64)
    assert torch.allclose(p5n._project_diagonal(S, Q, w), p5n._project_exact(S, Q, w),
                          atol=1e-10, rtol=0)


def test_default_storage_and_quadrature_leave_the_solver_path_unchanged():
    a = p5n.solve_p5_neural(n_modes=2, projection="diagonal", **TINY)
    b = p5n.solve_p5_neural(n_modes=2, projection="diagonal", storage="raw",
                            mc_points=None, **TINY)
    assert np.array_equal(a["lam"], b["lam"])
    for x, y in zip(a["shapes"], b["shapes"]):
        assert np.array_equal(x, y)


def test_mc_training_is_reproducible_and_records_batch_offdiagonals():
    kw = dict(n_modes=3, projection="diagonal", storage="orthogonalized", mc_points=40,
              resample_every=1, batch_diag=True, **TINY)
    a = p5n.solve_p5_neural(**kw)
    b = p5n.solve_p5_neural(**kw)
    assert np.array_equal(a["lam"], b["lam"])
    off = a["batch_offdiag"]
    assert off.shape == (2, 3)
    assert np.isnan(off[:, :2]).all()             # 저장 모드가 둘 이상일 때만 정의된다
    assert np.isfinite(off[:, 2]).all() and (off[:, 2] > 0).all()


def test_neural_quadrature_on_the_l_bracket_is_the_numpy_composite_grid():
    """신경 arm의 구적과 FEM 쪽 인증·채점 격자가 같은 점이어야 한다(E21)."""
    from eigen_benchmark.problems import p5_block as p5
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    pts, w = p5n.quadrature((16, 8, 6), "cpu", g)
    P, W = p5._tensor_grid((16, 8, 6), g)
    assert np.allclose(pts.numpy(), P, atol=0) and np.allclose(w.numpy(), W, atol=0)


def test_input_scale_is_the_bounding_box_of_each_structure():
    """E21g — 망이 보는 좌표를 축별로 외접 상자로 나눈다. 상자의 축척은 커밋된 E8·E19와 같다."""
    assert p5n.input_scale(p5.P5_GEOMETRY) == (1.0, 0.2, 0.04)
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    assert p5n.input_scale(g) == (1.0, 0.7, 0.1)


def test_input_normalisation_on_the_l_bracket_keeps_the_clamped_face_exact():
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    kw = dict(n_modes=1, n_seeds=2, iters=10, width=16, depth=2, device="cpu", geometry=g,
              grid=(6, 3, 3), audit_grid=(6, 3, 3))
    base = p5n.solve_p5_neural(**kw)
    norm = p5n.solve_p5_neural(**kw, normalize_input=True)
    assert not np.array_equal(base["lam"], norm["lam"])
    sc = torch.tensor(p5n.input_scale(g), dtype=torch.float64)
    ens = p5n.EnsembleVecMLP3D(2, width=16, depth=2, seed=0, device="cpu")
    face = torch.zeros(5, 3, dtype=torch.float64)
    face[:, 1] = torch.linspace(0, 0.2, 5)
    S = p5n.state_batch(ens, face, scale=sc, geometry=g)
    assert float(S[:, :3].abs().max()) < 1e-300


def test_mc_quadrature_on_the_l_bracket_is_uniform_over_its_volume():
    """E21d — 블록을 부피에 비례해 고르고 그 안에서 균등하게 뽑으므로 L 전체에서 균등이고, 가중은 모두 V/n이다."""
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    n = 6000
    pts, w = p5n.mc_quadrature(n, np.random.default_rng(3), "cpu", g)
    V = (0.8 * 0.2 + 0.2 * 0.2 + 0.2 * 0.5) * 0.1
    assert pts.shape == (n, 3) and abs(float(w.sum()) - V) < 1e-15
    assert float(w.max() - w.min()) == 0.0
    P = pts.numpy()
    inside = ((P[:, 0] >= 0) & (P[:, 0] <= 1) & (P[:, 1] >= 0) & (P[:, 2] >= 0) & (P[:, 2] <= 0.1)
              & ((P[:, 1] <= 0.2) | ((P[:, 0] >= 0.8) & (P[:, 1] <= 0.7))))
    assert bool(inside.all())
    blk = np.where(P[:, 1] >= 0.2, 2, np.where(P[:, 0] >= 0.8, 1, 0))
    for b, vb in enumerate((0.016, 0.004, 0.010)):
        e = vb / V
        assert abs(float(np.mean(blk == b)) - e) < 4 * np.sqrt(e * (1 - e) / n)
    pts2, _ = p5n.mc_quadrature(n, np.random.default_rng(3), "cpu", g)
    assert torch.equal(pts, pts2)


def test_box_mc_draws_are_unchanged_by_the_l_extension():
    """커밋된 상자 MC 실행(E15·E17·E20)의 표본은 그대로여야 한다."""
    g = p5.P5_GEOMETRY
    pts, _ = p5n.mc_quadrature(10, np.random.default_rng(5), "cpu", g)
    ref = np.random.default_rng(5).uniform(size=(10, 3)) * np.array([g["L"], g["b"], g["h"]])
    assert np.array_equal(pts.numpy(), ref)


# ── E22 두 층 이종재료 ─────────────────────────────────────────────────────────
BI = {"kind": "bilayer", "L": 1.0, "b": 0.2, "h": 0.1, "E": 1.0, "E_top": 0.1,
      "rho": 1.0, "nu": 0.29}


def test_neural_quadrature_on_the_bilayer_is_the_numpy_split_grid():
    from eigen_benchmark.problems import p5_block as p5
    pts, w = p5n.quadrature((8, 4, 3), "cpu", BI)
    P, W = p5._tensor_grid((8, 4, 3), BI)
    assert np.array_equal(pts.numpy(), P) and np.array_equal(w.numpy(), W)


def test_neural_modulus_follows_the_layers_and_stays_a_float_off_them():
    import torch
    pts = torch.tensor([[0.5, 0.1, 0.01], [0.5, 0.1, 0.09]], dtype=torch.float64)
    assert p5n.young_at(pts, BI).tolist() == [1.0, 0.1]
    E = p5n.young_at(pts, dict(p5n.P5_GEOMETRY))
    assert isinstance(E, float) and E == p5n.P5_GEOMETRY["E"]


def test_neural_energy_is_the_certification_integral_on_the_bilayer():
    """신경 Rayleigh 몫의 분자·분모는 FEM 쪽 인증 형식과 같은 적분이다(점마다의 영률 포함)."""
    import torch
    from eigen_benchmark.problems import p5_block as p5
    P, W = p5._tensor_grid((12, 6, 4), BI)
    u, G = p5.manufactured_field(P, BI)
    a, b = p5._forms_from_field(u, G, W, BI, pts=P)
    S = torch.as_tensor(np.concatenate([u.T, G.transpose(1, 2, 0).reshape(9, -1)]))
    lam = p5n.rq_from_state(S, torch.as_tensor(W), BI["nu"],
                            p5n.young_at(torch.as_tensor(P), BI), BI["rho"])
    assert float(lam) == pytest.approx(a / b, rel=1e-12)


def test_solve_reports_the_audit_quotient_with_the_layer_moduli():
    """감사 Λ는 점마다의 영률로 잰 값이어야 한다 — 돌려받은 망으로 다시 재면 같다."""
    import torch
    from torch.func import vmap
    out = p5n.solve_p5_neural(n_modes=1, geometry=BI, return_params=True, **TINY)
    ens = p5n.EnsembleVecMLP3D(TINY["n_seeds"], TINY["width"], TINY["depth"], seed=0,
                               device="cpu")
    params = {k: torch.as_tensor(v) for k, v in out["params"][0].items()}
    pts, w = p5n.quadrature(TINY["audit_grid"], "cpu", BI)
    E = p5n.young_at(pts, BI)
    lam = vmap(lambda p: p5n.rq_from_state(p5n._state_one(p, ens.base, pts, None, None, BI),
                                           w, BI["nu"], E, BI["rho"]))(params)
    assert np.allclose(lam.numpy(), out["lam"][:, 0], rtol=1e-12, atol=0)


def test_monte_carlo_refuses_the_bilayer():
    with pytest.raises(NotImplementedError):
        p5n.mc_quadrature(100, np.random.default_rng(0), "cpu", BI)


# ── E21e (2026-10-09): 사다리를 점 묶음으로 ───────────────────────────────────

def test_ladder_can_be_measured_in_point_chunks_without_moving_the_values():
    """E21e의 사다리 최상단은 78만 점이라 한 번에 재면 GB10 메모리를 넘는다. 점마다의 상태는
    점끼리 독립이므로 묶음으로 나눠 잰 뒤 이어 붙여도 같은 사다리다. L-브래킷에서 잰다."""
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    kw = dict(n_modes=2, n_seeds=2, iters=10, width=16, depth=2, device="cpu", geometry=g,
              audit_grid=(6, 3, 3), check_grids=[(8, 4, 4), (10, 5, 5)])
    whole = p5n.solve_p5_neural(**kw)
    parts = p5n.solve_p5_neural(**kw, ladder_chunk=37)
    np.testing.assert_allclose(parts["lam_ladder"], whole["lam_ladder"], rtol=1e-12, atol=0)
    assert np.array_equal(parts["lam"], whole["lam"])


def test_ladder_chunking_is_off_by_default_so_committed_runs_are_unchanged():
    import inspect
    assert inspect.signature(p5n.solve_p5_neural).parameters["ladder_chunk"].default is None



def test_chunked_ladder_releases_the_allocator_cache_before_and_between_rungs(monkeypatch):
    """E21e 보완 — 90시간 학습의 할당자 캐시를 든 채 사다리에 들어가지 않도록, 묶음 평가를 켤 때만 사다리 직전과
    칸마다 캐시를 비운다. 기본 경로는 부르지 않는다."""
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    kw = dict(n_modes=1, n_seeds=2, iters=4, width=8, depth=1, device="cpu", audit_grid=(10, 6, 5),
              check_grids=[(12, 7, 6), (14, 8, 7)])
    p5n.solve_p5_neural(**kw)
    assert calls == []
    p5n.solve_p5_neural(**kw, ladder_chunk=50)
    assert len(calls) == 3                      # 사다리 직전 1 + 칸마다 1


# ── E21h: 저장된 망에서 단계를 시작한다(2026-10-14) ─────────────────────────────────

G_LB = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}


def _stored(out, n, width, depth, n_seeds=2):
    """순차 풀이가 돌려준 망 파라미터로 앞 n개 모드의 앙상블을 되살린다."""
    ens = []
    for m in range(n):
        e = p5n.EnsembleVecMLP3D(n_seeds, width, depth, seed=0, device="cpu")
        for k_, v in out["params"][m].items():
            e.params[k_] = torch.as_tensor(v)
        ens.append(e)
    return ens


@pytest.mark.parametrize("kw", [
    dict(mc_points=40, resample_every=1),
    dict(resample_every=3, family_axes=((4, 5), (2, 3), (2,))),
    dict(mc_points=40, resample_every=1, normalize_input=True),
])
def test_a_stage_started_from_stored_networks_is_the_sequential_stage(kw):
    """E21h — 앞 모드를 학습하지 않고 저장된 망으로 두면 그 단계들이 뽑던 표본만큼 난수를 넘긴다. 그러면 다음 단계의
    초기화·표본 순서·사영이 순차 풀이와 같다. 새 단계의 모든 출력과 저장 단계의 감사 격자 값이 같아야 한다."""
    base = dict(n_modes=3, n_seeds=2, iters=8, width=8, depth=2, device="cpu", geometry=G_LB,
                audit_grid=(6, 4, 3), snapshot_every=2, projection="exact", **kw)
    seq = p5n.solve_p5_neural(**base, return_params=True)
    st = p5n.solve_p5_neural(**base, stored=_stored(seq, 2, 8, 2), stored_iters=8)
    for key in ("lam", "lam_train", "projected_mass_fraction"):
        np.testing.assert_array_equal(st[key], seq[key])
    for a, b in zip(st["shapes"], seq["shapes"]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(st["hist"][2], seq["hist"][2])
    assert st["hist"][0].size == 0 and st["hist"][1].size == 0      # 저장 단계는 학습하지 않는다
    assert st["n_stored"] == 2 and seq["n_stored"] == 0


def test_stored_networks_need_their_iterations_matching_seeds_and_a_stage_left():
    one = [p5n.EnsembleVecMLP3D(2, 8, 2, seed=0, device="cpu")]
    kw = dict(n_seeds=2, iters=4, width=8, depth=2, device="cpu")
    with pytest.raises(ValueError):                 # 표본 순서를 넘기려면 그 단계의 반복 수가 필요하다
        p5n.solve_p5_neural(n_modes=2, stored=one, **kw)
    with pytest.raises(ValueError):                 # 시드 수가 다르면 짝이 깨진다
        p5n.solve_p5_neural(n_modes=2, stored=[p5n.EnsembleVecMLP3D(3, 8, 2, seed=0, device="cpu")],
                            stored_iters=4, **kw)
    with pytest.raises(ValueError):                 # 풀 단계가 남지 않는다
        p5n.solve_p5_neural(n_modes=1, stored=one, stored_iters=4, **kw)
    with pytest.raises(ValueError):                 # 직교화 저장의 계수는 저장 단계에서 다루지 않는다
        p5n.solve_p5_neural(n_modes=2, stored=one, stored_iters=4, storage="orthogonalized", **kw)


def test_the_new_stage_may_be_wider_than_the_stored_ones():
    """E21h 폭 조건 — 저장된 망은 제 폭(8) 그대로, 새 단계만 요청한 폭(16)이다. 중간 망은 새 단계에서만 나온다."""
    base = dict(n_modes=2, n_seeds=2, iters=4, depth=2, device="cpu", geometry=G_LB,
                audit_grid=(6, 4, 3), snapshot_every=2, mc_points=30, resample_every=1)
    seq = p5n.solve_p5_neural(**base, width=8, return_params=True)
    seen = []
    st = p5n.solve_p5_neural(**base, width=16, stored=_stored(seq, 1, 8, 2), stored_iters=4,
                             return_params=True, on_checkpoint=seen.append, checkpoint_every=2)
    np.testing.assert_array_equal(st["lam"][:, 0], seq["lam"][:, 0])
    assert st["width"] == 16
    assert st["params"][0]["net.0.weight"].shape == (2, 8, 3)
    assert st["params"][1]["net.0.weight"].shape == (2, 16, 3)
    assert [(c["mode"], c["it"]) for c in seen] == [(2, 2)]


def test_checkpoints_report_the_parameters_without_touching_the_path():
    """E21h 곁가지 — 콜백은 학습 중인 단계의 파라미터 사본만 받고 결과는 그대로다. 마지막 반복은 보내지 않는다."""
    kw = dict(n_modes=2, projection="exact", **TINY)
    a = p5n.solve_p5_neural(**kw)
    seen = []
    b = p5n.solve_p5_neural(**kw, on_checkpoint=seen.append, checkpoint_every=2)
    np.testing.assert_array_equal(a["lam"], b["lam"])
    for x, y in zip(a["shapes"], b["shapes"]):
        np.testing.assert_array_equal(x, y)
    assert [(c["mode"], c["it"]) for c in seen] == [(1, 2), (1, 4), (2, 2), (2, 4)]
    assert all(isinstance(v, np.ndarray) for c in seen for v in c["params"].values())
