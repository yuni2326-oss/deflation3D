"""P5 원고가 인용하는 통계 — 시드 군집 부트스트랩과 시드 수준 로지스틱.

원고 수치는 전부 커밋된 산출물에서 와야 한다. 이 시험은 두 가지를 지킨다.
(1) 모듈이 **이미 보고한 값**을 커밋 자료에서 그대로 재현한다(E11, E14).
(2) 사다리 값이 **없는** 셀을 해상으로 세지 않는다 — 2026-09-30 결과 문서에서
    E3(사다리 도입 전 실행)의 결측 60셀을 "미해상 0/60"으로 센 오류가 있었다.
"""
import numpy as np
import pytest

from eigen_benchmark.drivers import p5_stats as st

DATA = "docs/_generated/data/paper4"


def test_seed_fraction_is_per_seed_correct_share_over_the_wall_modes():
    recs = [{"seed": s, "mode": m, "outcome": "correct" if (s + m) % 2 else "spurious"}
            for s in range(3) for m in range(1, 7)]
    f = st.seed_fraction(recs)
    # 시드 0: 모드 5만 correct → 1/3, 시드 1: 모드 4·6 → 2/3, 시드 2: 모드 5 → 1/3
    assert np.allclose(f, [1 / 3, 2 / 3, 1 / 3])


def test_seed_fraction_refuses_a_seed_missing_a_mode():
    """군집 부트스트랩은 시드가 완결돼 있어야 한다 — 빠진 모드를 0으로 채우면 안 된다."""
    recs = [{"seed": 0, "mode": m, "outcome": "correct"} for m in (4, 5)]
    with pytest.raises(ValueError):
        st.seed_fraction(recs)


def test_ladder_counts_keep_missing_apart_from_resolved():
    recs = [{"mode": 4, "ladder_tail_spread": 5e-2},     # 미해상
            {"mode": 5, "ladder_tail_spread": 1e-6},     # 해상
            {"mode": 6, "ladder_tail_spread": None},     # 결측
            {"mode": 6}]                                 # 결측(필드 없음)
    c = st.ladder_counts(recs)
    assert c == {"unresolved": 1, "resolved": 1, "missing": 2}


def test_ladder_counts_report_e3_as_unmeasured_not_resolved():
    """E3는 사다리 도입 전에 돌았다. 60셀 전부 결측이어야 한다."""
    recs = [r for r in st.load(DATA, "e3_audit_resolution", arm="exact") if r["mode"] >= 4]
    c = st.ladder_counts(recs)
    assert c["missing"] == 60 and c["resolved"] == 0 and c["unresolved"] == 0


def test_paired_bootstrap_reproduces_the_committed_e11_contrasts():
    """2026-09-24 결과 문서: (a″)−(a′) = +0.400 [+0.267, +0.550],
    (a″)−(b) = +0.133 [−0.067, +0.317]."""
    reorth = st.seed_fraction(st.load(DATA, "e11_reorth_diag"))
    diag = st.seed_fraction(st.load(DATA, "v2", arm="diagonal", condition="resampled"))
    exact = st.seed_fraction(st.load(DATA, "v2", arm="exact", condition="resampled"))
    p, lo, hi = st.paired_bootstrap(reorth, diag)
    assert (round(p, 3), round(lo, 3), round(hi, 3)) == (0.400, 0.267, 0.550)
    p, lo, hi = st.paired_bootstrap(reorth, exact)
    assert (round(p, 3), round(lo, 3), round(hi, 3)) == (0.133, -0.067, 0.317)


def test_logistic_fit_recovers_coefficients_from_a_large_synthetic_sample():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 0, 20000)
    p = 1 / (1 + np.exp(-(-5.0 - 3.0 * x)))
    y = (rng.uniform(size=x.size) < p).astype(float)
    b, ok = st.logistic_fit(np.column_stack([np.ones_like(x), x]), y)
    assert ok
    assert abs(b[0] + 5.0) < 0.15 and abs(b[1] + 3.0) < 0.1


def test_weighted_fit_equals_the_fit_on_duplicated_rows():
    """부트스트랩은 재표집을 가중으로 넣는다 — 행을 복제한 적합과 같아야 한다."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 0, 60)
    y = (rng.uniform(size=60) < 1 / (1 + np.exp(2 + 2 * x))).astype(float)
    w = rng.integers(0, 3, 60).astype(float)
    X = np.column_stack([np.ones_like(x), x])
    bw, _ = st.logistic_fit(X, y, w)
    keep = np.repeat(np.arange(60), w.astype(int))
    bd, _ = st.logistic_fit(X[keep], y[keep])
    assert np.allclose(bw, bd, atol=1e-7)


def test_e14_registered_estimate_matches_the_reported_value():
    """2026-09-30 결과 문서: β₁ = −3.239 [−5.276, −2.588], 120점, correct 26."""
    obs = st.e14_observations(DATA)
    assert len(obs["y"]) == 120 and int(obs["y"].sum()) == 26
    r = st.e14_registered(obs)
    assert round(r["b1"], 3) == -3.239
    assert (round(r["lo"], 3), round(r["hi"], 3)) == (-5.276, -2.588)
    assert r["n_failed"] == 0


def test_ladder_threshold_matches_the_driver():
    pytest.importorskip("torch")
    from eigen_benchmark.drivers import run_p5_neural as run
    assert st.LADDER_RESOLVED == run.LADDER_RESOLVED


def test_e12_verdict_applies_the_prespecified_rules():
    """사전규정 2026-09-26 §3 — 규칙 1은 두 규약 모두, 규칙 2는 두 해상도 모두를 요구한다."""
    r = {"rule1_resampled_high_minus_low": (0.5, 0.3, 0.7),
         "rule1_fixed_high_minus_low": (0.6, 0.5, 0.8),
         "rule2_low_resampled_minus_fixed": (0.2, 0.08, 0.38),
         "rule2_high_resampled_minus_fixed": (0.1, -0.03, 0.23),   # 한 해상도만 0 제외
         "rule3_interaction": (-0.13, -0.32, 0.05)}
    v = st.e12_verdict(r)
    assert v == {"rule1": True, "rule2": False, "rule3": False, "indecomposable": False}


def test_fixed_effects_bootstrap_keeps_separated_resamples():
    """E13-b의 correct 시드는 둘(8, 13)뿐이라, 둘 다 안 뽑힌 재표본(약 (18/20)^20 = 12 %)에서
    그 구조의 더미가 발산한다. β₁은 나머지 두 구조가 식별하므로 유한해야 하고, 구간은
    그 재표본을 버리지 않고 낸다."""
    obs = st.e14_observations(DATA)
    r = st.e14_fixed_effects(obs)
    assert 0.10 < r["n_failed"] / r["B"] < 0.14
    assert r["n_nonfinite"] == 0
    assert r["hi"] < 0 and r["hi_converged"] < 0


def test_committed_stats_artifact_carries_the_figure_curve():
    """그림 렌더러는 계산하지 않는다 — 곡선과 띠가 산출물에 있어야 한다."""
    import json, os
    p = os.path.join(DATA, "p5_stats.json")
    if not os.path.exists(p):
        pytest.skip("p5_stats.json이 없다 — p5_stats.main을 먼저 돌린다")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)["e14_curve"]
    x, fit, lo, hi = (np.array(c[k]) for k in ("x", "fit", "lo", "hi"))
    assert len(x) == len(fit) == len(lo) == len(hi) > 50
    assert np.all(lo <= fit + 1e-12) and np.all(fit <= hi + 1e-12)
    assert np.all(np.diff(fit) < 0)          # β₁ < 0 이면 곡선은 단조 감소


def test_e13_contrasts_reproduce_the_committed_values():
    """2026-09-26 결과 문서: E13-a 모드 4–6 정확 − 대각 +0.500 [+0.350, +0.650],
    QR − 대각 +0.483 [+0.333, +0.633]; E13-b 모드 3–6(예측 벽부터) 정확 − 대각
    +0.487 [+0.312, +0.650], QR − 대각 +0.375 [+0.250, +0.500]."""
    r = st.e13(DATA)
    def t(x):
        return tuple(round(v, 3) for v in x)
    assert t(r["e13a_thick"]["exact_minus_diagonal"]) == (0.500, 0.350, 0.650)
    assert t(r["e13a_thick"]["qr_minus_diagonal"]) == (0.483, 0.333, 0.633)
    assert t(r["e13b_clamped2"]["exact_minus_diagonal"]) == (0.487, 0.312, 0.650)
    assert t(r["e13b_clamped2"]["qr_minus_diagonal"]) == (0.375, 0.250, 0.500)
    assert r["e13a_thick"]["modes"] == [4, 5, 6] and r["e13b_clamped2"]["modes"] == [3, 4, 5, 6]


def test_arm_key_does_not_confuse_diagonal_with_reorth_diagonal():
    recs = [{"arm": "p5_neural_3d[diagonal projection]"},
            {"arm": "p5_neural_3d[reorth diagonal projection]"}]
    assert [r for r in recs if st.arm_key("diagonal") in r["arm"]] == recs[:1]
    assert [r for r in recs if st.arm_key("reorth diagonal") in r["arm"]] == recs[1:]


def test_e14_sensitivity_drops_stages_already_contaminated_by_the_wall():
    """사후 민감도 분석. E13-b는 벽이 모드 3에서 시작해 k = 3 저장 스팬에 이미 붕괴한 모드 3
    출력이 들어 있다 — 사전규정이 k ≥ 4를 뺀 이유가 E13-b에서는 k = 3에 해당한다.
    k = 2만으로도 β₁이 0을 제외하고 음수여야 등록된 판독이 오염에 기대지 않는다."""
    obs = st.e14_observations(DATA)
    r = st.e14_sensitivity(obs)
    assert r["k2_only"]["n"] == 60 and r["k2_only"]["n_correct"] == 26
    assert r["k2_only"]["hi"] < 0
    assert round(r["k2_only"]["b1"], 3) == -2.756
    assert r["pre_wall"]["n"] == 100 and r["pre_wall"]["hi"] < 0


# ── E15 — 저장 규약 × 몬테카를로 구적 (사전규정 2026-09-30) ─────────────────────

def _fake(arm, outcomes):
    """outcomes[seed][mode-1] = (outcome, matched, mac, e_lam, converged)."""
    recs = []
    for s, row in enumerate(outcomes):
        for m, (oc, j, mac, e, conv) in enumerate(row, start=1):
            recs.append({"arm": arm, "seed": s, "mode": m, "outcome": oc,
                         "matched_mode": j, "mac": mac, "e_lam": e, "converged": conv,
                         "projected_mass_fraction": 0.5, "batch_offdiag": 0.03 if m >= 3 else float("nan")})
    return recs


def test_gate_free_accuracy_ignores_the_convergence_gate_only():
    ok = {"outcome": "non_converged", "matched_mode": 4, "mode": 4, "mac": 0.95,
          "e_lam": 0.01, "converged": False}
    assert st.accurate(ok)                                   # 게이트만 무시한다
    assert not st.accurate({**ok, "outcome": "invalid_quadrature"})
    assert not st.accurate({**ok, "e_lam": 0.06})
    assert not st.accurate({**ok, "matched_mode": 3})
    assert not st.accurate({**ok, "mac": 0.89})


def test_e15_contrast_is_exact_minus_rowan_on_gate_free_accuracy():
    good = [("correct", m, 0.99, 0.01, True) for m in range(1, 7)]
    bad = [("correct", m, 0.99, 0.01, True) for m in range(1, 4)] + \
          [("spurious", m, 0.99, 0.2, True) for m in range(4, 7)]
    b = _fake("p5_neural_3d[exact projection]", [good] * 20)
    r = _fake("p5_neural_3d[diagonal projection, orthogonalized storage]", [bad] * 20)
    res = st.e15_from_records(b, r)
    assert res["acc_4_6"] == {"exact_mc": 60, "rowan_mc": 0}
    assert res["contrast"][0] == 1.0 and res["verdict"] == "rule1"
    assert res["rowan_failure_classes"] == {"spurious": 60}
    res = st.e15_from_records(b, b)
    assert res["verdict"] == "rule2"


# ── E16 정사각 단면 (사전규정 2026-09-30-p5-e16-prespecification.md) ─────────

def _e16_recs(correct_at, n_seeds=20, modes=(1, 2, 3, 4, 5, 6), mu=None, fail="lower_mode_basin"):
    """correct_at(mode, seed) → bool. mu[seed]는 모드 1 출력의 쌍 혼합도(4 f_y f_z)."""
    out = []
    for s in range(n_seeds):
        for m in modes:
            ok = correct_at(m, s)
            fy = 0.5 * (1 - np.sqrt(1 - (mu[s] if mu is not None else 0.0)))
            out.append({"mode": m, "seed": s, "outcome": "correct" if ok else fail,
                        "component_fractions": [0.0, fy, 1.0 - fy]})
    return out


def _e16_spans(c12):
    return [{"k": "2", "stage_mode": "3", "seed": str(s), "max_offdiag": str(c)}
            for s, c in enumerate(c12)]


def _obs14(n_seeds=20, b0=-5.0, b1=-3.0, seed=7):
    """분리되지 않는 합성 E14 관측 — 시드마다 x 여섯 개, y는 로지스틱 난수."""
    rng = np.random.default_rng(seed)
    x = np.tile([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5], n_seeds)
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(b0 + b1 * x)))).astype(float)
    return {"x": x, "y": y, "seed": np.repeat(np.arange(n_seeds), 6),
            "k": np.full(x.size, 2), "structure": np.zeros(x.size, int)}


def test_poisson_binomial_reduces_to_the_binomial_for_equal_probabilities():
    from math import comb
    pmf = st.poisson_binomial_pmf(np.full((1, 6), 0.3))[0]
    want = [comb(6, k) * 0.3 ** k * 0.7 ** (6 - k) for k in range(7)]
    assert np.allclose(pmf, want) and pmf.sum() == pytest.approx(1.0)


def test_e16_primary_contrast_is_exact_minus_diagonal_at_mode_3():
    recs = {"exact": _e16_recs(lambda m, s: True),
            "diagonal": _e16_recs(lambda m, s: m != 3),
            "reorth_diagonal": _e16_recs(lambda m, s: True),
            "qr": _e16_recs(lambda m, s: True)}
    spans = {a: _e16_spans([0.5] * 20) for a in recs}
    obs = _obs14()
    r = st.e16_from_records(recs, spans, obs, B=2000)
    assert r["instrument_ok"] and r["primary"]["contrast"][0] == pytest.approx(1.0)
    assert r["primary"]["verdict"] == "rule1"
    assert r["mode3_failure_forms"]["diagonal"] == {"lower_mode_basin": 20}
    assert r["mu_identical_across_arms"] is True
    recs["qr"][0] = {**recs["qr"][0], "component_fractions": [0.0, 0.3, 0.7]}
    assert st.e16_from_records(recs, spans, obs, B=200)["mu_identical_across_arms"] is False
    # 선형 C의 중앙값이다 — 짝수 개면 10**median(log C)와 다르다
    assert r["mixing"]["baseline_k2_median_C"] == pytest.approx(np.median(10 ** obs["x"]))


def test_e16_is_not_read_when_the_exact_arm_cannot_find_the_second_pair_member():
    """모드 2는 저장이 하나라 모든 규칙이 정확 사영과 같다. 정확 arm이 거기서 실패하면
    쌍 채점이나 2단계 자체가 고장이므로 모드 3의 대비를 읽지 않는다."""
    bad = _e16_recs(lambda m, s: m != 2)
    recs = {a: bad for a in ("exact", "diagonal", "reorth_diagonal", "qr")}
    spans = {a: _e16_spans([0.5] * 20) for a in recs}
    obs = _obs14()
    r = st.e16_from_records(recs, spans, obs, B=2000)
    assert not r["instrument_ok"] and r["primary"]["verdict"] == "not_read"


def test_e14_transfer_flags_an_observed_count_outside_the_predictive_range():
    """C₁₂ ≈ 1이면 법칙은 모드 3 correct를 거의 0으로 예측한다. 20/20이면 범위 밖(위)이다."""
    obs = _obs14()
    got = st.e14_transfer(np.zeros(20), 20, obs, B=500)
    assert got["expected"] < 2 and got["verdict"] == "above"
    assert st.e14_transfer(np.zeros(20), 0, obs, B=500)["verdict"] == "consistent"


def test_mixing_regression_separates_a_sloped_from_a_flat_response():
    rng = np.random.default_rng(0)
    mu = np.linspace(0, 1, 20)
    sloped = -2.0 + 1.5 * mu + 0.05 * rng.normal(size=20)
    flat = -0.5 + 0.05 * rng.normal(size=20)
    assert st.mixing_regression(mu, sloped, B=2000)["verdict"] == "rule_a"
    assert st.mixing_regression(mu, flat, B=2000)["verdict"] == "rule_b"


# ── 후속 E17–E20 (사전규정 2026-10-02-p5-e17-e20-prespecification.md) ──────

def _acc_recs(acc_at, n_seeds=20, modes=(1, 2, 3, 4, 5, 6), outcome="correct"):
    """acc_at(mode, seed) → bool. 게이트 없는 정확(st.accurate)이 그 값이 되도록 만든다."""
    out = []
    for s in range(n_seeds):
        for m in modes:
            ok = acc_at(m, s)
            out.append({"mode": m, "seed": s, "outcome": outcome if ok else "spurious",
                        "matched_mode": m if ok else m + 1, "mac": 0.99, "e_lam": 0.01,
                        "cluster": [m]})
    return out


def test_unpaired_bootstrap_resamples_the_two_groups_independently():
    a, b = np.ones(20), np.zeros(20)
    pt, lo, hi = st.unpaired_bootstrap(a, b, B=500)
    assert pt == lo == hi == 1.0
    pt, lo, hi = st.unpaired_bootstrap(np.r_[np.ones(10), np.zeros(10)], np.zeros(20), B=2000)
    assert pt == pytest.approx(0.5) and 0.2 < lo < 0.5 < hi < 0.8


def test_e17_separates_placement_from_frequency():
    """MC 두 칸만 좋으면 배치 주효과(규칙 1), 매 반복 두 칸만 좋으면 빈도 주효과(규칙 2)다."""
    good, bad = _acc_recs(lambda m, s: True), _acc_recs(lambda m, s: m < 4)
    r = st.e17_from_cells({("gauss", 250): bad, ("gauss", 1): bad, ("mc", 250): good,
                           ("mc", 1): good}, B=500)
    assert r["verdict"] == {"rule1": True, "rule2": False, "rule3": False, "indecomposable": False}
    r = st.e17_from_cells({("gauss", 250): bad, ("gauss", 1): good, ("mc", 250): bad,
                           ("mc", 1): good}, B=500)
    assert r["verdict"]["rule2"] and not r["verdict"]["rule1"]


def test_e18_compares_the_diagonal_mode3_across_structures_without_pairing():
    e16 = _acc_recs(lambda m, s: m != 3)
    e18 = {"diagonal": _acc_recs(lambda m, s: True), "exact": _acc_recs(lambda m, s: True)}
    spans = [{"k": "2", "stage_mode": "3", "seed": str(s), "max_offdiag": "0.01"} for s in range(20)]
    obs = {"x": np.tile([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5], 20),
           "y": (np.random.default_rng(7).uniform(size=120) < 0.5).astype(float),
           "seed": np.repeat(np.arange(20), 6), "k": np.full(120, 2), "structure": np.zeros(120, int)}
    r = st.e18_from_records(e18, spans, e16, obs, B=500)
    assert r["primary"]["contrast"][0] == pytest.approx(1.0) and r["primary"]["verdict"] == "rule1"
    assert "e14_transfer" in r and r["correct_mode3"] == {"e18_diagonal": 20, "e16_diagonal": 0}


def test_e19_reads_modes_5_and_6_without_the_gate():
    e16x = _acc_recs(lambda m, s: m < 5)
    fixed = _acc_recs(lambda m, s: True, outcome="non_converged")   # 게이트에 걸려도 정확
    r = st.e19_from_records({"e19_budget": fixed, "e19_norm": e16x}, e16x, B=500)
    assert r["e19_budget"]["verdict"] == "rule1" and r["e19_budget"]["contrast"][0] == 1.0
    assert r["e19_norm"]["verdict"] == "rule2"


def test_e20_contrast_is_large_sample_minus_e15r():
    r = st.e20_from_records(_acc_recs(lambda m, s: True), _acc_recs(lambda m, s: m < 5), B=500)
    assert r["contrast"][0] == pytest.approx(2 / 3) and r["verdict"] == "rule1"


# ── E21 L-브래킷 (사전규정 2026-10-04-p5-e21-prespecification.md) ─────────────

def _e21_recs(correct_at, n_seeds=20, modes=(1, 2, 3, 4, 5, 6), fail="lower_mode_basin"):
    """correct_at(mode, seed) → bool. 게이트 없는 정확도 correct와 같아지도록 필드를 채운다."""
    out = []
    for s in range(n_seeds):
        for m in modes:
            ok = correct_at(m, s)
            out.append({"mode": m, "seed": s, "outcome": "correct" if ok else fail,
                        "matched_mode": m if ok else max(1, m - 1), "mac": 0.99,
                        "e_lam": 0.01, "cluster": [m]})
    return out


def _e21_spans(c2, c3):
    rows = [{"k": "2", "stage_mode": "3", "seed": str(s), "max_offdiag": str(c)}
            for s, c in enumerate(c2)]
    return rows + [{"k": "3", "stage_mode": "4", "seed": str(s), "max_offdiag": str(c)}
                   for s, c in enumerate(c3)]


def test_e21_primary_contrast_is_exact_minus_diagonal_over_modes_4_to_6():
    recs = {"exact": _e21_recs(lambda m, s: True),
            "diagonal": _e21_recs(lambda m, s: m < 4),
            "qr": _e21_recs(lambda m, s: m < 5)}
    spans = {a: _e21_spans([0.01] * 20, [0.95] * 20) for a in recs}
    r = st.e21_from_records(recs, spans, _obs14(), B=2000)
    assert r["instrument_ok"] and r["primary"]["verdict"] == "rule1"
    assert r["primary"]["contrast"][0] == pytest.approx(1.0) and r["primary"]["floor"] is False
    assert r["other_contrasts"]["qr_minus_diagonal"][0] == pytest.approx(1 / 3)
    assert r["other_contrasts"]["exact_minus_qr"][0] == pytest.approx(2 / 3)
    assert r["failure_forms_4_6"]["diagonal"] == {"lower_mode_basin": 60}
    assert r["correct_by_mode"]["qr"][4] == 20 and r["accurate_by_mode"]["qr"][5] == 0
    assert r["c_median_by_arm"]["exact"] == {2: pytest.approx(0.01), 3: pytest.approx(0.95)}


def test_e21_is_not_read_when_stage_2_fails_and_flags_a_floor():
    """모드 2는 저장이 하나라 세 규칙이 정확 사영과 같다 — 거기서 실패하면 읽지 않는다."""
    bad = _e21_recs(lambda m, s: m != 2)
    spans = {a: _e21_spans([0.01] * 20, [0.95] * 20) for a in ("exact", "diagonal", "qr")}
    r = st.e21_from_records({a: bad for a in spans}, spans, _obs14(), B=500)
    assert not r["instrument_ok"] and r["primary"]["verdict"] == "not_read"
    low = {"exact": _e21_recs(lambda m, s: m < 4 or (m == 4 and s < 5)),
           "diagonal": _e21_recs(lambda m, s: m < 4), "qr": _e21_recs(lambda m, s: m < 4)}
    assert st.e21_from_records(low, spans, _obs14(), B=500)["primary"]["floor"] is True


def test_e21_transfer_pools_stages_2_and_3_of_the_diagonal_arm():
    """S1은 E14 법칙의 관측 단위(시드 × k ∈ {2, 3})를 그대로 쓴다. 관측은 대각 arm의 모드 3·4
    correct 합이다."""
    recs = {"exact": _e21_recs(lambda m, s: True),
            "diagonal": _e21_recs(lambda m, s: m <= 3),
            "qr": _e21_recs(lambda m, s: True)}
    c2, c3 = [0.01 + 0.001 * s for s in range(20)], [0.95] * 20
    spans = {a: _e21_spans(c2, c3) for a in recs}
    obs = _obs14()
    r = st.e21_from_records(recs, spans, obs, B=500)
    assert r["e14_transfer"] == st.e14_transfer(np.log10(c2 + c3), 20, obs, B=500)
    assert r["e14_transfer"]["n"] == 40
    assert r["e14_transfer_by_stage"][2]["observed"] == 20
    assert r["e14_transfer_by_stage"][3]["observed"] == 0
    assert r["e14_transfer_k2_only"]["n"] == 40


def test_e21_refuses_a_span_row_whose_target_is_not_the_next_mode():
    recs = {a: _e21_recs(lambda m, s: True) for a in ("exact", "diagonal", "qr")}
    spans = {a: _e21_spans([0.01] * 20, [0.95] * 20) for a in recs}
    spans["diagonal"][25] = {**spans["diagonal"][25], "stage_mode": "5"}
    with pytest.raises(ValueError):
        st.e21_from_records(recs, spans, _obs14(), B=200)


# ── E22 두 층 이종재료 (사전규정 2026-10-04-p5-e22-prespecification.md) ───────

def test_e22_uses_the_same_rules_as_e21_and_compares_with_e13a_without_the_gate():
    recs = {"exact": _e21_recs(lambda m, s: True),
            "diagonal": _e21_recs(lambda m, s: m < 4),
            "qr": _e21_recs(lambda m, s: m < 5)}
    spans = {a: _e21_spans([0.01] * 20, [0.95] * 20) for a in recs}
    obs = _obs14()
    r22 = st.e22_from_records(recs, spans, obs, B=500)
    r21 = st.e21_from_records(recs, spans, obs, B=500)
    for k in ("primary", "other_contrasts", "e14_transfer", "correct_by_mode"):
        assert r22[k] == r21[k]
    e13 = {"exact": _e21_recs(lambda m, s: m < 5), "diagonal": _e21_recs(lambda m, s: m < 4),
           "qr": _e21_recs(lambda m, s: m < 4)}
    r = st.e22_from_records(recs, spans, obs, recs13a=e13, B=500)
    assert r["vs_e13a_gate_free"]["exact"][0] == pytest.approx(2 / 3)
    assert r["vs_e13a_gate_free"]["diagonal"][0] == pytest.approx(0.0)


def test_shape_right_cells_report_their_eigenvalue_error():
    """형상은 맞은 셀(정합 = 목표, MAC ≥ 0.9)의 e_λ — 모서리 특이(E21)·계면(E22)의 값을 본다."""
    recs = _e21_recs(lambda m, s: True)
    for r in recs:
        if r["mode"] == 4:
            r["e_lam"] = 0.021 + 0.01 * r["seed"]       # 0.021 … 0.211
    got = st.elam_shape_right(recs)
    assert got[4]["n"] == 20 and got[4]["median"] == pytest.approx(0.116)
    assert got[4]["over_threshold"] == 17                # e_λ > 0.05
    assert got[1]["median"] == pytest.approx(0.01) and got[1]["over_threshold"] == 0
    r21 = st.e21_from_records({a: recs for a in ("exact", "diagonal", "qr")},
                              {a: _e21_spans([0.01] * 20, [0.95] * 20) for a in ("exact", "diagonal", "qr")},
                              _obs14(), B=200)
    assert r21["elam_shape_right"]["exact"][4]["over_threshold"] == 17


# ── E21b L-브래킷 정확 arm 매 반복 재추첨 (사전규정 2026-10-06-p5-e21b-prespecification.md) ──

def _exploit_recs(n_seeds=20, below=(), ratio=1.0):
    """모드 4의 seeds in below에서 훈련 Λ가 참조 아래, 감사/훈련 비 ratio."""
    out = []
    for s in range(n_seeds):
        for m in (1, 2, 3, 4, 5, 6):
            bad = m == 4 and s in below
            out.append({"mode": m, "seed": s, "outcome": "spurious" if bad else "correct",
                        "matched_mode": m, "mac": 0.99, "e_lam": 0.9 if bad else 0.01,
                        "Lam_ref": 1.0, "Lam_train": 0.5 if bad else 1.0,
                        "Lam": 0.5 * ratio if bad else 1.0, "ladder_tail_spread": 0.1 if bad else 1e-6})
    return out


def test_training_exploitation_counts_cells_below_the_rayleigh_bound_on_the_training_grid():
    got = st.training_exploitation(_exploit_recs(below=range(5), ratio=40.0))
    assert got[4]["train_below_ref"] == 5 and got[1]["train_below_ref"] == 0
    assert got[4]["median_audit_over_train_failing"] == pytest.approx(40.0)
    assert got[1]["median_audit_over_train_failing"] is None


def test_e21b_contrast_is_redrawn_minus_e21_exact_on_gate_free_accuracy():
    g1 = _acc_recs(lambda m, s: True)
    ex = _acc_recs(lambda m, s: m < 4)
    for r_ in g1 + ex:
        r_.update(Lam=1.0, Lam_train=1.0, Lam_ref=1.0, ladder_tail_spread=1e-6)
    r = st.e21b_from_records(g1, ex, B=2000)
    assert r["contrast"][0] == pytest.approx(1.0) and r["verdict"] == "rule1"
    assert r["acc_4_6"] == {"e21lb_exact_g1": 60, "e21lb_exact": 0}
    assert set(r["exploitation"]) == {"e21lb_exact_g1", "e21lb_exact"}
    assert st.e21b_from_records(ex, ex, B=500)["verdict"] == "rule2"


# ── E21c L-브래킷 정확 arm 촘촘한 격자족 (사전규정 2026-10-08-p5-e21c-prespecification.md) ──

def test_e21c_uses_the_e21b_quantity_against_the_same_e21_exact_arm():
    dense = _acc_recs(lambda m, s: m < 5)
    ex = _acc_recs(lambda m, s: m < 4)
    for r_ in dense + ex:
        r_.update(Lam=1.0, Lam_train=1.0, Lam_ref=1.0, ladder_tail_spread=1e-6)
    r = st.e21c_from_records(dense, ex, B=2000)
    assert r["contrast"][0] == pytest.approx(1 / 3) and r["verdict"] == "rule1"
    assert r["acc_4_6"] == {"e21lb_exact_dense": 20, "e21lb_exact": 0}
    rb = st.e21b_from_records(dense, ex, B=2000)
    assert rb["contrast"] == r["contrast"] and rb["acc_4_6"] == {"e21lb_exact_g1": 20, "e21lb_exact": 0}


# ── E21e L-브래킷 정확 arm 고해상 격자족 (사전규정 2026-10-09-p5-e21e-prespecification.md) ──

def test_e21e_judges_against_e21_exact_and_reads_against_e21c():
    hires = _acc_recs(lambda m, s: True)
    ex = _acc_recs(lambda m, s: m < 4)
    dense = _acc_recs(lambda m, s: m < 5)
    for r_ in hires + ex + dense:
        r_.update(Lam=1.0, Lam_train=1.0, Lam_ref=1.0, ladder_tail_spread=1e-6)
    r = st.e21e_from_records(hires, ex, dense, B=2000)
    assert r["contrast"][0] == pytest.approx(1.0) and r["verdict"] == "rule1"
    assert r["acc_4_6"] == {"e21lb_exact_hires": 60, "e21lb_exact": 0}
    v = r["vs_e21c"]
    assert v["contrast"][0] == pytest.approx(2 / 3) and v["verdict"] == "rule1"
    assert v["acc_4_6"] == {"e21lb_exact_hires": 60, "e21lb_exact_dense": 20}
    assert st.e21e_from_records(ex, ex, ex, B=500)["vs_e21c"]["verdict"] == "rule2"



# ── E21d L-브래킷 정확 arm 매 반복 균등 몬테카를로 (사전규정 2026-10-09-p5-e21d-prespecification.md) ──

def test_e21d_judges_against_e21b_and_reads_against_e21_exact():
    mc = _acc_recs(lambda m, s: True)
    g1 = _acc_recs(lambda m, s: m < 4)
    ex = _acc_recs(lambda m, s: m < 5)
    for r_ in mc + g1 + ex:
        r_.update(Lam=1.0, Lam_train=1.0, Lam_ref=1.0, ladder_tail_spread=1e-6)
    r = st.e21d_from_records(mc, g1, ex, B=2000)
    assert r["contrast"][0] == pytest.approx(1.0) and r["verdict"] == "rule1"
    assert r["acc_4_6"] == {"e21lb_exact_mc1": 60, "e21lb_exact_g1": 0}
    v = r["vs_e21_exact"]
    assert v["contrast"][0] == pytest.approx(2 / 3) and v["verdict"] == "rule1"
    assert v["acc_4_6"] == {"e21lb_exact_mc1": 60, "e21lb_exact": 20}
    assert st.e21d_from_records(g1, g1, g1, B=500)["verdict"] == "rule2"


def test_vs_e21_exact_is_unchanged_by_the_general_helper():
    """E21b·E21c·E21e의 커밋된 판정이 그대로여야 한다 — 일반화한 비교 함수가 같은 값을 낸다."""
    a = _acc_recs(lambda m, s: s % 3 == 0)
    b = _acc_recs(lambda m, s: s % 5 == 0)
    for r_ in a + b:
        r_.update(Lam=1.0, Lam_train=1.0, Lam_ref=1.0, ladder_tail_spread=1e-6)
    x = st._vs_e21_exact(a, b, "new", 500, 7)
    y = st._vs_condition(a, b, "new", st.E21_TAGS["exact"], 500, 7)
    assert x == y


# ── E21f 지도 적합 판정 (사전규정 2026-10-10-p5-e21f-prespecification.md) ──

def test_e21f_rule_compares_the_deflated_fit_with_the_e21d_stage():
    rows8 = [{"e_lam_check_0": 0.02, "Lam_deflated": 1.6}] * 20
    good = [{"Lam_deflated": 1.30, "e_lam_check_0": 0.01}] * 20
    r = st.e21f_from_rows(good, rows8, [1.6] * 20)
    assert r["verdict"] == "rule1" and r["control_ok"] is True
    assert r["lam_deflated_median"] == pytest.approx(1.30) and r["lam_e21d_median"] == pytest.approx(1.6)
    bad = [{"Lam_deflated": 1.70, "e_lam_check_0": 0.30}] * 20
    assert st.e21f_from_rows(bad, rows8, [1.6] * 20)["verdict"] == "rule2"
    assert st.e21f_from_rows(good, [{"e_lam_check_0": 0.2, "Lam_deflated": 1.9}] * 20,
                             [1.6] * 20)["control_ok"] is False



# ── E21g 입력 정규화 (사전규정 2026-10-13-p5-e21g-prespecification.md) ──

def test_e21g_primary_is_mode_six_under_monte_carlo_and_secondary_the_gauss_arm():
    mc_norm = _acc_recs(lambda m, s: True)
    mc = _acc_recs(lambda m, s: m < 6)
    ga_norm = _acc_recs(lambda m, s: m < 5)
    ex = _acc_recs(lambda m, s: m < 4)
    for r_ in mc_norm + mc + ga_norm + ex:
        r_.update(Lam=1.0, Lam_train=1.0, Lam_ref=1.0, ladder_tail_spread=1e-6)
    r = st.e21g_from_records(mc_norm, mc, ga_norm, ex, B=2000)
    assert r["mode6"]["contrast"][0] == pytest.approx(1.0) and r["mode6"]["verdict"] == "rule1"
    assert r["mode6"]["acc_6"] == {"e21lb_exact_mc1_norm": 20, "e21lb_exact_mc1": 0}
    assert r["mc_4_6"]["acc_4_6"] == {"e21lb_exact_mc1_norm": 60, "e21lb_exact_mc1": 40}
    assert r["gauss_4_6"]["contrast"][0] == pytest.approx(1 / 3)
    assert r["gauss_4_6"]["acc_4_6"] == {"e21lb_exact_norm": 20, "e21lb_exact": 0}
    assert st.e21g_from_records(mc, mc, ex, ex, B=500)["mode6"]["verdict"] == "rule2"


def test_e21h_contrasts_mode_six_of_each_condition_with_the_run_it_took_its_modes_from():
    """사전규정 2026-10-14 E21h — 세 대비 모두 모드 6의 게이트 없는 정확, 시드 짝지음. 주 판정은 정규화·반복 4배
    대 E21g-mc다. 새 실행의 기록은 모드 6뿐이다."""
    six = lambda ok: _acc_recs(lambda m, s: s in ok, modes=(6,))
    base_norm = _acc_recs(lambda m, s: m < 6 or s == 14)      # E21g-mc — 모드 6은 시드 14만
    base_raw = _acc_recs(lambda m, s: m < 6)                  # E21d — 모드 6은 0
    r = st.e21h_from_records(six(set(range(12))), base_norm, six(set()), base_raw, six({14}), B=2000)
    assert r["primary"] == "sched_norm"
    sn, sr, wn = r["sched_norm"], r["sched_raw"], r["wide_norm"]
    assert sn["verdict"] == "rule1" and sn["contrast"][0] == pytest.approx(11 / 20)
    assert sn["acc_6"] == {"e21lb_s6_norm_x4": 12, "e21lb_exact_mc1_norm": 1}
    assert sr["verdict"] == "rule2" and sr["acc_6"] == {"e21lb_s6_raw_x4": 0, "e21lb_exact_mc1": 0}
    assert wn["verdict"] == "rule2" and wn["contrast"][0] == 0.0
    assert wn["acc_6"] == {"e21lb_s6_norm_w256": 1, "e21lb_exact_mc1_norm": 1}


def test_e21h_is_absent_until_its_records_exist(tmp_path):
    assert st.e21h(str(tmp_path)) is None


def test_e23_contrasts_mode_six_against_the_base_run_and_the_fourfold_schedule():
    """사전규정 2026-10-17 E23 — 주 판정은 lowtail 대 E21g-mc, 보조는 lowtail 대 E21h sched_norm과 fourier 대
    E21g-mc다. 모두 모드 6의 게이트 없는 정확, 시드 짝지음."""
    six = lambda ok: _acc_recs(lambda m, s: s in ok, modes=(6,))
    base_norm = _acc_recs(lambda m, s: m < 6 or s == 14)      # E21g-mc — 모드 6은 시드 14만
    r = st.e23_from_records(six(set(range(15))), base_norm, six(set(range(20))), six({14}), B=2000)
    assert r["primary"] == "lowtail"
    lo, ls, ff = r["lowtail"], r["lowtail_vs_sched"], r["fourier"]
    assert lo["verdict"] == "rule1" and lo["contrast"][0] == pytest.approx(14 / 20)
    assert lo["acc_6"] == {"e21lb_s6_norm_x4lo": 15, "e21lb_exact_mc1_norm": 1}
    assert ls["verdict"] == "rule3" and ls["acc_6"] == {"e21lb_s6_norm_x4lo": 15, "e21lb_s6_norm_x4": 20}
    assert ff["verdict"] == "rule2" and ff["contrast"][0] == 0.0


def test_e23_is_absent_until_its_records_exist(tmp_path):
    assert st.e23(str(tmp_path)) is None


def _cell(mode, seed, matched, mac=0.95, e=0.01, spread=None, outcome="correct"):
    return {"mode": mode, "seed": seed, "matched_mode": matched, "mac": mac, "e_lam": e, "outcome": outcome,
            "ladder_tail_spread": spread}


def test_e24_contrasts_each_layer_between_the_two_solvers_with_seeds_paired():
    """사전규정 2026-10-17 E24 — 대수 층: 모드 2–6에서 저장된 낮은 모드로 돌아간 셀, 계기 층: 모드 4–6 사다리 불안정, 최적화 층:
    모드 6 정확. 강형식은 시드 0–9의 모드 3–6이 모드 1로 돌아가고, 변분형은 모두 정확하다고 둔다."""
    def strong(m, s):
        if m >= 3 and s < 10:
            return _cell(m, s, 1, spread=0.5 if m >= 4 else None)
        return _cell(m, s, m, spread=1e-6)
    pin = [strong(m, s) for m in range(1, 7) for s in range(20)]
    var = [_cell(m, s, m, spread=0.5 if (m >= 4 and s < 4) else 1e-6) for m in range(1, 7) for s in range(20)]
    r = st.e24_from_records({"box": pin, "lbracket": pin}, {"box": var, "lbracket": var}, "box", B=2000)
    assert r["primary"] == "algebra@box" and r["judged"]["optimization"] == "lbracket"
    alg, ins, opt = r["box"]["algebra"], r["box"]["instrument"], r["box"]["optimization"]
    assert alg["verdict"] == "rule1" and alg["count"] == {"strong": 40, "variational": 0}
    assert alg["contrast"][0] == pytest.approx(40 / 100)
    assert ins["count"] == {"strong": 30, "variational": 12} and ins["verdict"] == "rule1"
    assert opt["count"] == {"strong": 10, "variational": 20} and opt["verdict"] == "rule3"
    assert r["box"]["accurate_by_mode"][1] == {"strong": 20, "variational": 20}
    assert not st.returned_to_stored(_cell(3, 0, 1, outcome="invalid_quadrature"))
    assert not st.returned_to_stored(_cell(3, 0, 1, mac=0.5)) and not st.ladder_unstable(_cell(4, 0, 4))


def test_e24_is_absent_until_its_records_exist(tmp_path):
    assert st.e24(str(tmp_path)) is None


def test_e25_combines_the_remedied_stage_and_judges_the_protocol_against_the_original_setting():
    """사전규정 2026-10-17 E25 — 프로토콜 arm(처방 반영) 대 원 설정 정확 arm, 모드 1–6 게이트 없는 정확. 처방은 단계 6을
    다시 푼 기록으로 그 단계만 바꾼다."""
    six = range(1, 7)
    orig_exact = [_cell(m, s, m if m <= 3 else m + 2) for m in six for s in range(20)]
    orig_diag = [_cell(m, s, m if m <= 3 else 1) for m in six for s in range(20)]
    protocol = [_cell(m, s, m if m < 6 else 8) for m in six for s in range(20)]
    remedy = (6, [_cell(6, s, 6) for s in range(20)], None)
    span = [{"seed": str(s), "k": str(k), "stage_mode": str(k + 1), "max_offdiag": str(0.01 if k == 2 else 0.9)}
            for s in range(20) for k in (2, 3)]
    obs14 = st.e14_observations(st.DATA_DIR)
    r = st.e25_from_records({"orig_diag": orig_diag, "orig_exact": orig_exact, "protocol": protocol}, span,
                            obs14, remedy, B=500)
    assert r["primary"]["verdict"] == "rule1" and r["primary"]["contrast"][0] == pytest.approx(0.5)
    assert r["accurate_by_mode"]["protocol"][6] == 20 and r["accurate_by_mode"]["protocol_first_pass"][6] == 0
    assert r["onset"] == {"predicted": 4, "observed": 4, "hit": True}
    assert r["remedy"]["stage"] == 6 and r["remedy"]["verdict"] == "rule1"
    assert r["detector"]["verdict"] == "rule2"            # 원 설정은 둘 다 모드 4–6에서 0
    with pytest.raises(ValueError):
        st.e25_combine(protocol, (5, [], None))


def test_e25_is_absent_until_its_records_exist(tmp_path):
    assert st.e25(str(tmp_path)) is None


def test_e26_contrasts_the_protocol_with_the_original_setting_per_structure():
    """사전규정 2026-10-18 E26 — 구조마다 모드 1–6 게이트 없는 정확, 프로토콜 대 원 설정, 시드 짝지음."""
    orig = [_cell(m, s, m if m <= 3 or s < 5 else 8) for m in range(1, 7) for s in range(20)]
    prot = [_cell(m, s, m) for m in range(1, 7) for s in range(20)]
    ff = [{**_cell(6, s, 6 if s < 12 else 8), "Lam": 1.30 if s < 12 else 1.55} for s in range(20)]
    d = [{**_cell(6, s, 8), "Lam": 1.55} for s in range(20)]
    fb = [{**_cell(6, s, 6), "Lam": 1.000 + 0.001 * s} for s in range(20)]
    e15b = [{**_cell(6, s, 6), "Lam": 1.010} for s in range(20)]
    r = st.e26_from_records({"thick": prot}, {"thick": orig}, ff, d, B=1000, ff_base=fb, e15b=e15b)
    assert r["thick"]["verdict"] == "rule1" and r["thick"]["contrast"][0] == pytest.approx(45 / 120)
    assert r["thick"]["accurate_by_mode"]["protocol"][6] == 20 and r["thick"]["accurate_by_mode"]["original"][6] == 5
    assert r["fourier_raw"]["acc_6"] == {"fourier_raw": 12, "e21d": 0} and r["fourier_raw"]["verdict"] == "rule1"
    # 재풀이 일관성: 12시드가 (1.55 − 1.30)/1.55 ≈ 16 % 내려가 허용오차 5 %를 넘고, 나머지는 그대로다.
    assert r["fourier_raw"]["resolve"]["n_above_tol"] == 12 and r["fourier_raw"]["resolve"]["n_lowered"] == 12
    # 음성 대조: 정답 단계의 재풀이는 Λ를 1 % 안에서만 바꾼다 — 검사가 울리지 않는다.
    rb = r["fourier_base"]["resolve"]
    assert rb["n_above_tol"] == 0 and rb["max"] == pytest.approx(0.01 / 1.01) and rb["n_lowered"] == 10
    assert r["fourier_base"]["verdict"] == "rule2"


def test_resolve_drop_refuses_mismatched_seeds():
    with pytest.raises(ValueError):
        st.resolve_drop([{"mode": 6, "seed": 0, "Lam": 1.0}], [{"mode": 6, "seed": 1, "Lam": 1.0}])


def test_lowest_lambda_pick_keeps_the_lower_solution_per_seed_without_reading_the_reference():
    """시드마다 감사 Λ가 가장 낮은 풀이를 고른다. 첫 풀이(모드 8, Λ 1.59)·재풀이 A(시드 0–9 정확, Λ 1.30)·재풀이 B(시드
    5–14 정확, Λ 1.31)면 고른 풀이는 시드 0–14가 정확이다 — 각 재풀이보다 많다."""
    first = [{**_cell(6, s, 8), "Lam": 1.59} for s in range(20)]
    a = [{**_cell(6, s, 6), "Lam": 1.30} if s < 10 else {**_cell(6, s, 8), "Lam": 1.60} for s in range(20)]
    b = [{**_cell(6, s, 6), "Lam": 1.31} if 5 <= s < 15 else {**_cell(6, s, 8), "Lam": 1.58} for s in range(20)]
    r = st.lowest_lambda_pick({"first": first, "a": a, "b": b})
    assert r["accurate"] == 15 and r["accurate_each"] == {"first": 0, "a": 10, "b": 10}
    assert r["picked"][0] == "a" and r["picked"][12] == "b" and r["picked"][17] == "b"
    with pytest.raises(ValueError):
        st.lowest_lambda_pick({"first": first, "short": a[:19]})


def test_e26_tbracket_contrasts_raw_with_the_original_and_the_normalized_protocol():
    """E26d — 주: 원좌표 프로토콜 대 원 설정 정확(모드 1–6), 보조: 굽힘 모드 1·3에서 원좌표 대 정규화 프로토콜. 정규화가
    모드 1의 고유값 오차만 키운다고 두면(형상은 맞음) 보조 대비가 양수다."""
    raw = [_cell(m, s, m, e=0.02) for m in range(1, 7) for s in range(20)]
    orig = [_cell(m, s, m if m <= 3 else 8) for m in range(1, 7) for s in range(20)]
    norm = [_cell(m, s, m, e=0.07 if m == 1 and s < 15 else 0.02) for m in range(1, 7) for s in range(20)]
    r = st.e26_tbracket(raw, orig, norm, B=1000)
    assert r["verdict"] == "rule1" and r["contrast"][0] == pytest.approx(0.5)
    c = r["coordinates"]
    assert c["verdict"] == "rule1" and c["contrast"][0] == pytest.approx(15 / 40)
    assert c["accurate_by_mode_norm"][1] == 5 and c["e_lam_median_norm_m1"] == pytest.approx(0.07)
    assert st.e26_tbracket(raw, orig, None, B=200).get("coordinates") is None


def test_e26e_crosses_the_resolve_flags_with_the_first_pass_classes():
    """E26e — 첫 풀이에서 더 높은 모드(시드 0–5, Λ 1.10)였던 시드를 재풀이가 모드로 되돌리면(Λ 0.50) 검사가 울리고, 정확했던
    시드(Λ 0.50 → 0.501)는 울리지 않는다. 울린 시드만 재풀이를 지키면 정확이 14 → 20이 된다."""
    first = [({**_cell(2, s, 4), "Lam": 1.10} if s < 6 else {**_cell(2, s, 2), "Lam": 0.50}) for s in range(20)]
    res = [{**_cell(2, s, 2), "Lam": 0.50 if s < 6 else 0.501} for s in range(20)]
    r = st.e26e_from_records(first, res, 2)
    assert r["by_first_class"]["higher"] == {"n": 6, "flagged": 6, "recovered": 6}
    assert r["by_first_class"]["accurate"] == {"n": 14, "flagged": 0, "recovered": 14}
    assert (r["accurate_first"], r["accurate_resolved"], r["accurate_after_test"]) == (14, 20, 20)


def test_e26e_rule_picks_every_stage_with_at_least_five_higher_mode_cells():
    recs = ([_cell(2, s, 4 if s < 6 else 2) for s in range(20)] + [_cell(3, s, 4 if s < 4 else 3) for s in range(20)]
            + [_cell(4, s, 5 if s < 5 else 2) for s in range(20)])
    assert st.e26e_rule_stages(recs) == [2, 4]


# ── E27 (사전규정 2026-10-23-p5-e27-prespecification.md) ─────────────────────────────

def _r27(mode, seed, lam, cls="accurate", spread=1e-6):
    """정확·더 높은 모드·그 밖 셀 하나."""
    matched = {"accurate": mode, "higher": mode + 1, "other": mode}[cls]
    return {"mode": mode, "seed": seed, "Lam": lam, "outcome": "correct" if cls == "accurate" else "spurious",
            "matched_mode": matched, "mac": 0.99 if cls != "other" else 0.3,
            "e_lam": 0.01 if cls == "accurate" else 0.5, "ladder_tail_spread": spread}


def test_e27_select_stage_prefers_stable_then_lowest_eigenvalue_then_order():
    """참조를 읽지 않는 선택 — 사다리가 안정한 후보 가운데 감사 Λ가 가장 낮은 것, 동률이면 후보 순서. 안정한 후보가 없으면
    전 후보 가운데 가장 낮은 것."""
    recs = {"first": [_r27(3, 0, 1.0), _r27(3, 1, 1.0), _r27(3, 2, 1.0)],
            "ff": [_r27(3, 0, 0.9, spread=2e-3), _r27(3, 1, 0.9), _r27(3, 2, 1.0)],
            "long": [_r27(3, 0, 0.95), _r27(3, 1, 0.95), _r27(3, 2, 1.0)]}
    assert st.e27_select_stage(recs, 3) == {0: "long", 1: "ff", 2: "first"}
    assert st.e27_select_stage({"first": [_r27(3, 0, 1.0, spread=1.0)], "ff": [_r27(3, 0, 0.5, spread=1.0)]}, 3) == {0: "ff"}
    with pytest.raises(ValueError):
        st.e27_select_stage({"first": [_r27(3, 0, 1.0)], "ff": [_r27(3, 1, 1.0)]}, 3)


def test_e27_counts_the_selected_solutions_and_reads_the_resolve_test_per_stage():
    """고른 풀이로 모드를 세고, 단계마다 첫 풀이 대비 Λ 감소가 허용오차를 넘은 시드(울림)를 첫 풀이의 부류와 교차한다:
    정확했는데 울리면 거짓 양성, 고른 풀이가 정확인데 첫 풀이가 더 높은 모드였고 울리지 않으면 놓침."""
    cands = {1: {"first": [_r27(1, 0, 1.0), _r27(1, 1, 1.0)]},
             2: {"first": [_r27(2, 0, 2.0), _r27(2, 1, 3.0, "higher")],
                 "ff": [_r27(2, 0, 2.01), _r27(2, 1, 2.0)]},
             3: {"first": [_r27(3, 0, 5.0, "higher"), _r27(3, 1, 4.0)],
                 "ff": [_r27(3, 0, 5.0, "higher"), _r27(3, 1, 4.02)],
                 "long": [_r27(3, 0, 4.0), _r27(3, 1, 4.01)],
                 "norm": [_r27(3, 0, 5.1, "higher"), _r27(3, 1, 4.05)]}}
    sel = {k: st.e27_select_stage(c, k) for k, c in cands.items()}
    assert sel[2] == {0: "first", 1: "ff"} and sel[3] == {0: "long", 1: "first"}
    orig = [_r27(1, 0, 1.0), _r27(2, 0, 2.0), _r27(3, 0, 9.0, "other"),
            _r27(1, 1, 1.0), _r27(2, 1, 9.0, "other"), _r27(3, 1, 9.0, "other")]
    first_pass = cands[1]["first"] + cands[2]["first"] + cands[3]["first"]
    aux = {3: {"reinit": [_r27(3, 0, 5.0, "higher"), _r27(3, 1, 4.0)]}}
    out = st.e27_from_records(cands, sel, orig, first_pass, aux, B=200)
    assert out["accurate_by_mode"] == {1: 2, 2: 2, 3: 2} and out["accurate_total"] == 6
    s2, s3 = out["stages"][2], out["stages"][3]
    assert (s2["flagged"], s2["false_positive"], s2["missed"]) == (1, 0, 0)
    assert s2["selected_each"] == {"first": 1, "ff": 1} and s2["repairs_of_higher"] == {"ff": 1}
    assert s3["repairs_of_higher"] == {"ff": 0, "long": 1, "norm": 0} and s3["selected_each"]["long"] == 1
    assert s3["first_class"] == {"accurate": 1, "higher": 1, "other": 0} and s3["unstable_selected"] == 0
    assert out["primary"]["reference_by_mode"] == {1: 2, 2: 1, 3: 0}
    assert out["vs_first_pass"]["reference_by_mode"] == {1: 2, 2: 1, 3: 1}
    assert out["aux"]["s3_reinit"] == {"accurate": 1, "repairs_of_higher": 0, "flagged_vs_first": 0}
    assert out["check_s3_first"] == {"n": 1, "same_class": 1, "max_rel_dlam": 0.0}
    assert s3["invalid_selected"] == 0
    assert out["primary"]["verdict"] in ("rule1", "rule2", "rule3")


def test_e28_classifies_first_solves_by_the_stage_mac_when_the_run_scored_only_modes_up_to_its_stage():
    """사전규정 §8 — 단계 k만 푼 실행의 기록은 참조 모드 1–k로만 짝지어, 더 높은 모드에 머문 첫 풀이가 "그 밖"으로 보인다.
    재채점의 분류(`higher`)를 주면 그 단계의 첫 풀이를 그것으로 나눠 고친 수와 놓침을 센다. 정확 판정은 기록 그대로다."""
    cands = {1: {"first": [_r27(1, 0, 1.0), _r27(1, 1, 1.0), _r27(1, 2, 1.0)]},
             2: {"first": [_r27(2, 0, 2.0), _r27(2, 1, 4.0, "other"), _r27(2, 2, 2.05, "other")],
                 "ff": [_r27(2, 0, 2.1), _r27(2, 1, 2.0), _r27(2, 2, 2.0)]}}
    sel = {k: st.e27_select_stage(c, k) for k, c in cands.items()}
    first_pass = cands[1]["first"] + cands[2]["first"]
    plain = st.e28_cc_from_records(cands, sel, first_pass, first_pass, B=200)["stages"][2]
    assert plain["first_class"] == {"accurate": 1, "higher": 0, "other": 2} and plain["repairs_of_higher"] == {"ff": 0}
    out = st.e28_cc_from_records(cands, sel, first_pass, first_pass, B=200, higher={2: {0: False, 1: True, 2: True}})
    s2 = out["stages"][2]
    assert s2["first_class"] == {"accurate": 1, "higher": 2, "other": 0} and s2["repairs_of_higher"] == {"ff": 2}
    assert (s2["flagged"], s2["false_positive"], s2["missed"]) == (1, 0, 1)      # 시드 2: 2.05 → 2.0은 울리지 않는다
    assert out["accurate_by_mode"] == {1: 3, 2: 3}                               # 정확 판정은 그대로


def test_e27_square_first_pass_contrast():
    sq = [_r27(m, s, 1.0) for m in range(1, 7) for s in range(3)]
    o16 = [_r27(m, s, 1.0, "accurate" if m < 5 else "other") for m in range(1, 7) for s in range(3)]
    out = st.e27_square_from_records(sq, o16, B=200)
    assert out["accurate_by_mode"]["first_pass"] == {m: 3 for m in range(1, 7)}
    assert out["accurate_by_mode"]["original"] == {1: 3, 2: 3, 3: 3, 4: 3, 5: 0, 6: 0}
    assert out["verdict"] == "rule1"


# ── E28 (사전규정 2026-10-23-p5-e28-prespecification.md) ─────────────────────────────

def test_e28_dispositions_cross_the_reference_free_status_with_accuracy():
    first = {0: _r27(3, 0, 5.0, "higher"), 1: _r27(3, 1, 4.0), 2: _r27(3, 2, 4.0)}
    chosen = {0: _r27(3, 0, 4.0), 1: _r27(3, 1, 4.0), 2: _r27(3, 2, 3.9, "other", spread=5e-3)}
    d = st.dispositions(chosen, first)
    assert (d["stable"], d["improved"], d["no_further_improvement"]) == (2, 1, 2)
    assert (d["reportable"], d["undetermined"], d["reportable_accurate"], d["undetermined_accurate"]) == (2, 1, 2, 0)


def test_e28_restart_pick_keeps_the_lowest_restart_without_reading_the_reference():
    first = [_r27(6, 0, 1.6, "higher"), _r27(6, 1, 1.6, "higher")]
    rs = {"reinit1": [_r27(6, 0, 1.59, "higher"), _r27(6, 1, 1.30)],
          "reinit2": [_r27(6, 0, 1.31), _r27(6, 1, 1.58, "higher")]}
    out = st.restart_pick(first, rs, 6)
    assert out["accurate_each"] == {"reinit1": 1, "reinit2": 1}
    assert out["accurate_lowest_restart"] == 2 and out["accurate_lowest_with_first"] == 2
    assert out["flagged_vs_first"] == 2 and len(out["lowest_restart_records"]) == 2


def test_e28_tbracket_control_counts_false_positives_on_accurate_first_solves():
    first = [_r27(k, s, 1.0 + k) for k in (1, 2) for s in range(3)]
    res = {1: [_r27(1, s, 2.0) for s in range(3)],
           2: [_r27(2, 0, 3.0), _r27(2, 1, 2.5, "higher"), _r27(2, 2, 3.01)]}
    out = st.e28_tb_from_records(first, res)
    assert out[2]["flagged"] == 1 and out[2]["false_positive"] == 1 and out[1]["false_positive"] == 0
    assert out["total"] == {"false_positive": 1, "accurate_first": 6, "accurate_selected": 5}


def test_e28_traction_weight_contrast_reads_the_eigenvalue_mismatch():
    def rec(m, s, lam, lp, acc=False):
        r = _r27(m, s, lam, "accurate" if acc else "other")
        r["lam_param"] = lp
        return r
    w1 = [rec(m, s, 1.0, 3.0) for m in range(1, 7) for s in range(3)]
    w10 = [rec(m, s, 1.0, 1.05, acc=(m == 1)) for m in range(1, 7) for s in range(3)]
    out = st.e28_pinn_from_records(w10, w1, B=200)
    assert out["mismatch"]["verdict"] == "rule3" and out["accuracy"]["w10"] == 3 and out["accuracy"]["w1"] == 0
    assert out["mismatch"]["median_rel"]["w10"] == pytest.approx(0.05)


# ── E29 (사전규정 2026-10-30-p5-e29-prespecification.md) — 한 단계 더 ─────────────────────────────────

E29_LAM = {1: 0.0035, 2: 0.0127, 3: 0.029, 4: 0.142, 5: 0.293, 6: 1.288, 7: 1.379}


def _r29(mode, seed, lam, matched=None, mac=0.99, spread=1e-6, ref_mode=None):
    """단계 `mode`의 기록 하나. 짝(기본 = 단계)과 그 짝의 참조로 e_λ·결과를 정한다(실행이 그렇게 채점한다)."""
    matched = mode if matched is None else matched
    lr = E29_LAM[mode if ref_mode is None else ref_mode]
    e = abs(lam - lr) / lr
    ok = matched == mode and mac >= 0.9 and e <= 0.05
    return {"mode": mode, "seed": seed, "Lam": lam, "Lam_ref": lr, "matched_mode": matched, "mac": mac, "e_lam": e,
            "outcome": "correct" if ok else "spurious", "ladder_tail_spread": spread}


def _e29_case():
    """시드 0: 단계 6이 모드 8, 단계 7이 건너뛴 모드 6. 시드 1: 단계 6 정확, 단계 7은 모드 7. 시드 2: 단계 7이 모드 7이지만
    사다리 불안정. 시드 3: 단계 5가 모드 6(더 높은 모드, 단계 6이 모드 8이라 울리지 않던 것), 단계 7이 모드 5를 찾는다."""
    first, s7 = [], []
    for s in range(4):
        for k in range(1, 5):
            first.append(_r29(k, s, E29_LAM[k] * 1.01))
    first += [_r29(5, 0, 0.296), _r29(6, 0, 1.596, matched=6, mac=0.0),
              _r29(5, 1, 0.296), _r29(6, 1, 1.30),
              _r29(5, 2, 0.296), _r29(6, 2, 1.596, matched=6, mac=0.0),
              _r29(5, 3, 1.30, matched=6), _r29(6, 3, 1.596, matched=6, mac=0.0)]
    s7 = [_r29(7, 0, 1.30, matched=6), _r29(7, 1, 1.40), _r29(7, 2, 1.39, spread=5e-3), _r29(7, 3, 0.293, matched=5)]
    id6 = {0: (8, 0.99), 2: (8, 0.99), 3: (8, 0.99)}
    id7 = {0: (6, 0.99), 1: (7, 0.99), 2: (7, 0.99), 3: (5, 0.99)}
    return first, s7, id7, id6


def test_e29_stage_seven_flags_the_last_stage_and_reorders_seven_outputs():
    """단계 7이 사다리를 통과하고 단계 6보다 5 % 넘게 낮으면 단계 6을 울린다. 단계 7이 건너뛴 모드를 찾으면 일곱 출력의
    정렬(순위 1–6)이 그 셀을 되찾는다. 앞 단계를 새로 울리는 경우(단계 6이 더 높아 그동안 울리지 않던 단계 5)도 센다."""
    first, s7, id7, id6 = _e29_case()
    out = st.e29_from_records(first, s7, id7, id6)
    assert (out["seeds"], out["stage7_stable"], out["flags_stage6"]) == (4, 3, 2)
    assert (out["stage6_accurate"], out["false_positives"], out["flags_on_inaccurate"]) == (1, 0, 2)
    assert out["new_earlier_flags"] == 1 and out["per_seed"][3]["new_earlier_flags"] == [5]
    assert out["stage7_modes"] == {"5": 1, "6": 1, "7": 2}
    assert [r["reordered6"] for r in out["per_seed"]] == [5, 6, 5, 4]
    assert [r["reordered7"] for r in out["per_seed"]] == [6, 6, 5, 6]
    assert (out["reordered_accurate_six_outputs"], out["reordered_accurate_seven_outputs"]) == (20, 23)
    assert out["flagged_min_drop"] == pytest.approx(1 - 1.30 / 1.596)
    assert out["accurate_max_drop"] == pytest.approx(1 - 1.40 / 1.30)
    plain = st.e29_from_records(first, s7)          # 재채점이 없으면 기록의 짝(참조 모드 1–k)을 쓴다
    assert plain["flags_stage6"] == 2 and plain["stage7_modes"] == {"5": 1, "6": 1, "7": 2}
    with pytest.raises(ValueError):
        st.e29_from_records(first, s7[:3])


def test_e29_reads_the_registered_verdicts_from_the_files(tmp_path):
    import json
    first, s7, id7, id6 = _e29_case()
    w = lambda name, recs: (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    w("p5_neural_e21lb_exact_mc1_records.jsonl", first)
    w("p5_neural_e29_lb_s7_records.jsonl", s7)
    assert st.e29(str(tmp_path)) is None                       # manifest가 없으면 끝나지 않은 실행
    (tmp_path / "manifest_p5_neural_e29_lb_s7.json").write_text("{}")
    out = st.e29(str(tmp_path))
    assert list(out) == ["lbracket"] and out["lbracket"]["flags_stage6"] == 2
    assert out["lbracket"]["verdict"] == {"prediction": {"flags_stage6_at_least": 14}, "hit": False}
    assert out["lbracket"]["ident_source"]["stage7"] == "records (stage MAC missing)"
    rows = "seed,mode_ref,mac_ref\n" + "".join(f"{s},{m},{a}\n" for s, (m, a) in id7.items())
    (tmp_path / "p5_exploit_e29_lb_s7_stage_mac.csv").write_text(rows, encoding="utf-8")
    (tmp_path / "manifest_p5_exploit_e29_lb_s7_stage_mac.json").write_text(json.dumps({"reproduces_record": False, "n_ref": 10}))
    assert st.e29(str(tmp_path))["lbracket"]["ident_source"]["stage7"] == "records (stage MAC did not reproduce the run)"
    (tmp_path / "manifest_p5_exploit_e29_lb_s7_stage_mac.json").write_text(json.dumps({"reproduces_record": True, "n_ref": 10}))
    assert st.e29(str(tmp_path))["lbracket"]["ident_source"]["stage7"] == "stage MAC against FEM modes 1-10"


# ── E30 (사전규정 2026-10-30-p5-e30-prespecification.md) — 단계적 배분 ─────────────────────────────────

def _r30(k, s, lam, matched=None, spread=1e-6):
    lr = float(k)
    matched = k if matched is None else matched
    e = abs(lam - lr) / lr
    ok = matched == k and e <= 0.05
    return {"mode": k, "seed": s, "Lam": lam, "Lam_ref": lr, "matched_mode": matched, "mac": 0.99, "e_lam": e,
            "outcome": "correct" if ok else "spurious", "ladder_tail_spread": spread}


def _e30_case():
    """시드 0·2는 모든 단계가 first로 정확. 시드 1은 단계 3이 두 경로 모두 모드 4에 서고 단계 4가 건너뛴 모드 3을 찾아
    단계 3이 울린다(j* = 3). 2회차에서 held-rate가 단계 3을 고치고 단계 4–6을 다시 푼다."""
    def stage(k, rnd):
        out = {}
        for p_ in (("first",) if k == 7 else ("first", "norm") + (("ff", "long") if rnd == 2 and k == 3 else ())):
            recs = []
            for s in range(3):
                f = {"first": 1.01, "norm": 1.02, "ff": 1.03, "long": 1.015}[p_]
                if s == 1 and rnd == 1 and k == 3:
                    recs.append(_r30(3, 1, 4.04 if p_ == "first" else 4.05, matched=4))
                elif s == 1 and rnd == 1 and k == 4:
                    recs.append(_r30(4, 1, 3.03 if p_ == "first" else 3.05, matched=3))
                elif s == 1 and rnd == 2 and k == 3 and p_ in ("first", "norm", "ff"):
                    recs.append(_r30(3, 1, 4.04 if p_ == "first" else 4.05, matched=4))
                else:
                    recs.append(_r30(k, s, k * f))
            out[p_] = recs
        return out
    cands1 = {k: stage(k, 1) for k in range(1, 8)}
    cands2 = {k: stage(k, 2) for k in range(3, 8)}
    sel1 = {k: {s: "first" for s in range(3)} for k in range(1, 8)}
    sel2 = {k: {s: "first" for s in range(3)} for k in range(3, 8)}
    sel2[3][1] = "long"
    plan = {"flagged": {"1": 3}, "flags": {"0": [], "1": [3], "2": []}, "j_min": 3}
    tags1 = {(k, p_): f"r1_{k}_{p_}" for k in range(1, 8) for p_ in cands1[k]}
    tags2 = {(k, p_): f"r2_{k}_{p_}" for k in range(3, 8) for p_ in cands2[k]}
    cost = {**{t: 1.0 for t in tags1.values()}, **{t: (4.0 if t.endswith("long") else 1.0) for t in tags2.values()}}
    first = [r for k in range(1, 7) for r in cands1[k]["first"]]
    return cands1, sel1, plan, cands2, sel2, cost, tags1, tags2, first


def test_e30_order_flags_and_restricted_selection():
    ch = {1: _r30(1, 0, 1.0), 2: _r30(2, 0, 4.0, matched=4), 3: _r30(3, 0, 2.0, matched=2), 4: _r30(4, 0, 4.1),
          7: _r30(7, 0, 3.5, spread=5e-3)}
    assert st.e30_order_flags(ch) == [2]                     # 단계 7은 사다리 불안정이라 경쟁자가 아니다
    ch[7] = _r30(7, 0, 3.5)
    assert st.e30_order_flags(ch) == [2, 4]                  # 단계 7이 마지막 보고 단계를 울린다
    recs = {"first": [_r30(3, 0, 4.0), _r30(3, 1, 4.0)], "norm": [_r30(3, 0, 3.1), _r30(3, 1, 3.1)],
            "long": [_r30(3, 0, 3.0), _r30(3, 1, 3.0)]}
    assert st.e30_select_stage(recs, 3, {0: ("first", "norm"), 1: ("first", "norm", "long")}) == {0: "norm", 1: "long"}


def test_e30_counts_the_final_chain_and_the_cost_each_seed_called():
    cands1, sel1, plan, cands2, sel2, cost, tags1, tags2, first = _e30_case()
    out = st.e30_from_records(cands1, sel1, plan, cands2, sel2, cost, tags1, tags2, first, 6.0, B=200)
    assert out["round1_accurate"] == 16 and out["accurate"] == 18 and out["first_pass_accurate"] == 16
    assert out["accurate_by_stage"] == {k: 3 for k in range(1, 7)}
    assert (out["flags_round1"], out["false_positives_round1"], out["flags_final"], out["false_positives_final"]) == (1, 0, 0, 0)
    assert out["round2"]["1"]["chosen"] == "long" and out["round2"]["1"]["accurate_after"] is True
    assert out["round2_reproduces_round1"] == {"compared": 10, "max_rel_dlam": 0.0, "mismatch": 0}
    c = out["cost"]
    assert c["round1_gpu_h_per_seed"] == 13.0 and c["ensemble_gpu_h"] == 27.0
    assert c["per_seed_ratio"] == {"0": pytest.approx(13 / 6), "1": pytest.approx(25 / 6), "2": pytest.approx(13 / 6)}
    assert c["mean_ratio"] == pytest.approx((2 * 13 + 25) / 18)
    assert out["d_first"]["verdict"] in ("rule1", "rule2")
    bad = {**plan, "flags": {"0": [], "1": [], "2": []}}
    with pytest.raises(ValueError):                         # 계획의 울림은 1회차 사슬에서 다시 계산한 것과 같아야 한다
        st.e30_from_records(cands1, sel1, bad, cands2, sel2, cost, tags1, tags2, first, 6.0, B=200)
    cands2[5]["first"][0] = _r30(5, 0, 5.06)                # 울리지 않은 시드의 2회차가 1회차를 되풀이하지 않으면 센다
    out = st.e30_from_records(cands1, sel1, plan, cands2, sel2, cost, tags1, tags2, first, 6.0, B=200)
    assert out["round2_reproduces_round1"]["mismatch"] == 1 and out["accurate"] == 18


def test_e30_chain_quality_counts_testability_flags_and_provisional_stages():
    """보완 사전규정 §4.4 — 단계 j는 뒤 단계(≤ 7) 가운데 사다리를 통과한 것이 있으면 검사 가능하다. 잠정(고른 해가 사다리 불안정)
    뒤의 단계는 위쪽 잠정이다."""
    ch0 = {k: _r30(k, 0, float(k) * 1.01) for k in range(1, 8)}                   # 모두 정확, 단계 7까지 안정
    ch1 = {**{k: _r30(k, 1, float(k) * 1.01) for k in range(1, 8)},
           3: _r30(3, 1, 4.04, matched=4), 4: _r30(4, 1, 3.03, matched=3),      # 단계 3이 높은 모드, 단계 4가 그 아래 모드
           5: _r30(5, 1, 5.05, spread=5e-3)}                                     # 단계 5는 정확하지만 잠정
    ch2 = {**{k: _r30(k, 2, float(k) * 1.01) for k in range(1, 6)},
           6: _r30(6, 2, 6.9), 7: _r30(7, 2, 7.07, spread=5e-3)}                # 단계 6 고유값 오차, 단계 7 불안정
    q = st.e30_chain_quality({0: ch0, 1: ch1, 2: ch2})
    assert q["accurate"] == 6 + 4 + 5 and q["all_accurate_seeds"] == 1
    assert q["provisional"] == 1 and q["upstream_provisional"] == 1              # 시드 1의 단계 5, 그 뒤 단계 6
    assert q["untestable_inaccurate"] == 1 and q["untestable_accurate"] == 0     # 시드 2의 단계 6(뒤 단계 7이 불안정)
    assert q["testable_inaccurate"] == 2                                         # 시드 1의 단계 3·4
    assert q["testable_inaccurate_flagged"] == 1                                 # 단계 3만 울린다(단계 4는 뒤 단계보다 낮다)
    assert q["testable_accurate"] == 6 + 4 + 5
    assert q["testable_accurate_flagged"] == 0
    assert q["seed_accurate_counts"] == {"0/6": 0, "1/6": 0, "2/6": 0, "3/6": 0, "4/6": 1, "5/6": 1, "6/6": 1}


def test_sorted_seed_accurate_reorders_by_the_audit_eigenvalue():
    recs = [_r30(k, 0, float(k) * 1.01) for k in range(1, 7)]
    swap = [_r30(1, 1, 1.01), _r30(2, 1, 3.03, matched=3), _r30(3, 1, 2.02, matched=2)] + [
        _r30(k, 1, float(k) * 1.01) for k in range(4, 7)]
    assert st.sorted_seed_accurate(recs + swap) == {0: 6, 1: 6}
    bad = [_r30(k, 2, float(k) * 1.2) if k == 6 else _r30(k, 2, float(k)) for k in range(1, 7)]
    assert st.sorted_seed_accurate(bad) == {2: 5}


def test_b3_compares_one_uniform_run_with_the_e30_chains():
    """B3(E31)은 한 실행이라 단계마다 기록이 하나다. 대비는 E30 − B3(E30이 나으면 규칙 1)과 B3 − 첫 통과이고, 순서 검사와
    정렬은 판독으로만 센다(단계 7이 없어 단계 6은 검사할 수 없다)."""
    def chain(s, skip):
        if not skip:
            return [_r30(k, s, float(k) * 1.01) for k in range(1, 7)]
        return [_r30(1, s, 1.01), _r30(2, s, 3.03, matched=3), _r30(3, s, 2.02, matched=2)] + [
            _r30(k, s, float(k) * 1.01) for k in range(4, 7)]
    b3 = [r for s in range(20) for r in chain(s, s < 5)]
    good = [r for s in range(20) for r in chain(s, False)]
    first = [r for s in range(20) for r in chain(s, s < 10)]
    r = st.b3_from_records(b3, good, good, first, 45.0, 18.0, B=2000)
    assert r["accurate"] == 110 and r["accurate_by_stage"] == {1: 20, 2: 15, 3: 15, 4: 20, 5: 20, 6: 20}
    assert (r["e30_final_accurate"], r["e30_round1_accurate"], r["first_pass_accurate"]) == (120, 120, 100)
    assert r["d_e30_final"]["verdict"] == "rule1" and r["d_e30_final"]["contrast"][0] == pytest.approx(10 / 120)
    assert r["d_first"]["verdict"] == "rule1" and r["d_first"]["contrast"][0] == pytest.approx(10 / 120)
    oc = r["order_check"]
    assert oc["flags"] == 5 and oc["false_positives"] == 0 and oc["inaccurate_stages_1_5"] == 10
    assert oc["by_seed"] == {str(s): [2] for s in range(5)}
    assert r["sorted_accurate"] == 120 and r["cost"]["ratio"] == pytest.approx(2.5)


def test_e31r_picks_stage_6_among_the_four_round_two_paths():
    """E31r(사후 판독): 단계 6만 네 경로(1회차 first·norm, ff, long)에서 2회차 규칙으로 고른 가상 사슬을 1회차 사슬과 견준다.
    사다리 불안정한 정확 해는 안정한 부정확 해에 밀린다(시드 11) — 정확한 경로가 있었는데 놓친 시드로 센다."""
    chain = [_r30(k, s, float(k) * 1.01) for s in range(20) for k in range(1, 6)] + \
        [_r30(6, s, 6.18, matched=7) for s in range(20)]
    first = [_r30(6, s, 6.18, matched=7) for s in range(20)]
    norm = [_r30(6, s, 6.20, matched=7) for s in range(20)]
    ff = [_r30(6, s, 6.06) if s < 5 else _r30(6, s, 6.19, matched=7) for s in range(20)]
    long = [_r30(6, s, 6.05) if s < 10 else _r30(6, s, 6.40) if s == 10 else
            _r30(6, s, 6.05, spread=5e-3) if s == 11 else _r30(6, s, 6.19, matched=7) for s in range(20)]
    r = st.e31r_from_records({"first": first, "norm": norm, "ff": ff, "long": long}, chain, B=2000)
    assert r["accurate_by_path"] == {"first": 0, "norm": 0, "ff": 5, "long": 11}
    assert r["mode6_shape_by_path"]["long"] == 12 and r["unstable_by_path"]["long"] == 1
    assert r["chosen"] == {"first": 10, "norm": 0, "ff": 0, "long": 10}
    assert r["stage6_accurate"] == 10 and r["seeds_with_an_accurate_path"] == 11 and r["selection_missed"] == [11]
    assert (r["round1_accurate"], r["hypothetical_accurate"]) == (100, 110)
    assert r["d_round1"]["verdict"] == "rule1" and r["d_round1"]["contrast"][0] == pytest.approx(10 / 120)
    assert r["per_seed"]["0"]["chosen"] == "long" and r["per_seed"]["11"]["long"] == [6.05, True]


# ── E31p 지도 적합 판독 (사전규정 2026-11-08-p5-e31p-prespecification.md) ──

def test_e31p_compares_the_deflated_fit_with_the_stage_6_of_each_seed():
    st6 = {s: 1.70 + 0.001 * s for s in range(20)}
    rows6 = [{"seed": str(s), "Lam_deflated": 1.65 if s < 19 else 1.80, "e_lam_check_0": 0.01, "mac_check_0": 0.999}
             for s in range(20)]
    rows7 = [{"seed": str(s), "Lam_deflated": 1.69, "e_lam_check_0": 0.008, "mac_check_0": 0.998} for s in range(20)]
    r = st.e31p_from_rows(rows6, rows7, st6, {s: 1.705 for s in range(20)})
    assert r["verdict"] == "rule1" and r["n_below_own_stage6"] == 19 and r["control_ok"] is True
    assert r["lam_deflated_median"] == pytest.approx(1.65) and r["lam_stage6_median"] == pytest.approx(1.7095)
    assert r["n7_below_own_stage6"] == 20 and r["n_below_round1_stage6"] == 19
    assert r["mac_held_out_median"] == {"6": pytest.approx(0.999), "7": pytest.approx(0.998)}
    bad = [{**x, "Lam_deflated": 1.75} for x in rows6]
    assert st.e31p_from_rows(bad, rows7, st6)["verdict"] == "rule2"
    assert "n_below_round1_stage6" not in st.e31p_from_rows(bad, rows7, st6)
    assert st.e31p_from_rows(rows6, [{**x, "e_lam_check_0": 0.2} for x in rows7], st6)["control_ok"] is False
    assert st.E31P_PREDICTION == {"verdict": "rule1", "n_below_own_stage6_at_least": 18, "control_ok": True}


def test_e31p_needs_both_fits(tmp_path):
    assert st.e31p(str(tmp_path)) is None


# ── E32 대수 층 보강 (사전규정 2026-11-09-p5-e32-prespecification.md) ──

def _e32_rec(m, s, ok=True, pm=0.5, off=None, spread=1e-5, kept=True, e=None):
    """게이트 없는 정확(`accurate`)은 짝 모드·MAC·e_λ로 정한다. 실패 셀은 형상을 지켰는지(`kept`)를 고른다."""
    return {"mode": m, "seed": s, "outcome": "correct" if ok else "spurious", "matched_mode": m if (ok or kept) else 1,
            "mac": 0.99 if (ok or kept) else 0.3, "e_lam": (0.01 if ok else 0.4) if e is None else e,
            "projected_mass_fraction": pm, "batch_offdiag": off, "ladder_tail_spread": spread}


def _e32_run(lost=(), off=(0.0, 0.03, 0.05, 0.055), pm_lost=0.001, n=20, kept_seeds=12):
    out = []
    for s in range(n):
        for m in range(1, 7):
            bad = m in lost
            out.append(_e32_rec(m, s, ok=not bad, pm=pm_lost if bad else 0.5, off=None if m < 3 else off[m - 3],
                                kept=s < kept_seeds))
    return out


def test_e32_pair_contrasts_the_exact_projection_with_the_storage_practice_seed_by_seed():
    ex, sto = _e32_run(), _e32_run(lost=(5, 6))
    r = st.e32_pair(ex, sto)
    assert r["verdict"] == "rule1" and r["contrast"][0] == pytest.approx(40 / 60)
    assert r["exact"]["acc_4_6"] == 60 and r["storage"]["acc_4_6"] == 20
    assert r["storage"]["accurate_by_mode"] == {1: 20, 2: 20, 3: 20, 4: 20, 5: 0, 6: 0}
    f = r["storage"]["failing_4_6"]
    assert f["n"] == 40 and f["shape_kept"] == 24 and f["outcomes"] == {"spurious": 40}
    assert f["remaining_mass_median"] == pytest.approx(0.001) and f["e_lam_median_by_mode"][5] == pytest.approx(0.4)
    assert r["storage"]["remaining_mass_median_by_mode"][5] == pytest.approx(0.001)
    assert r["storage"]["drift_median_by_stage"] == {3: 0.0, 4: 0.03, 5: 0.05, 6: 0.055}
    assert r["exact"]["ladder_stable"] == 120
    assert st.e32_pair(sto, sto)["verdict"] == "rule2"
    assert st.e32_pair(sto, ex)["verdict"] == "rule3"


def test_e32_drift_law_fits_the_log_slope_over_the_sample_sizes():
    d = {B: {4: 0.94 / np.sqrt(B), 5: 1.58 / np.sqrt(B)} for B in (283, 1134, 4536)}
    law = st.e32_drift_law(d, stages=(4, 5))
    assert law[4]["slope"] == pytest.approx(-0.5, abs=1e-9) and law[5]["ratio_283_1134"] == pytest.approx(np.sqrt(1134 / 283))
    assert law[4]["points"][0] == [283, pytest.approx(0.94 / np.sqrt(283))]
    two = st.e32_drift_law({1134: {4: 0.028}, 4536: {4: 0.014}}, stages=(4,))
    assert two[4]["slope"] is None and two[4]["ratio_283_1134"] is None


def test_e32_hits_score_each_registered_prediction():
    tb_ex, tb_sto = _e32_run(), _e32_run(lost=(5, 6))
    sto = {283: _e32_run(lost=(4, 5, 6), off=(0.0, 0.056, 0.094, 0.11)), 1134: _e32_run(lost=(5, 6)),
           4536: _e32_run(lost=(5,), off=(0.0, 0.015, 0.023, 0.027))}
    rows = [{"source": "e26_tb_raw_mc", "B": "1651", "stage": str(k), "seed": "0", "drift": str(v)}
            for k, v in ((3, 0.001), (4, 0.03), (5, 0.05), (6, 0.05))]
    r = st.e32_from_records(tb_sto, tb_ex, sto, _e32_run(), rows)
    assert r["drift_prediction"]["tbracket_from_e26d"] == {3: 0.001, 4: 0.03, 5: 0.05, 6: 0.05}
    assert r["b283"]["verdict"] == "rule1" and r["storage_by_B"][4536]["acc_4_6"] == 40
    pred = {"tb_verdict": "rule1", "tb_storage_acc_4_6_at_most": 40, "tb_storage_modes_at_most": {5: 0, 6: 0},
            "tb_failing_mass_at_most": 0.01, "tb_drift_factor": 1.25, "tb_drift_stages": [4, 5],
            "b283_ratio_range": [1.6, 2.5], "b283_ratio_stages": [4, 5], "b283_verdict": "rule1",
            "b283_exact_acc_4_6_at_least": 50, "b283_storage_modes_at_most": {4: 10}}
    assert all(st.e32_hits(r, pred).values()) and len(st.e32_hits(r, pred)) == 9
    miss = st.e32_hits(r, {**pred, "tb_drift_factor": 1.05, "tb_drift_stages": [6], "b283_storage_modes_at_most": {4: -1},
                           "tb_storage_modes_at_most": {4: 10}})
    assert miss["tb_drift"] is False and miss["b283_storage_modes"] is False and miss["tb_storage_modes"] is False
    assert st.e32_hits(r, {}) == {}
    assert set(st.e32_hits(r, st.E32_PREDICTION)) == {"tb_verdict", "tb_storage_modes", "tb_failing_mass", "tb_drift",
                                                       "b283_verdict", "b283_exact_acc_4_6", "b283_ratio"}
    assert st.E32_PREDICTION["tb_storage_modes_at_most"] == {4: 10, 5: 10}
    assert st.E32_PREDICTION["tb_drift_stages"] == [3, 4] and st.E32_PREDICTION["b283_ratio_range"] == [1.7, 2.5]


def test_e32_needs_every_run(tmp_path):
    assert st.e32(str(tmp_path)) is None
