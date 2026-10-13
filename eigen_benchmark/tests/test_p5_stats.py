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
