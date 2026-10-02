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
