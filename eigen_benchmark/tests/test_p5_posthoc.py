"""r4 사후 판독(`p5_posthoc`) — 감사–최상단 거리, 순서 검사의 분모, 정렬 판독, 저장 모드 식별."""
import pytest

from eigen_benchmark.drivers import p5_posthoc as ph
from eigen_benchmark.drivers.run_p5_neural import ladder_tail_spread

LAM = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}


def _rec(seed, stage, mode, mac, lam, lam_ref=None, spread=1e-6, outcome=None):
    ref = LAM[stage] if lam_ref is None else lam_ref
    e = abs(lam - ref) / ref
    acc = mode == stage and mac >= 0.9 and e <= 0.05
    return {"seed": seed, "mode": stage, "matched_mode": mode, "mac": mac, "Lam": lam, "Lam_ref": ref, "e_lam": e,
            "outcome": outcome or ("correct" if acc else "spurious"), "ladder_tail_spread": spread,
            "lam_ladder": [lam, lam, lam, lam]}


def test_d_audit_reads_the_audit_rung_that_the_tail_spread_omits():
    """검토의 반례 — (1.20, 1.00, 1.00, 1.00)은 꼬리 폭이 0이라 안정이지만 감사 값은 최상단에서 20 % 떨어져 있다."""
    lad = [1.20, 1.00, 1.00, 1.00]
    assert ladder_tail_spread(lad) == 0.0
    assert ph.d_audit({"lam_ladder": lad}) == pytest.approx(0.2)


def test_order_check_counts_testable_cells_and_reordering_repairs_a_skip_found_later():
    """시드 0: 단계 2가 모드 4에 서고 단계 3·4가 건너뛴 모드 2·3을 찾았다 — 순서 검사는 단계 2만 울리고, 정렬하면 네 셀이 모두
    맞는다. 시드 1: 단계 3은 형상이 맞고 고유값이 10 % 높으며, 마지막 단계는 모드 1–4 어느 것과도 닮지 않았다(더 높은 모드)."""
    seeds = {0: {1: _rec(0, 1, 1, 1.0, 1.0), 2: _rec(0, 2, 4, 1.0, 4.0), 3: _rec(0, 3, 2, 1.0, 2.0), 4: _rec(0, 4, 3, 1.0, 3.0)},
             1: {1: _rec(1, 1, 1, 1.0, 1.0), 2: _rec(1, 2, 2, 1.0, 2.0), 3: _rec(1, 3, 3, 0.99, 3.3),
                 4: _rec(1, 4, 2, 0.01, 5.5)}}
    c = ph.order_check(seeds)
    assert c["flags"] == [(0, 2)]
    assert (c["accurate"], c["accurate_testable"], c["accurate_flagged"]) == (3, 3, 0)
    assert (c["nonterminal_higher"], c["nonterminal_higher_flagged"]) == (1, 1)
    assert (c["nonterminal_lower"], c["nonterminal_lower_flagged"]) == (1, 0)
    assert (c["nonterminal_other"], c["nonterminal_other_flagged"]) == (1, 0)
    assert (c["terminal_inaccurate"], c["terminal_lower"]) == (2, 1)
    assert c["reordered_accurate"] == 6
    assert c["reorder_fail"] == {"beyond_modes": 1, "mixture": 0, "misaligned": 0, "eigenvalue": 1}


def test_order_check_needs_a_later_stage_that_passes_the_ladder():
    seeds = {0: {1: _rec(0, 1, 1, 1.0, 1.0), 2: _rec(0, 2, 4, 1.0, 4.0), 3: _rec(0, 3, 2, 1.0, 2.0, spread=5e-3),
                 4: _rec(0, 4, 3, 1.0, 3.0)}}
    c = ph.order_check(seeds)
    assert c["flags"] == [(0, 2)]                     # 단계 4(안정, Λ = 3 < 0.95·4)가 대신 울린다
    seeds[0][4]["ladder_tail_spread"] = 5e-3
    assert ph.order_check(seeds)["flags"] == []


def test_holds_modes_is_a_property_of_the_span_not_of_the_order():
    lad = {1: _rec(0, 1, 1, 1.0, 1.0), 2: _rec(0, 2, 2, 1.0, 2.0), 3: _rec(0, 3, 4, 1.0, 4.0), 4: _rec(0, 4, 3, 1.0, 3.0),
           5: _rec(0, 4, 3, 1.0, 3.0)}
    assert ph.holds_modes(lad, 3)
    assert not ph.holds_modes(lad, 4)                 # 단계 1–3이 모드 1·2·4를 담았다
    assert ph.holds_modes(lad, 5)                     # 단계 1–4가 모드 1–4를 담았다(3·4는 바뀐 순서)
    lad[2]["mac"] = 0.84
    assert not ph.holds_modes(lad, 5)


def test_threshold_counts_split_flags_by_accuracy_and_skip_the_probe_stage():
    """r5.3 문턱 민감도: 단계 1..min(마지막−1, 6)을 뒤의 사다리 안정 단계와 견준다. 시드 0은 단계 2가 모드 4에 서고 단계 3이
    3 % 낮다(작은 t에서만 울림). 시드 1은 정확한 단계 2보다 단계 3이 0.5 % 낮다(t = 0에서만 거짓 양성). 단계 5는 보고하지
    않는 덧붙인 단계라 세지 않지만 앞 단계를 울릴 수 있고, 시드 1에서는 사다리를 통과하지 못해 단계 4를 검사하지 못한다."""
    s0 = {1: _rec(0, 1, 1, 1.0, 1.0), 2: _rec(0, 2, 4, 1.0, 4.0, lam_ref=2.0), 3: _rec(0, 3, 2, 1.0, 3.88, lam_ref=3.0),
          4: _rec(0, 4, 4, 1.0, 4.0), 5: _rec(0, 4, 4, 1.0, 9.0)}
    s1 = {1: _rec(1, 1, 1, 1.0, 1.0), 2: _rec(1, 2, 2, 1.0, 2.0), 3: _rec(1, 3, 2, 1.0, 1.99, lam_ref=3.0),
          4: _rec(1, 4, 4, 1.0, 4.0), 5: _rec(1, 4, 4, 1.0, 0.5, spread=5e-3)}
    c0 = ph.threshold_counts({0: s0, 1: s1}, 0.0, last_reported=4)
    assert (c0["inaccurate"], c0["inaccurate_flagged"]) == (3, 1)
    assert (c0["accurate_testable"], c0["accurate_flagged"]) == (4, 1)
    assert c0["accurate_deepest_drop"] == pytest.approx(0.005)
    c2 = ph.threshold_counts({0: s0, 1: s1}, 0.02, last_reported=4)
    assert (c2["inaccurate_flagged"], c2["accurate_flagged"]) == (1, 0)
    c5 = ph.threshold_counts({0: s0, 1: s1}, 0.05, last_reported=4)
    assert (c5["inaccurate_flagged"], c5["accurate_flagged"]) == (0, 0)
    assert ph.THRESHOLDS == (0.0, 0.01, 0.02, 0.03, 0.05)


def test_drop_rows_record_the_deepest_drop_that_the_order_check_compares_with_t():
    """r6.2 검토 그림 자료: 단계마다 뒤의 사다리 안정 단계 가운데 가장 깊은 하강. 어느 t에서든 하강 > t인 행의 수가
    `threshold_counts`의 울림과 같고, 퇴화 군집 안의 짝은 `same_cluster`로 표시된다."""
    s0 = {1: _rec(0, 1, 1, 1.0, 1.0), 2: _rec(0, 2, 4, 1.0, 4.0, lam_ref=2.0), 3: _rec(0, 3, 2, 1.0, 3.88, lam_ref=3.0),
          4: _rec(0, 4, 4, 1.0, 4.0), 5: _rec(0, 4, 4, 1.0, 9.0)}
    s1 = {1: _rec(1, 1, 1, 1.0, 1.0), 2: _rec(1, 2, 2, 1.0, 2.0), 3: _rec(1, 3, 2, 1.0, 1.99, lam_ref=3.0),
          4: _rec(1, 4, 4, 1.0, 4.0), 5: _rec(1, 4, 4, 1.0, 0.5, spread=5e-3)}
    s1[2]["cluster"] = s1[3]["cluster"] = [2, 3]
    rows = ph.drop_rows("g", "x", {0: s0, 1: s1}, last_reported=4)
    assert len(rows) == 8 and {r["stage"] for r in rows} == {1, 2, 3, 4}
    r02 = next(r for r in rows if (r["seed"], r["stage"]) == (0, 2))
    assert r02["class"] == "higher" and r02["by_stage"] == 3 and r02["drop"] == pytest.approx(0.03)
    r12 = next(r for r in rows if (r["seed"], r["stage"]) == (1, 2))
    assert r12["class"] == "accurate" and r12["drop"] == pytest.approx(0.005) and r12["same_cluster"]
    r14 = next(r for r in rows if (r["seed"], r["stage"]) == (1, 4))
    assert not r14["testable"] and r14["drop"] == ""            # 덧붙인 단계 5는 사다리를 통과하지 못했다
    for t in (0.0, 0.01, 0.02, 0.05):        # t = 0.03은 시드 0의 3 % 하강과 같아 반올림이 가르는 경계다
        c = ph.threshold_counts({0: s0, 1: s1}, t, last_reported=4)
        hit = [r for r in rows if r["drop"] != "" and r["drop"] > t]
        assert sum(r["class"] == "accurate" for r in hit) == c["accurate_flagged"]
        assert sum(r["class"] != "accurate" for r in hit) == c["inaccurate_flagged"]
