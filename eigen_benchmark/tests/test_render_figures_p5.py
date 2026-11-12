"""논문4.P5의 그림 시험 — 2026-10-02 test_render_figures.py에서 떼어 냈다(P5 공개본이 이 파일만
담을 수 있도록). 그림은 data/paper4의 커밋된 산출물에서만 그린다."""
import os

import pytest

pytest.importorskip("matplotlib")
from eigen_benchmark.render import figures_p5      # noqa: E402

DATA_P5 = "docs/_generated/data/paper4"


@pytest.mark.parametrize("name", sorted(figures_p5.ALL))
def test_every_p5_figure_writes_a_nonempty_png(tmp_path, name):
    p = figures_p5.ALL[name](DATA_P5, str(tmp_path))
    assert os.path.exists(p) and os.path.getsize(p) > 5000


def test_p5_ladder_threshold_matches_the_driver():
    """그림이 쓰는 해상 문턱은 드라이버가 판정에 쓰는 값과 같아야 한다 — 다르면
    그림과 표가 다른 셀 집합을 그린다."""
    from eigen_benchmark.drivers import run_p5_neural as run
    assert figures_p5.LADDER_RESOLVED == run.LADDER_RESOLVED


def test_p5_atlas_wall_rule_reproduces_the_committed_e13_predictions():
    """그림 1이 테두리로 표시하는 모드는 성분 공유 규칙의 예측이다. E13 두 구조의 예측은
    신경 실행 전에 커밋되었으므로(49460de) 렌더러가 같은 규칙으로 같은 모드를 짚어야 한다."""
    import json
    for tag in ("e13a_thick", "e13b_clamped2"):
        with open(os.path.join(DATA_P5, f"p5_reference_{tag}.json"), encoding="utf-8") as f:
            ref = json.load(f)
        assert figures_p5.predicted_wall(ref["dominant"]) == ref["wall_predicted_at"], tag
    assert figures_p5.predicted_wall(list("zyzzzy")) == 4        # 본실험(관측)


def test_p5_atlas_reads_its_shapes_from_the_committed_artifact_only():
    """렌더러는 FEM을 풀지 않는다 — 문제 모듈을 불러오지 않아야 한다."""
    import inspect
    src = inspect.getsource(figures_p5)
    assert "p5_block" not in src and "solve(" not in src


def test_p5_atlas_adds_the_two_further_structures_with_the_wall_at_mode_four():
    """그림 1 확장(2026-10-09) — L-브래킷과 두 층 구조 행. 두 구조 모두 성분 공유 규칙이 벽을 모드 4로
    짚는다(원고 4.7절, 사전규정 2026-10-04)."""
    import json
    tags = [t for t, *_ in figures_p5.ATLAS]
    assert tags[-3:] == ["e21_lbracket", "e22_bilayer", "e25_tbracket"]
    assert set(figures_p5.ATLAS_EXTRA) == set(tags[-3:])
    for tag in figures_p5.ATLAS_EXTRA:
        p = os.path.join(DATA_P5, f"p5_mode_atlas_{tag}.json")
        if not os.path.exists(p):
            continue                       # T-브래킷 도감은 참조 뒤에 만든다(run_p5.extra_atlas)
        with open(p, encoding="utf-8") as f:
            atlas = json.load(f)
        assert figures_p5.predicted_wall([m["dominant"] for m in atlas["modes"]]) == 4, tag


def test_block_faces_skip_inner_faces_and_keep_the_default_order():
    import numpy as np
    axes = (np.linspace(0, 1, 3), np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    full = figures_p5._block_faces(P)
    assert len(full) == 10
    assert len(figures_p5._block_faces(P, frozenset({"x1"}))) == 9
    assert np.array_equal(figures_p5._block_faces(P, frozenset()), full)


def test_p5_stage6_counts_match_the_committed_auxiliary_reading():
    """그림 5가 쓰는 문턱 아래 시드 수 세기는 E21h 보조 판독(`p5_exploit.e21h_aux`, 커밋된
    `p5_exploit_e21h_hist.csv`)이 같은 이력에서 센 값과 같아야 한다 — 다르면 그림과 원고 본문이 다른 수를 말한다."""
    import csv
    rows = list(csv.DictReader(open(os.path.join(DATA_P5, "p5_exploit_e21h_hist.csv"), encoding="utf-8")))
    assert {r["tag"] for r in rows} == {tag for tag, *_ in figures_p5.STAGE6[:3]}
    p23 = os.path.join(DATA_P5, "p5_exploit_e23_hist.csv")        # r2: E23 두 조건(보조 판독이 생긴 뒤)
    if os.path.exists(p23):
        r23 = list(csv.DictReader(open(p23, encoding="utf-8")))
        assert {r["tag"] for r in r23} == {tag for tag, *_ in figures_p5.STAGE6[3:5]}
        rows += r23
    for thr in (1.45, 1.32):
        for r in rows:
            its, n = figures_p5.stage6_counts(DATA_P5, r["tag"], threshold=thr)
            i = int(r["it"]) // int(its[1] - its[0])
            assert int(its[i]) == int(r["it"])
            assert int(n[i]) == int(r[f"n_below_{thr:g}"]), (r["tag"], r["it"], thr)


def test_p5_stage6_panel_c_holds_conditions_iii_to_viii_of_table_6():
    """그림 6(c)는 표 6의 (iii)–(viii) 여섯 조건이고, 색은 범주형 슬롯 1–6을 고정 순서로 쓴다(순환 금지)."""
    tags = [t for t, *_ in figures_p5.STAGE6]
    assert tags == ["e21lb_s6_norm_x4", "e21lb_s6_raw_x4", "e21lb_s6_norm_w256", "e21lb_s6_norm_x4lo",
                    "e21lb_s6_norm_ff", "e21lb_s6_raw_ff"]
    assert [c for _, _, c, _ in figures_p5.STAGE6] == [figures_p5.C_DIAG, figures_p5.C_EXACT, figures_p5.C_QR,
                                                     figures_p5.C_FIXED, figures_p5.C_REORTH, figures_p5.C_SLOT6]
    assert len({str(ls) for *_, ls in figures_p5.STAGE6}) == 6


def test_p5_stage6_reductions_follow_the_fourfold_schedule():
    """세로선의 학습률 감소 시점은 반복 4배 일정의 StepLR(주기 = 반복 // 3)과 같아야 한다."""
    step = 80_000 // 3
    assert figures_p5.STAGE6_REDUCTIONS == (step, 2 * step)


def test_strong_form_figure_maps_stages_to_modes_and_reads_the_green_split(tmp_path):
    """그림 7(r2) — 단계 → 참조 모드 지도(MAC < 0.9는 0)와 Green 분해 산점도. 합성 기록으로 시험하고, Green 산출물이
    없는 구조는 (c)에서 빠진다."""
    import csv as csv_
    import json
    strong = "p5_pinn_3d[strong form, orthogonality penalty]"
    order = {1: 2, 2: 1, 3: 6, 4: 5, 5: 3, 6: 4}
    def rows(cond):
        return [{"arm": strong, "condition": cond, "mode": k, "seed": s, "outcome": "spurious",
                 "matched_mode": order[k], "mac": 0.97 if s < 15 else 0.5, "e_lam": 0.4}
                for k in range(1, 7) for s in range(20)]
    for name, cond in (("e24_pinn_thick", "resampled"), ("e24_pinn_lb", "mc")):
        with open(tmp_path / f"p5_neural_{name}_records.jsonl", "w", encoding="utf-8") as f:
            for r in rows(cond):
                f.write(json.dumps(r) + "\n")
    cnt = figures_p5.stage_mode_counts(rows("mc"))
    assert cnt[(1, 2)] == 15 and cnt[(1, 0)] == 5 and sum(cnt.values()) == 120
    with open(tmp_path / "p5_exploit_green_e24_pinn_thick.csv", "w", encoding="utf-8", newline="") as f:
        w = csv_.DictWriter(f, fieldnames=["Lam_grid", "lam_param", "surface_work"])
        w.writeheader()
        for i in range(30):
            w.writerow({"Lam_grid": 0.05, "lam_param": 0.05 + 0.02 * i, "surface_work": -0.02 * i})
    p = figures_p5.fig_strong(data_dir=str(tmp_path), outdir=str(tmp_path))
    assert os.path.getsize(p) > 10_000


def test_readouts_figure_carries_the_committed_first_pass_and_original_totals():
    """그림 7은 원고 표 5를 옮긴다 — 여덟 구조의 계산된 그대로·정렬 합계는 커밋된 사후 판독(order_check)과 같아야 한다."""
    import json
    with open(os.path.join(DATA_P5, "p5_posthoc.json"), encoding="utf-8") as f:
        oc = json.load(f)["order_check"]
    rows = {(g, r): (c, o) for g, r, n, c, o, rr, cost in figures_p5.READOUTS}
    for group, run in (("original", "Original setting"), ("first_pass", "First pass")):
        lads = {k: v for k, v in oc[group]["ladders"].items() if k != "(viii) normalized"}
        assert rows[("Eight structures", run)] == (sum(v["accurate"] for v in lads.values()),
                                                   sum(v["reordered_accurate"] for v in lads.values()))
    for lab, key in (("(iii) Clamped–clamped block", "(iii)"), ("(v) Near-degenerate section", "(v)"),
                     ("(vi) L-bracket", "(vi)")):
        v = oc["first_pass"]["ladders"][key]
        assert rows[(lab, "First pass")] == (v["accurate"], v["reordered_accurate"])


def test_order_check_figure_reads_the_thresholds_of_the_manuscript():
    """그림 6의 문턱: 등록 t = 0.05와, δ² ≤ 10⁻³·ξ ≤ 4.4 × 10⁻³에서 감사 고유값의 e_λ = 0.05를 인증하는 식 (9)의
    t_req = 1 − (1 − β)(1 − ξ)/(1 + e_λ) = 0.0528(원고 3.3절, 보충 Note S9; r6.5에서 R(u) 기준 0.0569를 바꿨다)."""
    from eigen_benchmark.drivers import p5_posthoc as ph
    assert figures_p5.ORDER_T == ph.ORDER_T == 0.05
    assert figures_p5.ORDER_T_CERT == round(1 - (1 - 1e-3) * (1 - 4.4e-3) / 1.05, 4) == 0.0528


def test_algebra_band_holds_every_cell_with_a_residual_and_reads_c_from_the_mass_weighted_form():
    """그림 5(b)의 띠 1 − cζ/(1 + ζ)(r6.4): c는 ζ > 0.05인 저장 관행 셀의 (1 − 비)(1 + ζ)/ζ 범위이므로 그 셀이 모두 띠
    안에 든다. 학습 몫이 질량 가중 평균이고 잔차의 몫이 정확히 뺀 출력의 절반이면 c = 0.5이다."""
    import csv
    rows = [{"stage": "5", "ratio_mean": str(1 - 0.5 * z / (1 + z)), "leak_mean": str(z)} for z in (0.1, 1.0, 10.0)]
    rows.append({"stage": "3", "ratio_mean": "0.1", "leak_mean": "5.0"})
    assert figures_p5.algebra_band(rows) == pytest.approx((0.5, 0.5), rel=1e-12)
    with open(os.path.join(DATA_P5, "p5_exploit_e32_mechanism.csv"), encoding="utf-8") as f:
        mech = list(csv.DictReader(f))
    lo, hi = figures_p5.algebra_band(mech)
    assert 0.40 < lo < hi < 0.95
    for r in mech:
        z, y = float(r["leak_mean"]), float(r["ratio_mean"])
        if int(r["stage"]) >= 4 and z > figures_p5.ALGEBRA_BAND_ZETA:
            assert 1 - hi * z / (1 + z) - 1e-12 <= y <= 1 - lo * z / (1 + z) + 1e-12
