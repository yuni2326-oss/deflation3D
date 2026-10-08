"""P5 참조 드라이버 — 산출물이 계획 §2의 요구를 담았는지."""
import csv
import json
import os

import pytest

from eigen_benchmark.drivers import run_p5


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    """드라이버를 **한 번만** 돌린다. 참조 사다리 최상단이 희소 풀이만 87초라
    테스트마다 다시 돌리면 스위트가 십 분을 넘긴다."""
    d = tmp_path_factory.mktemp("p5")
    return str(d), run_p5.main(outdir=str(d), quick=True)


def _rows(d, name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_driver_writes_all_four_certification_artifacts(out):
    tmp_path, res = out
    for name in ("p5_reference.csv", "p5_quadrature_study.csv",
                 "p5_shape_certification.csv", "p5_spectrum_gaps.csv",
                 "manifest_p5.json"):
        p = os.path.join(tmp_path, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, name
    assert res["grid"] is not None


def test_reference_rows_carry_the_uncertainty_the_plan_asks_for(out):
    """§2.1 — 참조값과 함께 외삽 잔차를 불확실도로 보고해야 한다. 잔차 없이
    참조값만 실으면 채점 게이트와 비교할 수 없다."""
    out[1]
    rows = _rows(out[0], "p5_reference.csv")
    assert len(rows) == run_p5.N_MODES_REF
    used = [r for r in rows if r["used_in_grid"] == "True"]
    assert len(used) == run_p5.N_MODES_USED
    for r in used:
        assert r["monotone"] == "True", r["mode"]
        assert 0.0 < float(r["rel_residual"]) < 0.05, r["mode"]
        # §2.5 sanity — 보 이론과 자릿수 수준 일치(참조가 아니라 기록용)
        assert 0.5 < float(r["sanity_ratio"]) < 2.0, r["mode"]


def test_quadrature_grid_is_the_smallest_that_meets_the_declared_tolerance(out):
    r = out[1]
    rows = _rows(out[0], "p5_quadrature_study.csv")
    ok = [x for x in rows if x["meets_tol"] == "True"]
    assert ok, "문턱을 만족하는 후보가 없다"
    chosen = [x for x in rows if x["chosen"] == "True"]
    assert len(chosen) == 1
    assert int(chosen[0]["n_points"]) == min(int(x["n_points"]) for x in ok)
    assert "x".join(map(str, r["grid"])) == chosen[0]["grid"]
    # 기준 격자가 스스로 수렴했는지 기록돼야 한다 — 없으면 인증이 순환이다
    assert float(rows[0]["reference_self_consistency"]) < 1e-10


def test_shape_certification_is_measured_at_the_scoring_quadrature_points(out):
    """§2.2 — 인증한 점과 채점이 쓰는 점이 달라선 안 된다.

    **문턱 통과는 여기서 단언하지 않는다.** 스모크는 축소 사다리로 돌고 그 최조밀
    단계는 인증을 낼 만큼 촘촘하지 않다 — 통과 여부는 실제 사다리의 산출물로 본다
    (아래 test_committed_reference_meets_every_threshold_the_plan_fixed)."""
    r = out[1]
    rows = _rows(out[0], "p5_shape_certification.csv")
    want = "x".join(map(str, r["grid"]))
    assert len(rows) == run_p5.N_MODES_USED
    for x in rows:
        assert x["grid"] == want
        assert 0.0 <= float(x["self_mac"]) <= 1.0
        assert float(x["threshold"]) == 0.999
        assert x["certified"] == str(float(x["self_mac"]) >= float(x["threshold"]))


def test_gap_rule_decision_is_recorded_for_every_scored_mode(out):
    """§2.3 — 쌍으로 채점할지 여부는 실행 전에 참조값으로 정해져야 한다."""
    out[1]
    rows = _rows(out[0], "p5_spectrum_gaps.csv")
    assert len(rows) == run_p5.N_MODES_USED
    for x in rows:
        assert x["scored_as_pair"] in ("True", "False")
        assert float(x["rule"]) == run_p5.GAP_RULE


def test_manifest_declares_what_was_not_run(out):
    """계획 §6의 공시 요구 — 미실행 항목을 숨기지 않는다."""
    with open(os.path.join(out[0], "manifest_p5.json"), encoding="utf-8") as f:
        m = json.load(f)
    assert "not_run" in m and m["not_run"]
    assert m["quick"] is True
    assert m["quadrature_grid"] and m["ladder"]
    # 왜 FEM 모드로 격자를 인증하지 않았는지가 남아야 한다
    assert "C0 FEM strain field" in m["quadrature_note"]


# ─────────────────────────────────────────────────────────────────────────────
# 실제 사다리의 산출물. 스모크가 아니라 **커밋된 값**이 계획의 문턱을 만족하는지
# 본다 — 참조를 쓰는 모든 후속 실험이 이 값에 기댄다.
# ─────────────────────────────────────────────────────────────────────────────

DATA = os.path.join("docs", "_generated", "data", "paper4")


def _committed(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        pytest.skip(f"{name}이 없다 — run_p5를 먼저 돌린다")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_committed_reference_meets_every_threshold_the_plan_fixed():
    ref = _committed("p5_reference.csv")
    cert = _committed("p5_shape_certification.csv")
    gaps = _committed("p5_spectrum_gaps.csv")
    used = [r for r in ref if r["used_in_grid"] == "True"]
    assert len(used) == run_p5.N_MODES_USED
    # §2.1 — 단조 수렴과, 채점 게이트(5 %)보다 훨씬 작은 참조 불확실도
    for r in used:
        assert r["monotone"] == "True", r["mode"]
    worst = max(float(r["rel_residual"]) for r in used)
    assert worst < 5e-3, worst
    # §2.2 — 전 모드 self-MAC 인증
    for x in cert:
        assert x["certified"] == "True", x["mode"]
    # §2.3 — 쌍 채점 판정이 모든 모드에 기록되고, 간격 규칙이 계획값이다
    assert len(gaps) == run_p5.N_MODES_USED
    for x in gaps:
        assert float(x["rule"]) == run_p5.GAP_RULE


def test_committed_quadrature_grid_is_certified_against_a_smooth_field():
    """FEM 모드로 인증하지 않았다는 사실이 산출물에 남아야 한다 — 그 이유가
    C⁰ 변형률 불연속이고, 그것은 P1 조각경계 정렬 요구의 3D 판본이다."""
    rows = _committed("p5_quadrature_study.csv")
    assert any("manufactured" in r["field"] for r in rows)
    ok = [r for r in rows if r["meets_tol"] == "True"]
    assert ok
    chosen = [r for r in rows if r["chosen"] == "True"]
    assert len(chosen) == 1
    assert int(chosen[0]["n_points"]) == min(int(r["n_points"]) for r in ok)


# ─────────────────────────────────────────────────────────────────────────────
# 모드 도감(원고 v4 그림 1) — 세 구조의 모드형상과 성격. 렌더러는 계산하지 않으므로
# 형상과 성격을 이 드라이버가 산출물로 쓴다.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np                                   # noqa: E402

from eigen_benchmark.problems import p5_block as p5  # noqa: E402


def _synthetic(kind):
    """폭 방향 두 줄(y = 0.15b, 0.85b)에서 표본한 가짜 변위 — 성격 규칙만 시험한다."""
    xs = np.linspace(0.05, 0.95, 25)
    U = np.zeros((25, 2, 3))
    w = np.sin(np.pi * xs)
    if kind == "bending":
        U[:, :, 2] = w[:, None]
    elif kind == "torsion":
        U[:, 0, 2], U[:, 1, 2] = -w, w
    elif kind == "lateral":
        U[:, :, 1] = w[:, None]
    elif kind == "axial":
        U[:, :, 0] = w[:, None]
    return U


@pytest.mark.parametrize("kind", ["bending", "torsion", "lateral", "axial"])
def test_mode_character_rule_separates_bending_from_torsion(kind):
    """z-지배는 굽힘과 비틀림을 합친 분류다(검토 2026-09-26 §2.3). 둘은 u_z의 폭 방향
    대칭성으로 갈린다 — 굽힘은 대칭, 비틀림은 반대칭."""
    U = _synthetic(kind)
    frac = (U ** 2).sum(axis=(0, 1)) / (U ** 2).sum()
    assert run_p5.mode_character(U, frac)["character"] == kind


def test_sign_normalization_is_invariant_to_the_eigensolvers_sign():
    """참조 고유벡터는 시작벡터가 고정되지 않아 부호가 실행마다 뒤집힐 수 있다."""
    rng = np.random.default_rng(0)
    U = rng.normal(size=(40, 3))
    assert np.array_equal(run_p5.normalize_sign(U), run_p5.normalize_sign(-U))


def test_mass_weighted_fraction_reproduces_table_1_of_the_manuscript():
    """원고 표 1의 지배 성분 열은 질량가중, 24 × 12 × 10 Gauss에서다. 비가중으로 재면
    모드 6이 0.928이 아니라 0.889가 된다(E13 참조 JSON이 비가중이었다)."""
    r = p5.solve(24, 8, 4, n_modes=6)
    fr = run_p5.component_fractions(r, geometry=p5.P5_GEOMETRY)
    assert np.round(fr.max(axis=1), 3).tolist() == [0.999, 0.986, 0.996, 0.961, 0.990, 0.928]
    assert "".join("xyz"[i] for i in fr.argmax(axis=1)) == "zyzzzy"


def test_committed_mode_atlas_agrees_with_the_references_it_draws():
    """도감의 지배 성분 라벨이 사전 커밋된 참조와 같아야 한다 — E13 예측이 그 라벨을 썼다."""
    p = os.path.join(DATA, "p5_mode_character.json")
    if not os.path.exists(p):
        pytest.skip("p5_mode_character.json이 없다 — run_p5.mode_atlas를 먼저 돌린다")
    with open(p, encoding="utf-8") as f:
        atlas = json.load(f)
    for tag in ("e13a_thick", "e13b_clamped2"):
        with open(os.path.join(DATA, f"p5_reference_{tag}.json"), encoding="utf-8") as f:
            ref = json.load(f)
        got = [m["dominant"] for m in atlas["structures"][tag]["modes"]]
        assert got == ref["dominant"], tag
    base = [m["dominant"] for m in atlas["structures"]["baseline"]["modes"]]
    assert "".join(base) == "zyzzzy"
    for s in atlas["structures"].values():
        for m in s["modes"][:4]:
            assert m["self_mac_levels"] >= 0.999


def test_sign_normalization_is_robust_to_a_tied_extreme():
    """비틀림 모드는 최대 |u_z|가 두 모서리에서 크기가 같고 부호가 반대다. 어느 쪽이 더 큰지는
    풀이 잡음이 정하므로, 동률을 잡음에 맡기면 모드 전체의 부호가 실행마다 뒤집힌다."""
    U = np.zeros((6, 3))
    U[1, 2], U[4, 2] = -1.0, 1.0
    a, b = U.copy(), U.copy()
    a[1, 2] -= 1e-13          # 음의 모서리가 근소하게 큼
    b[4, 2] += 1e-13          # 양의 모서리가 근소하게 큼
    assert np.sign(run_p5.normalize_sign(a)[1, 2]) == np.sign(run_p5.normalize_sign(b)[1, 2])


def test_width_symmetry_is_recorded_only_where_it_is_defined():
    """y·x-지배 모드의 u_z는 두께 중앙면에서 대칭상 0이라 남는 값은 풀이 잡음이다."""
    for kind in ("lateral", "axial"):
        U = _synthetic(kind)
        frac = (U ** 2).sum(axis=(0, 1)) / (U ** 2).sum()
        assert run_p5.mode_character(U, frac)["width_symmetry"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 구조별 참조(E16 정사각 단면의 전제). E13 참조는 커밋된 코드 없이 만들어졌다 — 이번에는
# 참조·군집·인증을 함수로 남긴다. 축퇴쌍은 단계마다 쌍 안의 회전이 달라 모드별 self-MAC이
# 원리적으로 실패하므로, 쌍은 부분공간(최소 주각의 cos²)으로 인증한다.
# ─────────────────────────────────────────────────────────────────────────────

SMALL_LADDER = [(6, 2, 2), (12, 4, 4), (24, 8, 8)]


def test_structure_reference_groups_the_bending_pairs_of_a_square_section(tmp_path):
    r = run_p5.structure_reference("sq_test", {"b": 0.10, "h": 0.10}, ladder=SMALL_LADDER,
                                   n_modes=6, outdir=str(tmp_path))
    assert r["clusters"][0] == (1, 2)                     # 굽힘 1차 쌍
    first = r["modes"][0]
    assert first["cluster"] == [1, 2] and first["character"].startswith("pair")
    assert first["shape_cert"] >= 0.99                    # 부분공간 인증은 회전에 불변
    assert os.path.exists(os.path.join(str(tmp_path), "p5_reference_sq_test.csv"))


def test_structure_reference_of_the_baseline_block_has_only_singletons(tmp_path):
    r = run_p5.structure_reference("base_test", {}, ladder=SMALL_LADDER, n_modes=6,
                                   outdir=str(tmp_path))
    assert all(len(c) == 1 for c in r["clusters"])
    assert [m["dominant"] for m in r["modes"]][:3] == ["z", "y", "z"]


def test_a_tied_square_section_torsion_is_called_torsion():
    """정사각 단면의 비틀림은 u_y와 u_z가 대칭상 같은 몫이라 지배 성분이 argmax의 잡음이다.
    동률이면 두 성분을 함께 지배로 두고 u_z의 폭 방향 대칭으로 가른다."""
    U = _synthetic("torsion")
    got = run_p5.mode_character(U, [0.001, 0.4995, 0.4995 + 1e-13])
    assert got["dominant"] == "yz" and got["character"] == "torsion"


def test_cluster_character_is_invariant_to_rotation_within_the_pair():
    """쌍의 성격은 쌍 안의 회전과 무관해야 한다. 성분 비율은 쌍 전체로 모은 값을 준다."""
    zb, yb = _synthetic("bending"), _synthetic("lateral")
    zb, yb = zb / np.linalg.norm(zb), yb / np.linalg.norm(yb)
    frac = [0.0, 0.5, 0.5]
    base = run_p5.mode_character(np.stack([zb, yb]), frac)
    t = 0.7
    rot = np.stack([np.cos(t) * zb + np.sin(t) * yb, -np.sin(t) * zb + np.cos(t) * yb])
    got = run_p5.mode_character(rot, frac)
    assert base["character"] == got["character"] == "bending"
    assert got["width_symmetry"] == pytest.approx(base["width_symmetry"], abs=1e-12)


def test_committed_atlas_labels_survive_the_tie_rule():
    """동률 규칙이 커밋된 세 구조의 라벨을 바꾸지 않아야 한다(최소 1·2위 차 0.60)."""
    p = os.path.join(DATA, "p5_mode_character.json")
    if not os.path.exists(p):
        pytest.skip("p5_mode_character.json이 없다")
    with open(p, encoding="utf-8") as f:
        atlas = json.load(f)
    for s in atlas["structures"].values():
        for m in s["modes"]:
            f3 = [m["fx"], m["fy"], m["fz"]]
            dom = run_p5.mode_character(_synthetic("bending"), f3)["dominant"]
            assert dom == m["dominant"]


L_GEO = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
L_SMALL_LADDER = [(10, 2, 1), (20, 4, 2), (40, 8, 4)]


def test_structure_reference_runs_on_the_l_bracket(tmp_path):
    """E21의 참조도 같은 함수로 만든다 — 다중 블록 메시·블록 합성 구적·블록 점 평가가 모두 지난다."""
    r = run_p5.structure_reference("lb_test", L_GEO, ladder=L_SMALL_LADDER, n_modes=6,
                                   cert_grid=(16, 8, 6), outdir=str(tmp_path))
    lam = [m["lam_star"] for m in r["modes"]]
    assert all(np.isfinite(lam)) and lam == sorted(lam) and lam[0] > 0
    # 가장 거친 단(두께 방향 요소 1개)은 비틀림(모드 6)을 아직 못 풀어 첫 쌍이 0.979다. 블록 점
    # 평가가 맞다면 인증은 세밀화와 함께 1로 다가간다 — 잘못 찾은 점이 있으면 수렴하지 않는다.
    for c in r["cluster_rows"]:
        assert c["per_pair"][-1] >= 0.999 and c["per_pair"][-1] >= c["per_pair"][0] - 1e-6
    assert os.path.exists(os.path.join(str(tmp_path), "p5_reference_lb_test.csv"))


def test_structure_reference_records_its_launch_provenance(tmp_path):
    """참조는 판정의 입력이다. 컨테이너의 git은 저장소 소유자가 달라 status를 거부하고, 그러면
    build()의 git_dirty가 늘 참으로 찍힌다(E16·E18 참조). 시작 시점의 출처를 함께 싣는다."""
    run_p5.structure_reference("lp_test", {}, ladder=SMALL_LADDER, n_modes=6,
                               outdir=str(tmp_path))
    man = json.load(open(os.path.join(str(tmp_path), "manifest_p5_reference_lp_test.json"),
                         encoding="utf-8"))
    assert {"git_sha", "git_dirty", "timestamp_utc"} <= set(man["launch"])


def test_l_ladder_keeps_cubic_elements_and_matching_block_faces():
    """E21 사다리: 블록 절점이 맞으려면 Δx = Δy(n_x = 5 n_y, L/b = 5), 요소는 정육면체(n_z = n_y/2)."""
    g = L_GEO
    for nx, ny, nz in run_p5.L_LADDER:
        assert nx * g["b"] == pytest.approx(ny * g["L"])
        assert ny * g["h"] == pytest.approx(nz * g["b"])
    assert [lv[1] for lv in run_p5.L_LADDER] == [4, 8, 16]


BI_GEO = {"kind": "bilayer", "L": 1.0, "b": 0.2, "h": 0.1, "E": 1.0, "E_top": 0.1,
          "rho": 1.0, "nu": 0.29}


def test_structure_reference_runs_on_the_bilayer(tmp_path):
    """E22의 참조도 같은 함수로 만든다 — 요소별 재료·층별 합성 구적이 함께 지난다."""
    r = run_p5.structure_reference("bi_test", BI_GEO, ladder=SMALL_LADDER, n_modes=6,
                                   cert_grid=(16, 8, 4), outdir=str(tmp_path))
    lam = [m["lam_star"] for m in r["modes"]]
    assert all(np.isfinite(lam)) and lam == sorted(lam) and lam[0] > 0
    assert all(len(c) == 1 for c in r["clusters"])
    assert r["worst_shape_cert"] >= 0.99
