"""P5 v2 드라이버 — 채점 순서와 게이트.

v1의 두 결함을 설계로 닫은 규칙을 회귀로 고정한다: 구적을 착취한 셀은 4분류 이전에
배제하고(§2.5), 수렴 게이트는 감사 격자 Λ의 꼬리 **최대** 변화를 읽는다(§3.1).
순수 함수만 검사하므로 GPU가 필요 없다.
"""
import pytest

from eigen_benchmark.drivers import run_p5_neural as run     # noqa: E402

# ── v2 §2.5 Rayleigh 하한 게이트 · §3.1 재보정 게이트 ────────────────────────

def test_rayleigh_lower_bound_gate_precedes_the_four_way_classification():
    """허용함수의 Rayleigh 몫은 λ₁보다 작을 수 없다. 작으면 그 셀의 Λ는 함수의
    성질이 아니라 구적 인공물이므로, 4분류에 **들어가기 전에** 배제해야 한다.
    4분류 안에서 처리하면 'non_converged'가 두 가지 다른 원인을 섞어 담는다."""
    L1 = 0.0016755365495313408
    assert run.classify_v2(lam=L1 * 0.5, lam1=L1, matched=1, mode=1,
                           mac=0.99, e_lam=0.001, conv=True) == "invalid_quadrature"
    # 하한을 아주 살짝 밑도는 것은 참조 불확실도이지 병리가 아니다 — tol이 흡수한다
    assert run.classify_v2(lam=L1 * (1 - 1e-9), lam1=L1, matched=1, mode=1,
                           mac=0.99, e_lam=0.001, conv=True) == "correct"
    # 하한 위에서는 기존 4분류가 그대로 작동한다
    assert run.classify_v2(lam=L1, lam1=L1, matched=1, mode=1, mac=0.99,
                           e_lam=0.001, conv=False) == "non_converged"
    assert run.classify_v2(lam=L1 * 3, lam1=L1, matched=1, mode=3, mac=0.99,
                           e_lam=0.001, conv=True) == "lower_mode_basin"
    assert run.classify_v2(lam=L1 * 3, lam1=L1, matched=3, mode=3, mac=0.1,
                           e_lam=0.9, conv=True) == "spurious"


def test_the_v2_gate_reads_the_worst_step_in_the_tail_not_just_the_endpoints():
    """v1 게이트는 꼬리의 **양 끝 두 점**만 봤다. 중간에 크게 튀었다가 돌아오면
    수렴으로 통과한다. v2는 꼬리 구간의 연속 스냅샷 사이 최대 상대변화를 본다."""
    import numpy as np
    # 꼬리 10 %가 3점이 되도록 30 스냅샷 — 2점이면 v1과 v2가 같아져 차이를 못 본다
    steady = np.ones((30, 1))
    steady[-2, 0], steady[-1, 0] = 1.0000001, 1.0000002
    assert run.converged_v2(steady, 0, tau=1e-5) is True

    spiky = np.ones((30, 1))
    spiky[-2, 0] = 2.0                       # 꼬리 한가운데서 튀었다가 돌아온다
    assert run.tail_variation(spiky, 0) > 0.5
    assert run.converged_v2(spiky, 0, tau=1e-5) is False
    # v1 게이트는 이것을 수렴으로 통과시킨다 — 양 끝만 보기 때문이다
    assert run._converged(spiky, 0) is True


def test_tau_calibration_uses_disjoint_seeds_and_the_ninety_percent_rule():
    """τ는 본실험 시드와 겹치지 않는 전용 시드에서, 90분위를 1·2·5×10^k로 올림해
    정한다. 본실험 결과를 보고 정하면 사전규정이 아니다."""
    import numpy as np
    vals = np.array([1e-4] * 9 + [3.1e-3] * 2)   # n=11이면 90분위가 원소에 정확히 닿는다
    assert float(np.percentile(vals, 90)) == pytest.approx(3.1e-3)
    assert run.tau_from_calibration(vals) == 5e-3
    assert run.tau_from_calibration(np.array([1.1e-6] * 10)) == 2e-6
    # 올림이지 반올림이 아니다 — 내림이면 보정 집합의 10 %가 이미 게이트를 못 넘는다
    assert run.tau_from_calibration(np.array([1.9e-3] * 10)) == 2e-3
    assert run.tau_from_calibration(np.array([2.1e-3] * 10)) == 5e-3
    assert set(run.TAU_SEEDS).isdisjoint(range(run.GRID_SEEDS_A))


# ── v2 드라이버 배선 (스모크) ────────────────────────────────────────────────

@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_tau_calibration_and_grid_v2_wire_end_to_end(tmp_path, monkeypatch):
    """기계검사만 한다 — 거친 quick 격자로는 어떤 인증도 성립하지 않는다(v1 B8).
    확인하는 것은 배선이다: τ가 파일로 남고, 본실험이 그 τ를 읽어 v2 스키마로 쓴다."""
    import json
    d = str(tmp_path)
    # 참조는 커밋된 산출물에서 복사한다 — 채점이 그것을 읽는다
    import shutil
    for f in ("p5_reference.csv",):
        shutil.copy(f"docs/_generated/data/paper4/{f}", d)

    cal = run.tau_calibration(outdir=d, quick=True, device="cpu")
    assert (tmp_path / "p5_tau_calibration.csv").exists()
    assert cal["tau"] > 0
    man = json.load(open(tmp_path / "manifest_p5_tau.json", encoding="utf-8"))
    assert man["seed_base"] == run.V2_TAU_SEED
    assert man["audit_grid"] == list(run.AUDIT_GRID)

    out = run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False,
                      device="cpu")
    assert out["tau"] == cal["tau"], "본실험이 보정된 τ를 읽지 않았다"
    rows = out["rows"]
    assert rows and all("invalid_quadrature" in r for r in rows)
    assert all(r["audit_grid"] == "30x15x12" for r in rows)
    # 분류 순서가 스키마에 남아야 사후에 규칙을 복원할 수 있다
    m2 = json.load(open(tmp_path / "manifest_p5_neural_v2.json", encoding="utf-8"))
    assert m2["classification_order"][0] == "invalid_quadrature"
    assert m2["tau"] == cal["tau"]


def test_v2_records_carry_the_dual_attribution_for_quadrature_failures():
    """§2.6 — 구적이 원인인 셀은 e_algebraic과 e_measurement를 **병기**한다.
    하나만 적으면 원고 §3.0의 귀속 규칙과 어긋난다."""
    assert run.DRIFT_FLAG == 0.05
    assert run.INVALID_ARM_LIMIT == 0.20
    assert run.OUTCOMES_V2[0] == "invalid_quadrature"
    assert len(run.OUTCOMES_V2) == 5


# ── 사후 실패양식 귀인 (2026-09-10, 사전규정 밖) ──────────────────────────────

def test_failure_form_splits_spurious_without_touching_the_prespecified_five():
    """사전규정의 5분류는 건드리지 않는다. `spurious` 하나만 세 갈래로 가른다 —
    형상을 못 알아본 셀(MAC 미달), 상위 모드를 잡은 셀, 목표 모드를 잡았는데
    고유값만 문턱을 넘은 셀. 앞의 둘과 마지막은 귀속이 다르다."""
    base = {"mode": 4, "matched_mode": 4, "mac": 0.999, "e_lam": 0.07}
    assert run.failure_form({**base, "outcome": "spurious"}) == "accuracy_shortfall"
    assert run.failure_form({**base, "outcome": "spurious",
                             "matched_mode": 6}) == "higher_mode"
    assert run.failure_form({**base, "outcome": "spurious",
                             "mac": 0.015}) == "unrecognizable"
    # MAC 미달이 먼저다 — 형상을 못 알아본 셀은 어느 모드를 잡았다고 말할 수 없다
    assert run.failure_form({**base, "outcome": "spurious", "matched_mode": 3,
                             "mac": 0.015}) == "unrecognizable"
    # 나머지 네 분류는 이름만 바꿔 그대로 통과시킨다
    assert run.failure_form({**base, "outcome": "correct"}) == "correct"
    assert run.failure_form({**base, "outcome": "lower_mode_basin"}) == "basin"
    assert run.failure_form({**base, "outcome": "non_converged"}) == "non_converged"


def test_threshold_sensitivity_reproduces_the_prespecified_count_at_its_own_threshold():
    """민감도 분석은 사전규정을 대체하지 않는다. 문턱을 0.05로 두면 커밋된
    `correct` 수와 **정확히** 일치해야 한다 — 일치하지 않으면 재계산 경로가
    사전규정과 다른 규칙을 쓰고 있다는 뜻이다."""
    recs = run.read_v2_records()
    for arm in {r["arm"] for r in recs}:
        for cond in {r["condition"] for r in recs if r["arm"] == arm}:
            sub = [r for r in recs if r["arm"] == arm and r["condition"] == cond]
            committed = sum(1 for r in sub if r["outcome"] == "correct")
            assert sum(1 for r in sub if run.correct_at(r, 0.05)) == committed, \
                f"{arm}/{cond}: 재계산이 사전규정 채점과 어긋난다"


def test_the_wall_is_not_an_artifact_of_where_the_threshold_falls():
    """대각 규칙의 0/60은 문턱을 세 배로 늘려도 0/60이다. 벽이 채점 문턱의
    위치에서 나온 것이 아니라는 것 — 1차 결과가 서려면 이것이 성립해야 한다."""
    recs = [r for r in run.read_v2_records()
            if r["condition"] == "resampled" and r["mode"] >= 4]
    diag = [r for r in recs if "diagonal" in r["arm"]]
    assert len(diag) == 60
    for t in run.SENSITIVITY_THRESHOLDS:
        assert sum(1 for r in diag if run.correct_at(r, t)) == 0
    # 반면 정확 사영은 문턱에 민감하다 — 모드 4의 실패가 문턱 바로 위에 몰려 있다
    qr = [r for r in recs if "qr" in r["arm"]]
    assert sum(1 for r in qr if run.correct_at(r, 0.05)) == 23
    assert sum(1 for r in qr if run.correct_at(r, 0.10)) == 34


def test_attribution_tables_cover_every_committed_cell():
    """분류표의 각 행은 그 (arm, 조건, 모드)의 20셀을 남김없이 담아야 한다.
    합이 n과 다르면 어떤 셀이 어느 칸에도 들어가지 않았다는 뜻이다."""
    rows = run.failure_attribution_rows(run.read_v2_records())
    assert rows
    for r in rows:
        assert sum(r[k] for k in run.FAILURE_FORMS) + r["correct"] == r["n"], r
    # 커밋된 사실 두 개를 회귀로 고정한다
    d46 = [r for r in rows if "diagonal" in r["arm"]
           and r["condition"] == "resampled" and r["mode"] >= 4]
    assert sum(r["basin"] for r in d46) == 58
    q4 = next(r for r in rows if "qr" in r["arm"]
              and r["condition"] == "resampled" and r["mode"] == 4)
    assert q4["accuracy_shortfall"] == 10
    assert q4["acc_above_reference"] == 10, "변분 상계이므로 전부 참조 위여야 한다"


# ── E2·E3 배선 (사전규정 2026-09-10) ────────────────────────────────────────

def test_e3_audit_grid_keeps_the_margin_over_its_own_training_family():
    """E3은 훈련 격자족을 감사 해상도까지 올린다. 그러면 본실험 감사 격자
    (30,15,12)를 훈련이 **넘어서므로** 감사 격자도 함께 올려야 한다 — 안 그러면
    v2 §2.2가 세운 훈련·감사 분리가 사라지고 E3이 재는 것이 무엇인지 알 수 없다."""
    from eigen_benchmark.problems import p5_block as p5
    mx = tuple(max(a) for a in run.E3_FAMILY_AXES)
    assert mx == (32, 16, 13)
    assert mx[0] > run.AUDIT_GRID[0], "격자족이 본실험 감사 격자를 넘지 않으면 E3이 아니다"
    for axis, m in zip(run.E3_AUDIT_GRID, mx):
        assert axis >= p5.AUDIT_AXIS_MARGIN * m
    # 격자 수는 본실험과 같아야 재추첨 통계가 비교된다
    assert len(run.E3_FAMILY_AXES[0]) * len(run.E3_FAMILY_AXES[1]) \
        * len(run.E3_FAMILY_AXES[2]) == 45


def test_grid_v2_budget_and_tag_default_to_the_committed_main_experiment():
    """E2가 바꾸는 것은 `iters` 하나뿐이고, 기본값은 본실험 그대로여야 한다.
    태그도 기본이 v2라야 커밋된 산출물 이름이 유지된다."""
    import inspect
    sig = inspect.signature(run.grid_v2)
    assert sig.parameters["iters"].default is None
    assert sig.parameters["tag"].default == "v2"
    assert sig.parameters["audit_grid"].default is None
    assert sig.parameters["family_axes"].default is None
    assert run.GRID_ITERS == 20000


def test_e3_check_grid_stays_finer_than_its_own_audit_grid():
    """드리프트 계기는 감사 격자보다 **촘촘해야** 한다. 본실험의 (40,20,15)를 E3에
    그대로 쓰면 감사 격자 (48,24,20)보다 거칠어져, 드리프트가 감사 격자의 부족이
    아니라 계기 자신의 부족을 재게 된다."""
    for c, a in zip(run.E3_CHECK_GRID, run.E3_AUDIT_GRID):
        assert c > a
    for c, a in zip(run.V2_CHECK_GRID, run.AUDIT_GRID):
        assert c > a          # 본실험도 같은 관계였다
    assert run.V2_CHECK_GRID[0] < run.E3_AUDIT_GRID[0], \
        "본실험 검사격자를 E3에 그대로 쓸 수 없는 이유가 이것이다"


# ── E2·E3 판정 (결과가 나오기 전에 박는다) ──────────────────────────────────

def test_e2_verdict_applies_the_two_prespecified_rules_separately():
    """E2의 규칙은 둘이고 서로 독립이다. (1) 벽: diagonal 모드 4–6 correct가 0/60이면
    예산에 견고하다. (2) 근접실패: QR 모드 4의 **e_λ 중앙값**이 1차 판별자이고
    개수는 보조다. 결과를 보고 규칙을 고르지 않도록 코드로 박아 둔다."""
    def cell(arm, mode, outcome, e_lam):
        return {"arm": f"p5_neural_3d[{arm} projection]", "condition": "resampled",
                "mode": mode, "outcome": outcome, "e_lam": e_lam}
    # 벽이 유지되는 경우
    recs = [cell("diagonal", m, "lower_mode_basin", 0.9) for m in (4, 5, 6)] * 20
    v = run.e2_verdict(recs, arm="diagonal")
    assert v["correct_4_6"] == 0 and v["wall_robust"] is True
    # 한 셀이라도 correct면 벽 주장은 조건부가 된다
    recs[0] = cell("diagonal", 4, "correct", 0.01)
    assert run.e2_verdict(recs, arm="diagonal")["wall_robust"] is False

    # 근접실패: 중앙값이 0.05 아래로 내려가면 e_optimization
    qr = [cell("qr", 4, "correct", 0.04) for _ in range(11)] \
        + [cell("qr", 4, "spurious", 0.06) for _ in range(9)]
    v2 = run.e2_verdict(qr, arm="qr")
    assert v2["mode4_e_lam_median"] < 0.05
    assert v2["shortfall"] == "e_optimization"
    # 20,000회 수준(0.058)에 머물면 예산이 원인이 아니다
    qr_flat = [cell("qr", 4, "spurious", 0.06) for _ in range(20)]
    assert run.e2_verdict(qr_flat, arm="qr")["shortfall"] == "not_budget"
    # 그 사이는 판정을 미룬다 — 사전규정이 그렇게 정했다
    qr_mid = [cell("qr", 4, "spurious", 0.055) for _ in range(20)]
    assert run.e2_verdict(qr_mid, arm="qr")["shortfall"] == "indeterminate"


def test_e3_verdict_compares_against_both_committed_wilson_intervals():
    """E3은 셋 중 하나로 떨어진다 — 대조군(45/60) 구간 안이면 해상도, 재표집(16/60)
    구간 안이면 격자 적응, 둘 다 아니면 분리 불가 유지. 본실험 수치는 커밋된
    기록에서 읽어 하드코딩하지 않는다."""
    base = run.e3_reference_counts()
    assert base["control"] == 45 and base["resampled"] == 16 and base["n"] == 60
    assert run.e3_verdict(45)["reading"] == "resolution"
    assert run.e3_verdict(16)["reading"] == "grid_adaptation"
    assert run.e3_verdict(30)["reading"] == "both"
    # 구간이 겹치는 값은 한쪽으로 단정하지 않는다
    for k in (45, 16, 30):
        v = run.e3_verdict(k)
        assert set(v) >= {"correct_4_6", "wilson", "reading"}


# ── 2026-09-12 정전 대응: 모드 단위 곁가지 저장 ──────────────────────────────

def _fake_mode(mode: int, lam, *, n_q: int = 4):
    """솔버가 모드 하나를 끝냈을 때 넘기는 짐 — 채점에 필요한 것만 담는다."""
    import numpy as np
    n = len(lam)
    return {"mode": mode, "arm": "p5_neural_3d[exact projection]",
            "lam": np.asarray(lam, dtype=float),
            "lam_train": np.asarray(lam, dtype=float),
            "hist": np.tile(np.asarray(lam, dtype=float), (5, 1)),
            "shapes": np.ones((n, 3, n_q)),
            "proj_mass": np.ones(n), "seconds": 1.5,
            "span_diagnostics": None, "n_seeds": n, "iters": 20,
            "w": np.full(n_q, 0.25)}


def _writer_kit(d):
    import numpy as np
    ref = {1: {"lam_star": 1.0, "rel_residual": 0.0},
           2: {"lam_star": 2.0, "rel_residual": 0.0}}
    shapes_ref = [np.ones((3, 4)), -np.ones((3, 4))]
    return dict(outdir=d, tag="ckpt", condition="resampled", ref=ref, tau=5.0,
                shapes_ref=shapes_ref, audit=(30, 15, 12), width=16, depth=2)


def test_mode_writer_grows_the_partial_file_one_mode_at_a_time():
    """arm이 끝나야 저장하는 규칙은 21–48시간짜리 실행에 너무 성기다. 2026-09-12
    정전으로 E2 (b′) 24.2시간과 E3 53.3시간이 **산출물 0바이트**로 끝났다."""
    import csv
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        w = run.make_mode_writer(**_writer_kit(d))
        w(_fake_mode(1, [1.0, 1.0]))
        rows = list(csv.DictReader(open(f"{d}/p5_neural_ckpt_partial.csv")))
        assert [r["mode"] for r in rows] == ["1"]
        # 정전으로 여기서 끊겨도 그 모드의 채점 결과는 읽을 수 있어야 한다
        assert {"correct", "invalid_quadrature", "lower_mode_basin", "spurious",
                "non_converged", "p_correct"} <= set(rows[0])
        prog = json.load(open(f"{d}/p5_ckpt_progress.json", encoding="utf-8"))
        assert prog["modes_done"] == 1 and prog["complete"] is False

        w(_fake_mode(2, [2.0, 2.0]))
        rows = list(csv.DictReader(open(f"{d}/p5_neural_ckpt_partial.csv")))
        assert [r["mode"] for r in rows] == ["1", "2"]
        recs = [json.loads(x) for x in
                open(f"{d}/p5_neural_ckpt_partial_records.jsonl", encoding="utf-8")]
        assert len(recs) == 4, "시드별 기록도 모드마다 쌓여야 한다"
        assert json.load(open(f"{d}/p5_ckpt_progress.json",
                              encoding="utf-8"))["modes_done"] == 2


def test_mode_writer_never_lets_a_write_error_kill_the_run(monkeypatch, capsys):
    """곁가지가 21시간짜리 계산을 죽이면 본말이 뒤집힌다 — 컨테이너가 root로 쓰는
    디렉터리라 권한·디스크에서 터질 여지가 실제로 있다."""
    import tempfile

    def boom(*a, **k):
        raise OSError("디스크가 찼다")

    with tempfile.TemporaryDirectory() as d:
        w = run.make_mode_writer(**_writer_kit(d))
        monkeypatch.setattr(run.manifest, "write_csv", boom)
        w(_fake_mode(1, [1.0, 1.0]))          # 예외가 새 나가면 이 줄에서 실패한다
    assert "중간저장 실패" in capsys.readouterr().out


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_partial_rows_agree_with_the_final_rows_on_the_shared_columns(tmp_path):
    """곁가지는 정규 산출물과 **같은 규칙**으로 채점해야 한다. 최종 경로를 일부러
    건드리지 않았으므로(채점이 두 곳에 있다) 그 일치는 이 테스트가 지킨다."""
    import csv
    import json
    import shutil

    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
    json.dump({"tau": 5.0}, open(f"{d}/manifest_p5_tau.json", "w"))

    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False,
                device="cpu", tag="agree")

    final = list(csv.DictReader(open(f"{d}/p5_neural_agree.csv")))
    part = list(csv.DictReader(open(f"{d}/p5_neural_agree_partial.csv")))
    assert len(part) == len(final) > 0
    shared = set(final[0]) & set(part[0])
    assert {"correct", "p_correct", "mode", "tau"} <= shared
    for f, p in zip(final, part):
        for k in sorted(shared):
            assert f[k] == p[k], f"모드 {f['mode']} 열 {k}: 최종 {f[k]} vs 곁가지 {p[k]}"
    assert json.load(open(f"{d}/p5_agree_progress.json",
                          encoding="utf-8"))["complete"] is True


# ── E4 (2026-09-14): 감사 구적을 두 배로 ─────────────────────────────────────

def test_e4_doubles_both_quadratures_and_keeps_the_training_family_fixed():
    """E4는 E2 (b′)에서 **계기만** 바꾼다. 훈련 격자족이 함께 움직이면 최적화까지
    달라져 대조가 성립하지 않는다 — 그래서 격자족은 본실험 기본값 그대로다."""
    import inspect
    assert run.E4_AUDIT_GRID == tuple(2 * v for v in run.AUDIT_GRID)
    assert run.E4_CHECK_GRID == tuple(2 * v for v in run.V2_CHECK_GRID)
    # 훈련 격자족을 바꾸는 인자를 E4는 주지 않는다 — 기본값이 본실험이어야 한다
    assert inspect.signature(run.grid_v2).parameters["family_axes"].default is None


def test_e4_check_grid_stays_finer_than_its_own_audit_grid():
    """드리프트 계기는 자기 감사 격자보다 촘촘해야 한다. E4의 감사 격자는 본실험
    검사격자 (40,20,15)보다도 촘촘하므로 검사격자를 그대로 물려쓸 수 없다."""
    for c, a in zip(run.E4_CHECK_GRID, run.E4_AUDIT_GRID):
        assert c > a
    assert run.E4_AUDIT_GRID[0] > run.V2_CHECK_GRID[0], \
        "본실험 검사격자를 E4에 그대로 쓸 수 없는 이유가 이것이다"


def test_e4_audit_grid_clears_the_margin_over_the_unchanged_training_family():
    """E4는 격자족을 올리지 않으므로 여유가 v2 규칙(1.5배)보다 커진다 — 3배다.
    사전규정이 인증을 (40,20,16) 기준으로 돌린 것은 후보를 하나로 고정하기 위해서였고,
    실효 여유는 실제 격자족 최대 (20,10,8)에 대해 재야 한다."""
    from eigen_benchmark.problems import p5_block as p5
    mx = (20, 10, 8)                      # 본실험 격자족 최대(v2 §2.2)
    for axis, m in zip(run.E4_AUDIT_GRID, mx):
        assert axis >= 3.0 * m
        assert axis >= p5.AUDIT_AXIS_MARGIN * m


# ── 구적 사다리 수렴 계기 (2026-09-16, E4의 교훈) ───────────────────────────

def test_ladder_tail_spread_separates_a_resolved_cell_from_an_unresolved_one():
    """E4가 보인 것 — 드리프트(두 격자의 **차이**)는 수렴 여부를 재지 못한다.
    발산하는 셀에서도 이웃 두 값이 우연히 가까울 수 있다(E4 시드 2: 125 → 141).
    가장 촘촘한 차수들에서 값이 아직 움직이는지를 봐야 한다.

    두 수열 다 E4·E2의 실측 Λ다(같은 망을 네 차수에서 잰 것)."""
    resolved = [0.1463534, 0.1463501, 0.1463378, 0.1463365]     # 시드 6
    unresolved = [1.9310, 7.6574, 125.39, 140.61]               # 시드 2
    assert run.ladder_tail_spread(resolved) < 1e-3
    assert run.ladder_tail_spread(unresolved) > 0.9


def test_ladder_tail_spread_does_not_manufacture_noise_on_a_settled_cell():
    """**차이의 비를 쓰면 여기서 틀린다.** 이미 기계정밀도까지 수렴한 셀은 차이가
    잡음이고 그 비도 잡음이다 — 실제로 E4 시드 0(네 차수 값이 일곱 자리까지 동일)에서
    수축비가 3.39로 나와 미수렴으로 읽혔다. 폭은 그 함정이 없다."""
    assert run.ladder_tail_spread([0.1459325, 0.1459325, 0.1459325, 0.1459325]) == 0.0


def test_ladder_tail_spread_needs_at_least_three_rungs():
    """두 점으로는 수렴을 말할 수 없다 — 그것이 드리프트가 못 한 일이다."""
    import math
    assert math.isnan(run.ladder_tail_spread([1.0, 2.0]))


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_grid_v2_records_the_ladder_and_its_tail_spread_per_cell(tmp_path):
    """셀마다 수축비가 남아야 다음 사전규정이 그것을 판별자로 쓸 수 있다.
    **계측만 한다** — 분류는 건드리지 않으므로 커밋된 채점과 비교가 유지된다."""
    import csv
    import json
    import shutil

    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
    json.dump({"tau": 5.0}, open(f"{d}/manifest_p5_tau.json", "w"))

    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False, device="cpu",
                tag="lad", check_grids=[(40, 20, 15), (50, 25, 19)])

    recs = [json.loads(x) for x in open(f"{d}/p5_neural_lad_records.jsonl",
                                        encoding="utf-8")]
    assert recs, "기록이 비었다"
    for r in recs:
        assert len(r["lam_ladder"]) == 3, "감사 격자 + 두 칸이어야 한다"
        assert r["lam_ladder"][0] == r["Lam"], "0번 칸은 채점이 쓴 Λ 그대로여야 한다"
        assert "ladder_tail_spread" in r
        # 드리프트는 사다리의 첫 칸에서 오던 그대로여야 한다
        assert r["Lam_check"] == r["lam_ladder"][1]
    rows = list(csv.DictReader(open(f"{d}/p5_neural_lad.csv")))
    assert "ladder_spread_median" in rows[0]


# ── E5 (사전규정 2026-09-16) ────────────────────────────────────────────────

def test_e5_ladder_starts_at_e3s_drift_instrument():
    """E5는 E3와 **훈련 격자족 하나만** 달라야 한다. 사다리 첫 칸이 E3의 검사격자와
    같아야 `quadrature_drift`가 같은 계기가 되고, 그래야 두 실행이 비교된다."""
    assert run.E5_LADDER[0] == run.E3_CHECK_GRID
    for a, b in zip(run.E5_LADDER, run.E5_LADDER[1:]):
        assert all(y > x for x, y in zip(a, b)), "사다리는 축마다 촘촘해져야 한다"
    for axis, a in zip(run.E5_LADDER[0], run.E3_AUDIT_GRID):
        assert axis > a, "첫 칸도 감사 격자보다 촘촘해야 한다"
    assert len(run.E5_LADDER) >= 3, "수렴을 보려면 감사 격자까지 네 칸이 필요하다"


def test_ladder_resolved_threshold_sits_in_the_measured_gap():
    """문턱은 임의값이 아니다. E2+E4의 실데이터 20셀에서 수렴 셀의 최대 꼬리폭은
    9.4e-05, 미수렴 셀의 최소는 5.3e-02다. 문턱은 그 사이에 있어야 한다."""
    assert run.LADDER_RESOLVED == 1e-3
    assert 9.4e-05 < run.LADDER_RESOLVED < 5.3e-02


# ── E6 (사전규정 2026-09-17) ────────────────────────────────────────────────

def test_e6_ladder_matches_the_four_grids_already_held_at_the_larger_budget():
    """E6의 네 칸은 40,000회 쪽에서 조립한 네 격자와 **같아야** 한다 — 감사 격자를
    0번 칸으로 두면 (30,15,12) → (40,20,15) → (60,30,24) → (80,40,30)이고, 이는
    E2의 감사·검사와 E4의 감사·검사다. 다르면 두 예산의 꼬리폭을 견줄 수 없다."""
    assert run.E6_LADDER == (run.V2_CHECK_GRID, run.E4_AUDIT_GRID, run.E4_CHECK_GRID)
    ladder = (run.AUDIT_GRID,) + run.E6_LADDER
    for a, b in zip(ladder, ladder[1:]):
        assert all(y > x for x, y in zip(a, b)), "사다리는 축마다 촘촘해져야 한다"
    assert len(ladder) >= 3, "꼬리폭은 세 칸 이상을 요구한다"


def test_e6_changes_nothing_the_committed_main_experiment_fixed():
    """E6은 커밋된 본실험에 **계측만** 더한다 — 감사 격자·예산·격자족이 기본값이어야
    그 실행의 확장이지 별개 실행이 아니다.

    이 테스트는 새 코드를 끌어내지 않는다(오늘 기준 이미 통과한다). E6의 설계가 기대는
    기본값들이 나중에 바뀌면 깨지도록 박아 두는 **가드**다."""
    import inspect
    sig = inspect.signature(run.grid_v2)
    assert sig.parameters["iters"].default is None          # 기본값이 본실험 예산
    assert sig.parameters["audit_grid"].default is None     # 기본값이 AUDIT_GRID
    assert sig.parameters["family_axes"].default is None    # 기본값이 본실험 격자족
    assert run.GRID_ITERS == 20000


def test_grid_v2_can_record_gram_components_and_save_parameters(tmp_path):
    """검토 반영(2026-09-20) — 어느 Gram 쌍이 큰지를 CSV에 남기고, 망을 저장해 다음
    재측정이 순전파 비용이 되게 한다. 둘 다 **기본 꺼짐**이라 커밋된 실행은 불변이다."""
    import csv
    import json
    import shutil

    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
    json.dump({"tau": 5.0}, open(f"{d}/manifest_p5_tau.json", "w"))
    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False, device="cpu",
                tag="gr", gram_offdiag=True, save_params=True)
    # 망 저장 — 모드마다 층 파라미터가 시드 축을 앞에 두고 들어간다
    import numpy as np
    z = np.load(f"{d}/p5_params_gr.npz")
    assert len(z.files) > 0 and all(v.ndim >= 1 for v in z.values())
    assert any("_m1_" in f for f in z.files) and any("_m2_" in f for f in z.files)
    # `absc_` 열은 여기서 볼 수 없다 — quick 모드는 n_modes = 2라 저장 스팬이 k = 1까지고
    # 1×1 그람에는 비대각 쌍이 아예 없다. 그 경로는 솔버 수준 테스트가 n_modes = 3으로
    # 덮는다(`test_span_diagnostics_can_name_each_gram_offdiagonal`).
    diag = list(csv.DictReader(open(f"{d}/p5_span_diagnostics_gr.csv")))
    assert diag and int(diag[0]["k"]) == 1


def test_grid_v2_passes_input_normalisation_through_and_defaults_to_raw():
    """E8은 드라이버를 통해 돈다. 기본이 원좌표여야 커밋된 480셀이 불변이다."""
    import inspect
    assert inspect.signature(run.grid_v2).parameters["normalize_input"].default is False


# ── E10 (사전규정 2026-09-22): 에너지 MAC과 잠김 서명 ───────────────────────

def test_oracle_fit_records_the_energy_mac_not_just_the_displacement_one():
    """잠김 가설은 **변위는 맞고 변형률은 틀린** 셀을 예측한다. E1은 손실에 두 MAC을
    다 쓰면서 기록은 질량 MAC만 남겨 그 간극을 볼 수 없었다."""
    import csv
    import json
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
        run.oracle_fit(outdir=d, quick=True, device="cpu", tag="t")
        rows = list(csv.DictReader(open(f"{d}/p5_e1_oracle_fit_t.csv")))
        assert rows and "mac_energy" in rows[0], rows[0].keys()
        for r in rows:
            assert 0.0 <= float(r["mac_energy"]) <= 1.0 + 1e-12
        m = json.load(open(f"{d}/manifest_p5_e1_t.json", encoding="utf-8"))
        assert m["input_scaling"] == "raw"


def test_locking_signature_needs_both_halves():
    """사전규정 §3 — 질량 MAC ≥ 0.95 **그리고** 에너지 MAC < 0.5. 한쪽만으로 세면
    완전 붕괴 셀(질량 MAC 0.000)까지 잠김으로 잘못 센다."""
    assert run.locking_signature(mac_mass=0.96, mac_energy=0.10) is True
    assert run.locking_signature(mac_mass=0.000, mac_energy=0.00) is False   # 완전 붕괴
    assert run.locking_signature(mac_mass=0.99, mac_energy=0.90) is False    # 둘 다 맞음
    assert run.locking_signature(mac_mass=0.95, mac_energy=0.50) is False    # 경계는 제외


# ── E13 (사전규정 2026-09-24): 기하·BC를 드라이버까지 배선 ───────────────────

def test_grid_v2_defaults_keep_the_committed_geometry_and_boundary():
    """E13은 기하와 경계조건을 바꾼다. **기본값이 커밋된 것이어야** 480셀이 불변이다."""
    import inspect
    sig = inspect.signature(run.grid_v2)
    assert sig.parameters["geometry"].default is None
    assert sig.parameters["bc"].default == "clamped_x0"
    assert sig.parameters["ref_name"].default == "p5_reference"


def test_grid_v2_reads_the_reference_named_for_the_structure(tmp_path):
    """구조마다 참조가 다르므로 채점이 그 구조의 참조를 읽어야 한다. 이름이 틀리면
    다른 구조의 Λ*로 채점하게 되고 그 오류는 조용하다."""
    import shutil
    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv",
                f"{d}/p5_reference_alt.csv")
    got = run._reference(d, name="p5_reference_alt")
    assert set(got) >= {1, 2, 3} and got[1]["lam_star"] > 0
    import pytest as _pt
    with _pt.raises(FileNotFoundError):
        run._reference(d, name="p5_reference")


# ── E12 (사전규정 2026-09-26): 고정 훈련 격자를 감사 격자와 분리한다 ────────

def test_fixed_train_grid_is_separate_from_the_audit_grid():
    """본실험 대조군은 **감사 격자 자체에서** 훈련해 held-out 교란을 만든다. E12는 그
    축을 제거하려고 훈련 격자를 따로 지정한다 — 기본값은 커밋된 동작(감사 격자)이어야
    480셀이 불변이다."""
    import inspect
    assert inspect.signature(run.grid_v2).parameters["fixed_train_grid"].default is None


def test_e12_fixed_grids_match_each_family_mean():
    """고정 조건의 훈련 격자는 대응하는 재표집 격자족의 **평균 점수와 같아야** 해상도
    축이 흐려지지 않는다. 두 값 모두 각 격자족의 중앙 격자다."""
    import numpy as np
    from eigen_benchmark.neural import p5_neural as p5n
    lo = np.mean([nx * ny * nz for nx in p5n.FAMILY_NX
                  for ny in p5n.FAMILY_NY for nz in p5n.FAMILY_NZ])
    hi = np.mean([nx * ny * nz for ax in [run.E3_FAMILY_AXES]
                  for nx in ax[0] for ny in ax[1] for nz in ax[2]])
    assert int(np.prod(run.E12_FIXED_LOW)) == int(lo) == 1134
    assert int(np.prod(run.E12_FIXED_HIGH)) == int(hi) == 5400
    # 그리고 감사 격자보다 성겨야 한다 — 아니면 held-out이 깨진다
    for g in (run.E12_FIXED_LOW, run.E12_FIXED_HIGH):
        assert all(a < b for a, b in zip(g, run.E3_AUDIT_GRID))


def test_empty_arms_means_no_resampled_arm_not_the_default_three():
    """`arms=()`는 "재표집 arm 없음"이어야 한다. 빈 튜플이 falsy라 기본 3 arm으로
    대체되면 대조군만 돌리려는 호출이 조용히 네 조건을 돈다 — E12에서 실제로 그랬고
    21시간을 잃었다. `None`만 기본값을 뜻한다."""
    import inspect
    src = inspect.getsource(run.grid_v2)
    assert "if arms else GRID_ARMS_A" not in src, \
        "빈 튜플과 None을 구별하지 않는다"
    assert "arms is None" in src, "기본값 판정은 None으로 해야 한다"


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_grid_v2_writes_per_seed_span_rows_when_asked(tmp_path):
    """시드 수준 전이 산점의 x축이 되는 자료다. 요약 CSV와 **별도 파일**로 내보내
    커밋된 스키마를 건드리지 않는다."""
    import csv
    import json
    import shutil
    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
    json.dump({"tau": 5.0}, open(f"{d}/manifest_p5_tau.json", "w"))
    run.grid_v2(outdir=d, quick=True, arms=("diagonal",), control=False,
                device="cpu", tag="ps", gram_offdiag=True, per_seed_span=True)
    rows = list(csv.DictReader(open(f"{d}/p5_span_per_seed_ps.csv")))
    assert rows, "시드별 행이 비었다"
    for r in rows:
        assert {"arm", "condition", "stage_mode", "k", "seed", "max_offdiag"} <= set(r)
    # quick은 2시드 × (모드 2의 k=1) = 2행
    assert len({r["seed"] for r in rows}) == 2


def test_per_seed_span_is_off_by_default():
    import inspect
    assert inspect.signature(run.grid_v2).parameters["per_seed_span"].default is False


# ── 축퇴쌍(고유공간) 판정 — 정사각 단면 구조의 전제 ─────────────────────────────
#
# 축퇴 고유공간 안의 어떤 회전도 똑같이 고유벡터다. 참조 FEM조차 쌍 안에서 임의로 회전해
# 나오므로(정사각 단면에서 y 0.500 / z 0.500), 모드별 MAC으로 정합하면 45° 혼합 출력이
# MAC 0.5로 떨어져 spurious로 오판된다. 쌍은 부분공간으로 정합하고, 같은 쌍에서 앞 단계가
# 이미 찾은 방향을 다시 찾은 출력은 유인으로 본다.

import numpy as np                                             # noqa: E402


def _field(rng, n=240):
    """(3, n) 무작위 벡터장과 양의 가중."""
    return rng.normal(size=(3, n)), rng.uniform(0.5, 1.5, size=n)


def _orthonormal_pair(rng, w, n=240):
    """질량 내적으로 서로 직교하는 두 장."""
    a, b = rng.normal(size=(3, n)), rng.normal(size=(3, n))
    ip = lambda x, y: float((w * (x * y).sum(0)).sum())
    a = a / np.sqrt(ip(a, a))
    b = b - ip(a, b) * a
    return a, b / np.sqrt(ip(b, b))


def test_gap_rule_groups_near_degenerate_modes_into_clusters():
    lam = [1.0, 1.0 + 1e-9, 2.0, 2.0 + 1e-12, 3.0, 3.5]
    assert run.clusters_from_gaps(lam) == [(1, 2), (3, 4), (5,), (6,)]
    assert run.clusters_from_gaps([1.0, 1.5, 2.0]) == [(1,), (2,), (3,)]


def test_singleton_clusters_reproduce_the_modewise_match_bitwise():
    """군집이 모두 1개짜리이면 기존 정합(모드별 MAC의 argmax)과 비트 단위로 같아야 한다 —
    커밋된 실행의 판정이 바뀌면 안 된다."""
    rng = np.random.default_rng(0)
    refs = [rng.normal(size=(3, 240)) for _ in range(4)]
    phi, w = _field(rng)
    old = [run._mass_mac(phi, R, w) for R in refs]
    j_old, mac_old = int(np.nanargmax(old)) + 1, float(old[int(np.nanargmax(old))])
    m = run.match_clustered(phi, refs, w, [(1,), (2,), (3,), (4,)], target=2)
    assert (m["matched_mode"], m["mac"]) == (j_old, mac_old)
    assert m["dup_mac"] is None


def test_a_rotated_member_of_a_degenerate_pair_matches_the_pair():
    """45° 혼합은 모드별 MAC이 0.5지만 쌍의 부분공간 MAC은 1이다."""
    rng = np.random.default_rng(1)
    phi_, w = _field(rng)
    a, b = _orthonormal_pair(rng, w)
    c = rng.normal(size=(3, 240))                   # 쌍 밖의 다른 모드
    out = (a + b) / np.sqrt(2.0)
    assert abs(run._mass_mac(out, a, w) - 0.5) < 1e-9
    m = run.match_clustered(out, [a, b, c], w, [(1, 2), (3,)], target=2)
    assert m["matched_mode"] == 2 and abs(m["mac"] - 1.0) < 1e-9


def test_refinding_an_earlier_direction_of_the_same_pair_is_a_basin():
    rng = np.random.default_rng(2)
    _, w = _field(rng)
    a, b = _orthonormal_pair(rng, w)
    first = np.cos(0.3) * a + np.sin(0.3) * b       # 1단계가 찾은 방향
    again = 1.7 * first                             # 2단계가 같은 방향을 다시 찾음
    m = run.match_clustered(again, [a, b], w, [(1, 2)], target=2, earlier={1: first})
    assert m["matched_mode"] == 1 and m["dup_mac"] > 0.999
    assert run.classify_v2(lam=1.0, lam1=1.0, matched=m["matched_mode"], mode=2,
                           mac=m["mac"], e_lam=0.0, conv=True) == "lower_mode_basin"
    # 직교 방향은 새 모드다
    new = -np.sin(0.3) * a + np.cos(0.3) * b
    m = run.match_clustered(new, [a, b], w, [(1, 2)], target=2, earlier={1: first})
    assert m["matched_mode"] == 2 and m["dup_mac"] < 1e-9


def test_falling_back_into_a_lower_pair_is_a_basin():
    rng = np.random.default_rng(3)
    _, w = _field(rng)
    a, b = _orthonormal_pair(rng, w)
    c = rng.normal(size=(3, 240))
    out = 0.6 * a - 0.8 * b                         # 3단계 출력이 쌍 1로 떨어짐
    m = run.match_clustered(out, [a, b, c], w, [(1, 2), (3,)], target=3)
    assert m["matched_mode"] == 1                   # 군집의 가장 낮은 번호
    assert run.classify_v2(lam=1.0, lam1=1.0, matched=m["matched_mode"], mode=3,
                           mac=m["mac"], e_lam=0.0, conv=True) == "lower_mode_basin"


# ── E16 정사각 단면: 구조별 참조 단계와 τ ────────────────────────────────────

def test_reference_shapes_are_solved_at_the_requested_level():
    """정사각 단면은 사다리가 달라 최상단도 다르다. 참조 형상이 기준 구조의 최상단
    (48, 16, 8)에서 풀리면 다른 메시의 형상으로 채점하게 된다."""
    import numpy as np
    from eigen_benchmark.problems import p5_block as p5
    g = {**p5.P5_GEOMETRY, "b": 0.10, "h": 0.10}
    got = run._reference_shapes(2, (6, 4, 4), geometry=g, level=(4, 2, 2),
                                start_vector="ones")
    r = p5.solve(4, 2, 2, n_modes=2, geometry=g, start_vector="ones")
    pts, _ = p5._tensor_grid((6, 4, 4), g)
    want = p5.eval_mode_at(r["mesh"], r["vec_full"][:, 0], pts).T
    assert np.allclose(got[0], want)


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_tau_calibration_for_a_new_structure_keeps_the_committed_tau(tmp_path):
    """E16은 구조가 바뀌므로 τ를 다시 정한다. 그 보정이 기준 구조의 τ 파일을 덮으면 이후의
    기준 구조 실행이 조용히 다른 τ로 게이트된다 — 태그 붙은 파일에 따로 쓴다."""
    import json
    from eigen_benchmark.problems import p5_block as p5
    g = {**p5.P5_GEOMETRY, "b": 0.10, "h": 0.10}
    cal = run.tau_calibration(outdir=str(tmp_path), quick=True, device="cpu", tag="sq",
                              geometry=g, audit_grid=(30, 14, 14),
                              family_axes=((16, 17), (7, 8), (7, 8)))
    assert not (tmp_path / "manifest_p5_tau.json").exists()
    assert (tmp_path / "p5_tau_calibration_sq.csv").exists()
    man = json.load(open(tmp_path / "manifest_p5_tau_sq.json", encoding="utf-8"))
    assert man["tau"] == cal["tau"] and man["audit_grid"] == [30, 14, 14]
    assert man["geometry"]["b"] == 0.10
    assert man["family_axes"] == [[16, 17], [7, 8], [7, 8]]


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_grid_v2_scores_a_new_structure_against_its_own_reference_level(tmp_path):
    import json
    from eigen_benchmark.problems import p5_block as p5
    g = {**p5.P5_GEOMETRY, "b": 0.10, "h": 0.10}
    with open(tmp_path / "p5_reference_sq.csv", "w", encoding="utf-8") as f:
        f.write("mode,lam_star,rel_residual\n")
        for m, lam in enumerate([0.0103, 0.0103, 0.40, 0.40, 0.81, 2.47, 2.6, 2.6], 1):
            f.write(f"{m},{lam},0.001\n")
    run.grid_v2(outdir=str(tmp_path), quick=True, arms=("exact",), control=False,
                device="cpu", tau=1.0, geometry=g, ref_name="p5_reference_sq",
                ref_level=(4, 2, 2), audit_grid=(30, 14, 14), check_grid=(40, 19, 19),
                family_axes=((16, 17), (7, 8), (7, 8)), pair_scoring=True, tag="sq")
    man = json.load(open(tmp_path / "manifest_p5_neural_sq.json", encoding="utf-8"))
    assert man["reference_level"] == [4, 2, 2]
    assert man["clusters"][0] == [1, 2]


def test_e16_settings_carry_the_prespecified_design():
    """사전규정 2026-09-30 §2의 값이 실행 인자에 그대로 실려야 한다 — 손으로 조립하면
    한 arm만 다른 격자로 도는 사고가 조용히 난다."""
    import math
    from eigen_benchmark.drivers import p5_stats
    tags = set()
    for arm in ("diagonal", "reorth_diagonal", "exact", "qr"):
        k = run.e16_settings(arm)
        tags.add(k["tag"])
        assert k["arms"] == (arm,) and k["control"] is False
        assert k["tag"] == p5_stats.E16_TAGS[arm]
        assert k["geometry"]["b"] == k["geometry"]["h"] == 0.10
        assert k["pair_scoring"] and k["save_params"] and k["per_seed_span"]
        assert k["ref_name"] == "p5_reference_e16_square" and k["ref_level"] == (48, 12, 12)
        fam_max = [max(a) for a in k["family_axes"]]
        # 감사 격자는 격자족 최대의 1.5배 올림(v2 §2.2), y·z 대칭
        assert k["audit_grid"] == tuple(math.ceil(1.5 * v) for v in fam_max) == (30, 14, 14)
        assert k["check_grids"][0] == k["check_grid"]
        assert all(g[1] == g[2] for g in (k["audit_grid"], *k["check_grids"]))
    assert len(tags) == 4
    with pytest.raises(ValueError):
        run.e16_settings("mystery")


def test_e16_reference_artifact_matches_the_settings():
    """참조는 설정과 같은 구조·최상단·감사 격자에서 인증되어야 한다."""
    import json
    import os
    from eigen_benchmark.problems import p5_block as p5
    p = "docs/_generated/data/paper4/p5_reference_e16_square.json"
    if not os.path.exists(p):
        pytest.skip("E16 참조가 없다 — run_p5.structure_reference를 먼저 돌린다")
    ref = json.load(open(p, encoding="utf-8"))
    assert run.E16_GEOMETRY == {**p5.P5_GEOMETRY, "b": 0.10, "h": 0.10} == ref["geometry"]
    assert tuple(ref["ladder"][-1]) == run.E16_REF_LEVEL
    assert tuple(ref["cert_grid"]) == run.E16_AUDIT_GRID
    assert ref["clusters"][:4] == [[1, 2], [3, 4], [5], [6]]
    assert ref["shape_certified"] and ref["clusters_consistent_across_levels"]


def test_e16_tau_calibrates_on_the_square_family_and_reports_progress(monkeypatch):
    """τ 보정은 4–6시간이다 — 진행 로그 없이 돌리면 멈춤과 진행을 구별할 수 없다."""
    import inspect
    got = {}
    monkeypatch.setattr(run, "tau_calibration", lambda **kw: got.update(kw) or {"tau": 1.0})
    run.e16_tau(outdir="x", device="cpu")
    assert got["tag"] == run.E16_TAU_TAG and got["audit_grid"] == run.E16_AUDIT_GRID
    assert got["family_axes"] == run.E16_FAMILY_AXES and got["geometry"] == run.E16_GEOMETRY
    assert got["heartbeat_every"] == run.HEARTBEAT_ITERS
    sig = inspect.signature(inspect.unwrap(run.grid_v2))
    assert sig.parameters["ref_level"].default is None


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_run_manifests_record_the_launch_commit_not_the_end_commit(tmp_path, monkeypatch):
    """실행 도중 커밋이 쌓이면 끝날 때의 HEAD는 실행한 코드가 아니다(E15-r이 그렇게 된다)."""
    import json
    import shutil
    from eigen_benchmark.drivers import manifest as mf
    calls = {"n": 0}

    def sha():
        calls["n"] += 1
        return "a" * 40 if calls["n"] == 1 else "b" * 40
    monkeypatch.setattr(mf, "_head_sha", sha)
    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False, device="cpu",
                tau=5.0, tag="lp")
    man = json.load(open(f"{d}/manifest_p5_neural_lp.json", encoding="utf-8"))
    assert man["launch"]["git_sha"] == "a" * 40 and man["git_sha"] == "b" * 40
    calls["n"] = 0
    run.tau_calibration(outdir=d, quick=True, device="cpu", tag="lp")
    man = json.load(open(f"{d}/manifest_p5_tau_lp.json", encoding="utf-8"))
    assert man["launch"]["git_sha"] == "a" * 40


def test_p5_drivers_read_and_write_paper4_by_default(monkeypatch, tmp_path):
    """논문2의 기본 출력 경로가 비어 있어도 P5 드라이버는 data/paper4에서 읽어야 한다."""
    import inspect
    from eigen_benchmark.drivers import manifest, p5_stats, run_p5
    from eigen_benchmark.render import figures_p5
    monkeypatch.setattr(manifest, "DEFAULT_OUTDIR", str(tmp_path))
    assert run.read_v2_records(), "data/paper4의 본실험 기록을 읽지 못했다"
    assert "manifest.DEFAULT_OUTDIR" not in inspect.getsource(run_p5.mode_atlas)
    for path in (p5_stats.DATA_DIR, figures_p5.DATA_DIR):
        assert path.replace("\\", "/").endswith("data/paper4")
    assert figures_p5.FIG_DIR.replace("\\", "/").endswith("fig/paper4")


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_grid_v2_can_redraw_the_gauss_family_at_any_period(tmp_path, monkeypatch):
    """E17은 가우스 격자족을 매 반복 다시 뽑는다(본실험은 250회마다). 기본값은 그대로다."""
    import json
    import shutil
    from eigen_benchmark.neural import p5_neural as p5n
    seen = []
    real = p5n.solve_p5_neural

    def spy(*a, **kw):
        seen.append(kw.get("resample_every"))
        return real(*a, **kw)
    monkeypatch.setattr(p5n, "solve_p5_neural", spy)
    d = str(tmp_path)
    shutil.copy("docs/_generated/data/paper4/p5_reference.csv", d)
    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False, device="cpu",
                tau=5.0, tag="r1", resample_every=1)
    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False, device="cpu",
                tau=5.0, tag="r250")
    assert seen == [1, run.V2_RESAMPLE_EVERY]
    m1 = json.load(open(f"{d}/manifest_p5_neural_r1.json", encoding="utf-8"))
    assert m1["resample_every"] == 1 and "every 1 iterations" in m1["train_grid"]


def test_followup_settings_carry_the_prespecified_design():
    """사전규정 2026-10-02 E17–E20의 일곱 실행. 설정을 손으로 조립하지 않는다."""
    names = ("e17_gauss1", "e17_mc250", "e18_diag", "e18_exact", "e19_budget", "e19_norm",
             "e20r_b4")
    assert tuple(run.FOLLOWUP_RUNS) == names
    got = {n: run.followup_settings(n) for n in names}
    for n, (k, tau_file) in got.items():
        assert k["tag"] == n and k["control"] is False and k["save_params"]
    base_tau, sq_tau = "manifest_p5_tau.json", f"manifest_p5_tau_{run.E16_TAU_TAG}.json"
    g1, m250 = got["e17_gauss1"][0], got["e17_mc250"][0]
    assert g1["arms"] == ("exact",) and g1["resample_every"] == 1 and "mc_points" not in g1
    assert m250["arms"] == ("exact",) and m250["mc_points"] == 1134 and m250["mc_resample_every"] == 250
    for n in ("e18_diag", "e18_exact"):
        k = got[n][0]
        assert k["geometry"]["b"] == 0.105 and k["geometry"]["h"] == 0.10
        assert k["ref_name"] == "p5_reference_e18_near" and got[n][1] == sq_tau
        assert k["audit_grid"] == run.E16_AUDIT_GRID and k["pair_scoring"]
    assert got["e18_diag"][0]["arms"] == ("diagonal",) and got["e18_exact"][0]["arms"] == ("exact",)
    assert got["e19_budget"][0]["iters"] == 40000 and got["e19_norm"][0]["normalize_input"]
    for n in ("e19_budget", "e19_norm"):
        assert got[n][0]["arms"] == ("exact",) and got[n][0]["geometry"] == run.E16_GEOMETRY
        assert got[n][1] == sq_tau
    e20 = got["e20r_b4"][0]
    assert e20["mc_points"] == 4 * 1134 and e20["storage"] == "orthogonalized"
    assert e20["arms"] == ("diagonal",) and e20["batch_diag"]
    for n in ("e17_gauss1", "e17_mc250", "e20r_b4"):
        assert got[n][1] == base_tau
    with pytest.raises(ValueError):
        run.followup_settings("e99")


def test_followup_run_refuses_without_its_tau(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.followup_run("e18_diag", outdir=str(tmp_path))


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_tau_calibration_records_the_l_bracket_geometry(tmp_path):
    """E21의 τ 보정 매니페스트는 기하의 종류 표지("kind": "L")를 문자열 그대로 남긴다."""
    import json
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    run.tau_calibration(outdir=str(tmp_path), quick=True, device="cpu", tag="lt",
                        geometry=g, audit_grid=(16, 8, 6),
                        family_axes=((8, 9), (4, 5), (3, 4)))
    man = json.load(open(tmp_path / "manifest_p5_tau_lt.json", encoding="utf-8"))
    assert man["geometry"]["kind"] == "L" and man["geometry"]["a"] == 0.5


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch 없음")
def test_grid_v2_runs_on_the_l_bracket(tmp_path):
    """신경 본실험 경로가 L-브래킷에서 끝까지 돈다(기계 검사)."""
    import json
    from eigen_benchmark.drivers import run_p5
    g = {"kind": "L", "L": 1.0, "b": 0.2, "h": 0.1, "a": 0.5, "E": 1.0, "rho": 1.0, "nu": 0.29}
    d = str(tmp_path)
    run_p5.structure_reference("lq", g, ladder=[(10, 2, 1), (20, 4, 2), (40, 8, 4)],
                               n_modes=6, cert_grid=(16, 8, 6), outdir=d)
    run.grid_v2(outdir=d, quick=True, arms=("exact",), control=False, device="cpu", tau=5.0,
                geometry=g, ref_name="p5_reference_lq", ref_level=(40, 8, 4),
                audit_grid=(16, 8, 6), check_grid=(20, 10, 8), tag="lq")
    man = json.load(open(f"{d}/manifest_p5_neural_lq.json", encoding="utf-8"))
    assert man["geometry"]["a"] == 0.5 and man["reference_level"] == [40, 8, 4]


def test_e21_settings_carry_the_prespecified_design():
    """사전규정 2026-10-04 E21의 값이 실행 인자에 그대로 실려야 한다."""
    from eigen_benchmark.drivers import p5_stats
    tags = set()
    for arm in ("diagonal", "exact", "qr"):
        k = run.e21_settings(arm)
        tags.add(k["tag"])
        assert k["arms"] == (arm,) and k["control"] is False
        assert k["tag"] == p5_stats.E21_TAGS[arm]
        assert k["geometry"] == run.E21_GEOMETRY and k["geometry"]["kind"] == "L"
        assert k["bc"] == "clamped_x0" and not k.get("normalize_input", False)
        assert "mc_points" not in k and "resample_every" not in k      # 가우스 격자족, 250회
        assert k["pair_scoring"] and k["gram_offdiag"] and k["per_seed_span"] and k["save_params"]
        assert k["ref_name"] == "p5_reference_e21_lbracket" and k["ref_level"] == (80, 16, 8)
        assert k["family_axes"] == ((12, 13, 14, 15, 16), (6, 7, 8), (6, 7, 8))
        assert k["audit_grid"] == (24, 12, 12)
        # 사다리는 감사 격자의 4/3, 2, 8/3배(E6과 같은 배율)
        assert k["check_grid"] == k["check_grids"][0]
        assert [tuple(g) for g in k["check_grids"]] == [
            tuple(round(f * v) for v in k["audit_grid"]) for f in (4 / 3, 2, 8 / 3)]
    assert len(tags) == 3
    with pytest.raises(ValueError):
        run.e21_settings("reorth_diagonal")


def test_e21_family_clears_the_minimum_certified_l_grid_on_every_axis():
    """격자족의 가장 성긴 격자가 L의 최소 인증 격자이고, 어느 축이든 한 칸 줄이면 불합격이다."""
    from eigen_benchmark.problems import p5_block as p5
    lo = tuple(min(a) for a in run.E21_FAMILY_AXES)
    below = [tuple(v - (i == j) for j, v in enumerate(lo)) for i in range(3)]
    q = p5.quadrature_convergence([lo, *below], tol=1e-6, geometry=run.E21_GEOMETRY)
    err = [max(r["err_a"], r["err_b"]) for r in q["rows"]]
    assert lo == (12, 6, 6) and err[0] <= 1e-6 and all(e > 1e-6 for e in err[1:])


def test_e21_audit_grid_is_the_certified_one_and_clears_the_margin():
    import math
    from eigen_benchmark.problems import p5_block as p5
    fam_max = tuple(max(a) for a in run.E21_FAMILY_AXES)
    r = p5.certify_audit_grid(fam_max, geometry=run.E21_GEOMETRY)
    assert r["meets_tol"] and r["chosen"] == run.E21_AUDIT_GRID
    assert run.E21_AUDIT_GRID == tuple(math.ceil(1.5 * v) for v in fam_max)


def test_e21_reference_artifact_matches_the_settings():
    """참조는 설정과 같은 구조·사다리·감사 격자에서 인증되고, 벽 모드는 그 지배 성분에서 나온다."""
    import json
    import os
    from eigen_benchmark.drivers import p5_stats, run_p5
    from eigen_benchmark.render import figures_p5
    p = "docs/_generated/data/paper4/p5_reference_e21_lbracket.json"
    if not os.path.exists(p):
        pytest.skip("E21 참조가 없다 — run_p5.structure_reference를 먼저 돌린다")
    ref = json.load(open(p, encoding="utf-8"))
    assert ref["geometry"] == run.E21_GEOMETRY
    assert [tuple(lv) for lv in ref["ladder"]] == [tuple(lv) for lv in run_p5.L_LADDER]
    assert tuple(ref["ladder"][-1]) == run.E21_REF_LEVEL
    assert tuple(ref["cert_grid"]) == run.E21_AUDIT_GRID
    assert ref["clusters"][:6] == [[1], [2], [3], [4], [5], [6]]
    assert ref["shape_certified"] and ref["clusters_consistent_across_levels"]
    wall = figures_p5.predicted_wall([m["dominant"] for m in ref["modes"][:6]])
    assert p5_stats.E21_MODES == tuple(range(wall, 7))


def test_e21_tau_calibrates_on_the_l_family_and_reports_progress(monkeypatch):
    got = {}
    monkeypatch.setattr(run, "tau_calibration", lambda **kw: got.update(kw) or {"tau": 1.0})
    run.e21_tau(outdir="x", device="cpu")
    assert got["tag"] == run.E21_TAU_TAG == "e21_lbracket"
    assert got["audit_grid"] == run.E21_AUDIT_GRID and got["family_axes"] == run.E21_FAMILY_AXES
    assert got["geometry"] == run.E21_GEOMETRY and got["heartbeat_every"] == run.HEARTBEAT_ITERS


def test_e21_run_refuses_without_its_tau(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.e21_run("diagonal", outdir=str(tmp_path))


def test_e21_run_reads_its_own_tau(tmp_path, monkeypatch):
    import json
    (tmp_path / "manifest_p5_tau_e21_lbracket.json").write_text(json.dumps({"tau": 0.2}))
    (tmp_path / "manifest_p5_tau.json").write_text(json.dumps({"tau": 5.0}))
    got = {}
    monkeypatch.setattr(run, "grid_v2", lambda **kw: got.update(kw) or {"verdicts": None})
    run.e21_run("exact", outdir=str(tmp_path), device="cpu")
    assert got["tau"] == 0.2 and got["tag"] == "e21lb_exact" and got["arms"] == ("exact",)


def test_e22_settings_carry_the_prespecified_design():
    """사전규정 2026-10-04 E22의 값이 실행 인자에 그대로 실려야 한다."""
    from eigen_benchmark.drivers import p5_stats
    tags = set()
    for arm in ("diagonal", "exact", "qr"):
        k = run.e22_settings(arm)
        tags.add(k["tag"])
        assert k["arms"] == (arm,) and k["control"] is False
        assert k["tag"] == p5_stats.E22_TAGS[arm]
        assert k["geometry"] == run.E22_GEOMETRY and k["geometry"]["kind"] == "bilayer"
        assert k["geometry"]["E"] / k["geometry"]["E_top"] == pytest.approx(10.0)
        assert k["bc"] == "clamped_x0" and not k.get("normalize_input", False)
        assert "mc_points" not in k and "resample_every" not in k
        assert k["pair_scoring"] and k["gram_offdiag"] and k["per_seed_span"] and k["save_params"]
        assert k["ref_name"] == "p5_reference_e22_bilayer" and k["ref_level"] == (48, 16, 8)
        assert k["family_axes"] == ((16, 17, 18, 19, 20), (8, 9, 10), (3, 4))
        assert k["audit_grid"] == (30, 15, 6)
        assert k["check_grid"] == k["check_grids"][0]
        assert [tuple(g) for g in k["check_grids"]] == [
            tuple(round(f * v) for v in k["audit_grid"]) for f in (4 / 3, 2, 8 / 3)]
    assert len(tags) == 3
    with pytest.raises(ValueError):
        run.e22_settings("reorth_diagonal")


def test_e22_family_matches_e13a_in_plane_and_on_average():
    """평면 축은 E13-a 격자족과 같고, 층마다 3·4점이라 두께 평균 7점·평균 점수가 E13-a와 같다."""
    import itertools
    import numpy as np
    from eigen_benchmark.neural import p5_neural as p5n
    from eigen_benchmark.problems import p5_block as p5
    ax = run.E22_FAMILY_AXES
    assert ax[0] == p5n.FAMILY_NX and ax[1] == p5n.FAMILY_NY
    assert 2 * np.mean(ax[2]) == np.mean(p5n.FAMILY_NZ)
    n = [len(p5._tensor_grid(g, run.E22_GEOMETRY)[1]) for g in itertools.product(*ax)]
    box = {k: v for k, v in run.E22_GEOMETRY.items() if k not in ("kind", "E_top")}
    n13 = [len(p5._tensor_grid(g, box)[1])
           for g in itertools.product(p5n.FAMILY_NX, p5n.FAMILY_NY, p5n.FAMILY_NZ)]
    assert np.mean(n) == np.mean(n13) == 1134


def test_e22_family_is_certified_and_the_unsplit_grid_is_not():
    """층마다 나눈 격자족은 모두 인증을 지나고, 계면을 가로지르는 Gauss는 두께 방향 48점으로도
    1e-6에 닿지 못한다(불연속 피적분이라 대수적으로만 수렴한다 — 실측 약 n⁻²)."""
    import itertools
    from eigen_benchmark.problems import p5_block as p5
    g = run.E22_GEOMETRY
    q = p5.quadrature_convergence(list(itertools.product(*run.E22_FAMILY_AXES)), tol=1e-6,
                                  geometry=g)
    assert all(max(r["err_a"], r["err_b"]) <= 1e-6 for r in q["rows"])
    pr, wr = p5._tensor_grid((96, 40, 24), g)
    ar, _ = p5._forms_from_field(*p5.manufactured_field(pr, g), wr, g, pts=pr)
    box = {k: v for k, v in g.items() if k not in ("kind", "E_top")}
    pu, wu = p5._tensor_grid((20, 10, 48), box)
    au, _ = p5._forms_from_field(*p5.manufactured_field(pu, g), wu, g, pts=pu)
    assert abs(au / ar - 1) > 1e-6


def test_e22_audit_grid_is_the_certified_one_and_clears_the_margin():
    import math
    from eigen_benchmark.problems import p5_block as p5
    fam_max = tuple(max(a) for a in run.E22_FAMILY_AXES)
    r = p5.certify_audit_grid(fam_max, geometry=run.E22_GEOMETRY)
    assert r["meets_tol"] and r["chosen"] == run.E22_AUDIT_GRID
    assert run.E22_AUDIT_GRID == tuple(math.ceil(1.5 * v) for v in fam_max)


def test_e22_reference_artifact_matches_the_settings():
    import json
    import os
    from eigen_benchmark.drivers import p5_stats, run_p5
    from eigen_benchmark.render import figures_p5
    p = "docs/_generated/data/paper4/p5_reference_e22_bilayer.json"
    if not os.path.exists(p):
        pytest.skip("E22 참조가 없다 — run_p5.structure_reference를 먼저 돌린다")
    ref = json.load(open(p, encoding="utf-8"))
    assert ref["geometry"] == run.E22_GEOMETRY
    assert [tuple(lv) for lv in ref["ladder"]] == [tuple(lv) for lv in run_p5.LADDER]
    assert tuple(ref["ladder"][-1]) == run.E22_REF_LEVEL
    assert tuple(ref["cert_grid"]) == run.E22_AUDIT_GRID
    assert ref["clusters"][:6] == [[1], [2], [3], [4], [5], [6]]
    assert ref["shape_certified"] and ref["clusters_consistent_across_levels"]
    wall = figures_p5.predicted_wall([m["dominant"] for m in ref["modes"][:6]])
    assert p5_stats.E22_MODES == tuple(range(wall, 7))


def test_e22_tau_calibrates_on_the_bilayer_family(monkeypatch):
    got = {}
    monkeypatch.setattr(run, "tau_calibration", lambda **kw: got.update(kw) or {"tau": 1.0})
    run.e22_tau(outdir="x", device="cpu")
    assert got["tag"] == run.E22_TAU_TAG == "e22_bilayer"
    assert got["audit_grid"] == run.E22_AUDIT_GRID and got["family_axes"] == run.E22_FAMILY_AXES
    assert got["geometry"] == run.E22_GEOMETRY and got["heartbeat_every"] == run.HEARTBEAT_ITERS


def test_e22_run_reads_its_own_tau_and_refuses_without_it(tmp_path, monkeypatch):
    import json
    with pytest.raises(FileNotFoundError):
        run.e22_run("diagonal", outdir=str(tmp_path))
    (tmp_path / "manifest_p5_tau_e22_bilayer.json").write_text(json.dumps({"tau": 0.5}))
    got = {}
    monkeypatch.setattr(run, "grid_v2", lambda **kw: got.update(kw) or {"verdicts": None})
    run.e22_run("qr", outdir=str(tmp_path), device="cpu")
    assert got["tau"] == 0.5 and got["tag"] == "e22bi_qr" and got["arms"] == ("qr",)


def test_e21b_is_the_e21_exact_arm_with_the_family_redrawn_every_iteration():
    """사전규정 2026-10-06 E21b — E21 정확 arm과 재추첨 주기 하나만 다르다(E17의 가우스·1과 같은 조작)."""
    k = run.e21b_settings()
    base = run.e21_settings("exact")
    assert k["resample_every"] == 1 and k["tag"] == "e21lb_exact_g1"
    drop = lambda d: {kk: v for kk, v in d.items() if kk not in ("tag", "resample_every")}
    assert drop(k) == drop(base) and "resample_every" not in base


def test_e21b_run_reads_the_e21_tau_and_refuses_without_it(tmp_path, monkeypatch):
    import json
    with pytest.raises(FileNotFoundError):
        run.e21b_run(outdir=str(tmp_path))
    (tmp_path / "manifest_p5_tau_e21_lbracket.json").write_text(json.dumps({"tau": 10.0}))
    got = {}
    monkeypatch.setattr(run, "grid_v2", lambda **kw: got.update(kw) or {"verdicts": None})
    run.e21b_run(outdir=str(tmp_path), device="cpu")
    assert got["tau"] == 10.0 and got["resample_every"] == 1 and got["tag"] == "e21lb_exact_g1"


def test_e21c_is_the_e21_exact_arm_with_a_denser_family():
    """사전규정 2026-10-08 E21c — E21 정확 arm과 격자족 밀도(와 그에 맞춘 감사·사다리)만 다르다."""
    import math
    k, base = run.e21c_settings(), run.e21_settings("exact")
    changed = ("tag", "family_axes", "audit_grid", "check_grid", "check_grids")
    drop = lambda d: {kk: v for kk, v in d.items() if kk not in changed}
    assert drop(k) == drop(base) and "resample_every" not in k
    assert k["tag"] == "e21lb_exact_dense"
    assert k["family_axes"] == ((16, 17, 18, 19, 20), (8, 9, 10), (6, 7, 8))
    fam_max = tuple(max(a) for a in k["family_axes"])
    assert k["audit_grid"] == tuple(math.ceil(1.5 * v) for v in fam_max) == (30, 15, 12)
    assert k["check_grid"] == k["check_grids"][0]
    assert [tuple(g) for g in k["check_grids"]] == [
        tuple(round(f * v) for v in k["audit_grid"]) for f in (4 / 3, 2, 8 / 3)]


def test_e21c_audit_grid_is_certified_and_the_family_clears_the_minimum():
    from eigen_benchmark.problems import p5_block as p5
    k = run.e21c_settings()
    r = p5.certify_audit_grid(tuple(max(a) for a in k["family_axes"]), geometry=run.E21_GEOMETRY)
    assert r["meets_tol"] and r["chosen"] == k["audit_grid"]
    lo = tuple(min(a) for a in k["family_axes"])
    assert all(a >= b for a, b in zip(lo, (12, 6, 6)))      # E21의 최소 인증 격자 이상


def test_e21c_run_reads_the_e21_tau_and_refuses_without_it(tmp_path, monkeypatch):
    import json
    with pytest.raises(FileNotFoundError):
        run.e21c_run(outdir=str(tmp_path))
    (tmp_path / "manifest_p5_tau_e21_lbracket.json").write_text(json.dumps({"tau": 10.0}))
    got = {}
    monkeypatch.setattr(run, "grid_v2", lambda **kw: got.update(kw) or {"verdicts": None})
    run.e21c_run(outdir=str(tmp_path), device="cpu")
    assert got["tau"] == 10.0 and got["tag"] == "e21lb_exact_dense"
    assert got["audit_grid"] == (30, 15, 12)
