"""P5 신경 arm — §3.1 파일럿(아키텍처 고정)과 §4 Stage 0(갈래 판정 게이트).

**게이트가 있는 이유.** "3D에서 기전이 재현되는가"는 본실험 그리드를 태우기 전에
하루 안에 알 수 있다. Stage 0은 arm (b) 8시드로 modes 1–4를 풀고, 저장 스팬의
max off-diag(C)를 보고 갈래를 정한다. P1은 k = 2에서 0.991, P4는 2.2e-16이었다.

판정 문턱은 **실행 전에 고정된 값**이며 이 파일에 상수로 박혀 있다.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np

from .. import metrics as mt
from . import manifest

# ── §3.1 파일럿 (사전 고정) ─────────────────────────────────────────────────
PILOT_SEEDS = 4
PILOT_MODES = 2
PILOT_ITERS = 20000
PILOT_ARCHS = [(64, 4), (128, 5)]       # 후보. 작은 쪽을 우선한다.
PILOT_ELAM_MAX = 1e-3                   # mode-1 e_λ 중앙값 선택 규칙

# ── §4 Stage 0 (사전 고정) ─────────────────────────────────────────────────
STAGE0_SEEDS = 8
STAGE0_MODES = 4
STAGE0_ITERS = 20000
BRANCH_A_THRESHOLD = 0.10               # k=2 max off-diag 중앙값 ≥ → 갈래 A
BRANCH_B_THRESHOLD = 0.01               # ≤ → 갈래 B, 사이면 k=3,4로 재판정
ADDED = "2026-09-03, P5 확장(사전규정 초안 기준)"


def _reference(data_dir: str, name: str = "p5_reference") -> dict:
    """참조 Λ*를 읽는다. 없으면 실행을 거부한다 — 참조 없이 채점할 수 없다.

    `name`은 E13이 구조마다 다른 참조를 쓰기 위한 것이다(기본값은 커밋된 본실험 참조).
    이름을 틀리면 다른 구조의 Λ*로 채점하게 되고 그 오류는 조용하므로, 파일이 없으면
    만들지 않고 거부한다."""
    p = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p}가 없다 — run_p5로 참조와 인증을 먼저 만든다(계획 §2)")
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["mode"]): {"lam_star": float(r["lam_star"]),
                             "rel_residual": float(r["rel_residual"])}
            for r in rows}


def pilot(outdir=None, quick: bool = False, device=None) -> dict:
    """§3.1 — 후보 아키텍처 둘을 4시드 × modes 1–2로 돌려 하나를 고른다.

    **선택 규칙은 사전 고정이다**: mode-1 e_λ 중앙값이 1e-3 이하인 **가장 작은**
    아키텍처. 둘 다 만족하면 64×4, 둘 다 실패하면 128×5로 가고 실패를 기록한다.
    파일럿 결과는 본실험에 포함하지 않는다."""
    from ..neural import p5_neural as p5n

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    ref = _reference(d)
    n_seeds = 2 if quick else PILOT_SEEDS
    iters = 30 if quick else PILOT_ITERS
    archs = PILOT_ARCHS[:1] if quick else PILOT_ARCHS

    rows = []
    for width, depth in archs:
        out = p5n.solve_p5_neural(n_modes=PILOT_MODES, n_seeds=n_seeds,
                                  iters=iters, width=width, depth=depth,
                                  projection="exact", device=device)
        for m in range(PILOT_MODES):
            lam = out["lam"][:, m]
            e = np.abs(lam - ref[m + 1]["lam_star"]) / ref[m + 1]["lam_star"]
            rows.append({
                "width": width, "depth": depth,
                "n_params_per_seed": out["disclosure"]["n_params_per_seed"],
                "mode": m + 1, "n_seeds": n_seeds, "iters": iters,
                "lam_median": float(np.median(lam)),
                "lam_star": ref[m + 1]["lam_star"],
                "e_lam_median": float(np.median(e)),
                "e_lam_min": float(np.min(e)), "e_lam_max": float(np.max(e)),
                "seconds": float(out["stage_seconds"][m]),
                "ms_per_iter": float(out["stage_seconds"][m] / iters * 1000)})
        print(f"[p5 파일럿] {width}x{depth}: mode-1 e_λ 중앙값 "
              f"{rows[-PILOT_MODES]['e_lam_median']:.3e}  "
              f"({rows[-PILOT_MODES]['ms_per_iter']:.0f} ms/iter)")

    m1 = {(r["width"], r["depth"]): r for r in rows if r["mode"] == 1}
    passing = [a for a in archs if m1[a]["e_lam_median"] <= PILOT_ELAM_MAX]
    if passing:
        chosen, note = min(passing, key=lambda a: a[0] * a[1]), "rule satisfied"
    else:
        chosen = archs[-1]
        note = (f"no candidate met the pre-specified e_lam median <= "
                f"{PILOT_ELAM_MAX:g}; the largest candidate is taken and the "
                # ASCII만 쓴다 — 이 열은 렌더된 표에 그대로 인쇄된다
                f"failure is recorded (plan Section 9)")
    for r in rows:
        r["chosen"] = bool((r["width"], r["depth"]) == chosen)
        r["selection_rule"] = f"smallest arch with mode-1 median e_lam <= {PILOT_ELAM_MAX:g}"
        r["selection_note"] = note
    manifest.write_csv(os.path.join(d, "p5_pilot.csv"), rows)
    print(f"[p5 파일럿] 채택 {chosen[0]}x{chosen[1]} — {note}")

    manifest.write_json(os.path.join(d, "manifest_p5_pilot.json"), manifest.build({
        "driver": "run_p5_neural.pilot", "quick": bool(quick),
        "candidates": [list(a) for a in archs], "chosen": list(chosen),
        "n_seeds": n_seeds, "modes": PILOT_MODES, "iters": iters,
        "selection_rule": rows[0]["selection_rule"], "selection_note": note,
        "excluded_from_inference": "pilot cells fix the architecture and are not "
                                   "reported as benchmark outcomes",
        "added": ADDED}))
    return {"chosen": chosen, "rows": rows, "note": note, "outdir": d}


def stage0(outdir=None, width: int = 64, depth: int = 4, quick: bool = False,
           device=None) -> dict:
    """§4 — arm (b) 8시드로 modes 1–4를 풀고 저장 스팬의 비직교성으로 갈래를 정한다.

    측정하는 것은 **원시 저장 출력**의 max off-diag(C)다. 사영된 것을 재면 언제나
    직교로 나와 전제조건 질문 자체가 사라진다."""
    from ..neural import p5_neural as p5n

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    n_seeds = 2 if quick else STAGE0_SEEDS
    # 빠른 경로도 **모드 3까지** 푼다 — 판정이 k = 2 진단에 걸려 있고,
    # 모드 2는 k = 1만 넘겨받으므로 모드 2까지로는 게이트를 시험할 수 없다.
    n_modes = 3 if quick else STAGE0_MODES
    iters = 30 if quick else STAGE0_ITERS

    out = p5n.solve_p5_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                              width=width, depth=depth, projection="exact",
                              device=device)
    rows = []
    for m, sd in enumerate(out["span_diagnostics"], start=1):
        if sd is None:
            continue
        rows.append({"stage_mode": m, "k": sd["k"], "n_seeds": n_seeds,
                     "iters": iters, "width": width, "depth": depth,
                     "max_offdiag": sd["max_offdiag"],
                     "lam_min_normalized": sd["lam_min_normalized"],
                     "kappa_gram": sd["kappa_gram"],
                     "kappa_gram_times_eps": sd["kappa_gram_times_eps"],
                     "dev_identity": sd["dev_identity"],
                     "qr_orth_defect": sd["qr_orth_defect"],
                     "n_seeds_summarized": sd["n_seeds_summarized"]})
        print(f"[p5 Stage 0] k={sd['k']}: max off-diag {sd['max_offdiag']:.4f}  "
              f"lam_min(C) {sd['lam_min_normalized']:.3e}  "
              f"kappa2(G) {sd['kappa_gram']:.3e}")

    by_k = {r["k"]: r for r in rows}
    branch, basis = _decide(by_k)
    for r in rows:
        r["branch"] = branch
        r["threshold_a"] = BRANCH_A_THRESHOLD
        r["threshold_b"] = BRANCH_B_THRESHOLD
        r["decision_basis"] = basis
    manifest.write_csv(os.path.join(d, "p5_stage0.csv"), rows)
    print(f"[p5 Stage 0] 갈래 {branch} — {basis}")

    manifest.write_json(os.path.join(d, "manifest_p5_stage0.json"), manifest.build({
        "driver": "run_p5_neural.stage0", "quick": bool(quick),
        "arm": out["arm"], "n_seeds": n_seeds, "modes": n_modes, "iters": iters,
        "architecture": [width, depth],
        "threshold_branch_a": BRANCH_A_THRESHOLD,
        "threshold_branch_b": BRANCH_B_THRESHOLD,
        "branch": branch, "decision_basis": basis,
        "measured_on": "the raw stored outputs handed to each stage, before any "
                       "projection; measuring the projected span would always "
                       "look orthogonal and the question would vanish",
        "comparison": "P1 had max off-diag 0.991 at k = 2; P4 had 2.2e-16",
        "added": ADDED}))
    return {"branch": branch, "basis": basis, "rows": rows, "outdir": d}


def _decide(by_k: dict):
    """§4의 판정 규칙. 중간값이면 k = 3, 4로 재판정하고 여전히 중간이면 보수적으로 A."""
    if 2 not in by_k:
        raise ValueError("k = 2 진단이 없다 — modes 1-3 이상을 풀어야 한다")
    v2 = by_k[2]["max_offdiag"]
    if v2 >= BRANCH_A_THRESHOLD:
        return "A", f"k=2 max off-diag {v2:.4f} >= {BRANCH_A_THRESHOLD}"
    if v2 <= BRANCH_B_THRESHOLD:
        return "B", f"k=2 max off-diag {v2:.4f} <= {BRANCH_B_THRESHOLD}"
    deeper = [by_k[k]["max_offdiag"] for k in (3, 4) if k in by_k]
    if deeper and max(deeper) >= BRANCH_A_THRESHOLD:
        return "A", (f"k=2 {v2:.4f} was intermediate; deeper k reached "
                     f"{max(deeper):.4f} >= {BRANCH_A_THRESHOLD}")
    if deeper and max(deeper) <= BRANCH_B_THRESHOLD:
        return "B", (f"k=2 {v2:.4f} was intermediate; deeper k stayed at "
                     f"{max(deeper):.4f} <= {BRANCH_B_THRESHOLD}")
    return "A", (f"k=2 {v2:.4f} and deeper k remained intermediate; branch A is "
                 "taken as the conservative reading (full grid)")


def main(outdir=None, quick: bool = False, device=None) -> dict:
    """파일럿 → Stage 0. **순서가 규약의 일부다** — 아키텍처를 먼저 고정해야
    Stage 0의 저장 스팬이 본실험과 같은 함수족에서 나온다."""
    p = pilot(outdir, quick=quick, device=device)
    s = stage0(outdir, width=p["chosen"][0], depth=p["chosen"][1],
               quick=quick, device=device)
    return {"pilot": p, "stage0": s}


if __name__ == "__main__":
    print(main())


# ═══════════════════════════════════════════════════════════════════════════
# §5 본실험 그리드 — Stage 0이 갈래 A를 낸 뒤에만 돈다
# ═══════════════════════════════════════════════════════════════════════════

GRID_ARMS_A = ("diagonal", "exact", "qr")     # (a′), (b), (b′)
GRID_ARMS_B = ("diagonal", "qr")              # (b)는 P1에서 (b′)와 fp64 등가 확인됨
GRID_MODES = 6
GRID_SEEDS_A = 20
GRID_SEEDS_B = 12
GRID_ITERS = 20000
CONV_TOL, CONV_TAIL = 1e-3, 0.1               # 사전등록 수렴 게이트(무변경)
# §3.6 구적분리를 신경 arm의 상시 계기로 둔다. 사전 인증은 참조 모드를 덮을 뿐
# 최적화가 도달하는 함수를 덮지 못하므로, 셀마다 두 구적차수로 재서 비교한다.
GRID_CHECK_QUAD = (40, 20, 15)


# ── v2 사전규정 (2026-09-06) ────────────────────────────────────────────────
# v1의 두 결함을 설계로 닫는다. A1: 학습 기저가 훈련 구적을 착취한다 → 감사 격자 분리와
# Rayleigh 하한 게이트. A2: 2차원에서 정한 수렴 게이트가 3차원에서 표본을 소멸시킨다
# → 감사 격자 Λ의 꼬리 최대변화를 읽는 게이트와 전용 시드에서의 τ 보정.
AUDIT_GRID = (30, 15, 12)        # §2.2-2.3 인증 결과. 훈련 격자족 최대 (20,10,8)의 1.5배
RAYLEIGH_TOL = 1e-6              # §2.5 — Λ < Λ₁(1 − tol) 이면 invalid_quadrature
DRIFT_FLAG = 0.05                # §2.6 — 이 이상이면 e_algebraic·e_measurement 병기
INVALID_ARM_LIMIT = 0.20         # §6 — arm의 invalid 비율이 이를 넘으면 판정불가
OUTCOMES_V2 = ("invalid_quadrature",) + tuple(mt.OUTCOMES)
# §3.2 — τ 보정 전용 시드. 본실험 시드(0..GRID_SEEDS_A-1)와 서로소여야 한다.
TAU_SEEDS = tuple(range(9001, 9009))
TAU_CAL_MODES = (1, 3, 6)
TAU_CAL_ARM = "exact"


def classify_v2(lam, lam1, matched, mode, mac, e_lam, conv,
                rayleigh_tol: float = RAYLEIGH_TOL) -> str:
    """§4 분류 순서: invalid_quadrature → 게이트 → correct → basin → spurious.

    첫 분기가 핵심이다. 허용 시행함수의 Rayleigh 몫은 λ₁ 아래로 갈 수 없으므로,
    감사 격자에서 그 아래로 나온 값은 함수의 성질이 아니라 구적 인공물이다. 이것을
    4분류 안에서 처리하면 `non_converged`가 '최적화가 못 앉았다'와 '계기가 못 쟀다'를
    같은 칸에 섞어 담는다 — v1에서 실제로 그랬다.

    `tol`은 참조 자체의 불확실도를 흡수하는 여유다. P5 참조의 외삽 잔차는 9.6e-4이므로
    계획이 정한 1e-6은 그보다 촘촘하다. 그 간극에 떨어지는 셀은 `lam_margin`으로
    보고되어 사후에 식별 가능하다(§2.6 병기 규칙)."""
    import math
    if math.isfinite(lam) and lam < lam1 * (1.0 - rayleigh_tol):
        return "invalid_quadrature"
    if not conv or not math.isfinite(e_lam):
        return "non_converged"
    if matched == mode and mac >= mt.MAC_MIN and e_lam <= mt.ELAM_MAX:
        return "correct"
    if matched < mode and mac >= mt.MAC_MIN:
        return "lower_mode_basin"
    return "spurious"


def ladder_tail_spread(values, k: int = 3) -> float:
    """구적 사다리 끝 `k`칸의 상대 폭 — 가장 촘촘한 차수들에서 Λ가 아직 움직이는가.

    **드리프트와 다른 양이다.** 드리프트는 두 격자의 *차이*이지 수렴 여부가 아니어서,
    발산하는 셀에서도 이웃 두 값이 우연히 가까울 수 있다(E4 시드 2의 Λ가 125 → 141로
    상대차 0.121인데 그 수열은 1.93 → 7.66 → 125 → 141로 발산한다).

    **차이의 비(수축비)로도 안 된다.** 이미 기계정밀도까지 수렴한 셀은 차이가 잡음이고
    그 비도 잡음이다 — E4 시드 0(네 차수가 일곱 자리까지 같다)에서 수축비가 3.39로 나와
    미수렴으로 읽혔다. 실데이터 20셀로 두 후보를 재 보고 이쪽을 골랐다: 꼬리 폭은
    수렴 셀 최대 9.4e-05와 미수렴 셀 최소 5.3e-02 사이를 570배 간극으로 가른다.

    칸이 `k`개 미만이면 NaN이다 — 두 점으로는 못 하는 일이 정확히 드리프트가 못 한 일이다."""
    v = [float(x) for x in values]
    if len(v) < k:
        return float("nan")
    t = v[-k:]
    return (max(t) - min(t)) / abs(t[-1])


def tail_variation(hist, s: int, tail: float = CONV_TAIL) -> float:
    """감사 격자 Λ의 꼬리 구간에서 **연속 스냅샷 사이** 최대 상대변화.

    v1은 꼬리의 양 끝 두 점만 비교했다. 중간에 크게 튀었다가 제자리로 돌아오면
    수렴으로 통과한다 — 3차원처럼 정착이 느린 곳에서 실제로 일어나는 형태다."""
    import math
    import numpy as _np
    h = _np.asarray(hist, dtype=float)
    if h.ndim < 2 or h.shape[0] < 3:
        return float("inf")
    k = max(int(h.shape[0] * tail), 2)
    seg = h[-k:, s]
    if not _np.all(_np.isfinite(seg)):
        return float("inf")
    den = _np.maximum(_np.abs(seg[1:]), 1e-300)
    v = float(_np.max(_np.abs(_np.diff(seg)) / den))
    return v if math.isfinite(v) else float("inf")


def converged_v2(hist, s: int, tau: float, tail: float = CONV_TAIL) -> bool:
    """§3.1 — 게이트가 읽는 양은 감사 격자 Λ의 상대변화다(훈련 격자 J가 아니다).
    재표집 때문에 훈련 J는 격자가 바뀔 때마다 점프하므로 게이트로 쓸 수 없다."""
    return tail_variation(hist, s, tail) < tau


def tau_from_calibration(values, quantile: float = 90.0) -> float:
    """§3.2 — 보정 집합의 90분위를 1·2·5×10^k 중 **위로** 가장 가까운 값으로 올린다.

    올림이지 반올림이 아니다. 내림이면 보정 집합의 10 %가 이미 게이트를 못 넘는다."""
    import math
    import numpy as _np
    v = _np.asarray(list(values), dtype=float)
    v = v[_np.isfinite(v)]
    if v.size == 0:
        raise ValueError("보정 값이 비었다 — τ를 정할 수 없다")
    q = float(_np.percentile(v, quantile))
    if q <= 0:
        raise ValueError(f"90분위가 {q} — 보정 실행이 잘못됐다")
    e = math.floor(math.log10(q))
    for m in (1.0, 2.0, 5.0, 10.0):
        c = m * 10.0 ** e
        if c >= q * (1 - 1e-12):
            return float(c)
    return float(10.0 ** (e + 1))


def _converged(hist, s: int, tol: float = CONV_TOL, tail: float = CONV_TAIL) -> bool:
    """P1·P4와 **같은** 게이트 — 꼬리 10 %의 상대변화."""
    import math
    if len(hist) < 3:
        return False
    k = max(int(len(hist) * tail), 2)
    a, b = float(hist[-k][s]), float(hist[-1][s])
    return math.isfinite(b) and abs(b - a) / max(abs(b), 1e-300) <= tol


def _mass_mac(A, B, w) -> float:
    """질량 가중 MAC — 벡터장이므로 세 성분을 함께 내적한다(P4의 2성분 판본과 같은 정의)."""
    ab = float((w * (A * B).sum(axis=0)).sum())
    aa = float((w * (A ** 2).sum(axis=0)).sum())
    bb = float((w * (B ** 2).sum(axis=0)).sum())
    if aa <= 0 or bb <= 0:
        return float("nan")
    return ab ** 2 / (aa * bb)


def clusters_from_gaps(lams, rule: float | None = None) -> list:
    """참조 고유값을 간격 규칙(`run_p5.GAP_RULE`)으로 군집으로 묶는다 — 1-기반 모드 번호 튜플.

    인접 상대 간격 (λᵢ₊₁ − λᵢ)/λᵢ 가 규칙보다 작으면 같은 군집이다. 축퇴쌍은 쌍 안의 어떤
    회전도 고유벡터이므로 모드별이 아니라 군집(부분공간) 단위로 정합해야 한다."""
    from .run_p5 import GAP_RULE
    r = GAP_RULE if rule is None else rule
    out, cur = [], [1]
    for i in range(1, len(lams)):
        if (lams[i] - lams[i - 1]) / abs(lams[i - 1]) < r:
            cur.append(i + 1)
        else:
            out.append(tuple(cur))
            cur = [i + 1]
    out.append(tuple(cur))
    return out


def _subspace_mac(phi, Rs, w) -> float:
    """phi의 질량 에너지 가운데 span(Rs)가 담는 몫 bᵀG⁻¹b / (φ, φ)_W. Rs가 하나이면
    대수적으로 `_mass_mac`과 같다. 쌍 안의 참조 벡터가 어떤 회전이든 값이 같다."""
    ip = lambda A, B: float((w * (A * B).sum(axis=0)).sum())
    G = np.array([[ip(A, B) for B in Rs] for A in Rs])
    b = np.array([ip(A, phi) for A in Rs])
    pp = ip(phi, phi)
    if pp <= 0:
        return float("nan")
    return float(b @ np.linalg.solve(G, b)) / pp


def match_clustered(phi, shapes_ref, w, clusters, target: int, earlier=None,
                    dup_min: float | None = None) -> dict:
    """군집 단위 정합. 반환 matched_mode, mac, cluster, dup_mac.

    - 군집마다 부분공간 MAC을 재고 최댓값 군집에 정합한다. 목표 모드가 그 군집에 있으면
      matched_mode = target, 아니면 그 군집의 가장 낮은 번호(아래면 유인, 위면 spurious).
    - **같은 쌍 안의 중복.** 목표의 군집에서 앞 단계가 이미 찾은 방향(`earlier[m']`)과 MAC이
      `dup_min` 이상이면 새 모드가 아니라 그 방향을 다시 찾은 것이므로 matched_mode = m'
      (→ `classify_v2`가 유인으로 분류한다). 크기 1 군집에는 적용하지 않는다.
    - **하위 호환.** 군집이 모두 1개짜리이면 모드별 `_mass_mac`의 argmax와 비트 단위로 같다."""
    dup_min = mt.MAC_MIN if dup_min is None else dup_min
    if all(len(c) == 1 for c in clusters):
        macs = [_mass_mac(phi, R, w) for R in shapes_ref]
        j = int(np.nanargmax(macs)) + 1
        return {"matched_mode": j, "mac": float(macs[j - 1]), "cluster": (j,),
                "dup_mac": None}
    vals = [_mass_mac(phi, shapes_ref[c[0] - 1], w) if len(c) == 1
            else _subspace_mac(phi, [shapes_ref[k - 1] for k in c], w) for c in clusters]
    ci = int(np.nanargmax(vals))
    K, mac = clusters[ci], float(vals[ci])
    j = target if target in K else min(K)
    dup = None
    if target in K and len(K) > 1 and earlier:
        prev = [(mp, earlier[mp]) for mp in K if mp < target and mp in earlier]
        if prev:
            dup, dup_mode = max((_mass_mac(phi, f, w), mp) for mp, f in prev)
            dup = float(dup)
            if dup >= dup_min:
                j = dup_mode
    return {"matched_mode": j, "mac": mac, "cluster": tuple(K), "dup_mac": dup}


def _component_fractions(phi, w) -> list:
    """질량가중 성분 비율 [f_x, f_y, f_z] — 쌍 안의 혼합(예: 4 f_y f_z)을 시드마다 재기 위해."""
    e = (w * phi ** 2).sum(axis=1)
    return [float(v) for v in e / e.sum()]


def _reference_shapes(n_modes: int, grid, geometry=None, bc: str = "clamped_x0",
                      start_vector=None, level=None):
    """참조 모드형상을 **채점이 쓰는 구적점에서** 평가한다(§2.2가 인증한 그 점).

    `geometry`·`bc`는 E13이 구조를 바꿀 때 참조도 그 구조의 것이어야 하기 때문이다.
    `level`은 그 구조의 사다리 최상단이다(E16 정사각 단면은 대칭 사다리라 기준 구조의
    (48, 16, 8)과 다르다). 기본값은 커밋된 사다리의 최상단이다."""
    from ..problems import p5_block as p5
    from .run_p5 import LADDER
    pts, _ = p5._tensor_grid(grid, geometry)
    faces = "two_face" if bc == "two_face" else "x0"
    lv = LADDER[-1] if level is None else tuple(int(v) for v in level)
    r = p5.solve(*lv, n_modes=n_modes, geometry=geometry, faces=faces,
                 start_vector=start_vector)
    return [p5.eval_mode_at(r["mesh"], r["vec_full"][:, m], pts).T
            for m in range(n_modes)]        # 각 (3, n_q)


def grid(outdir=None, branch: str | None = None, width: int = 128, depth: int = 5,
         quick: bool = False, device=None, arms=None,
         resample_every: int | None = None) -> dict:
    """§5 — 갈래에 따른 본실험. 채점은 P1·P4와 **같은 규칙**을 그대로 쓴다.

    시드 수가 P1–P4의 50보다 작으므로(20 또는 12) Wilson 구간이 넓어진다.
    계획 §5가 그 사실을 사전 선언했고, 표 캡션에도 적는다."""
    import math

    import numpy as np

    from .. import metrics as mt
    from ..neural import p5_neural as p5n

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    if branch is None:
        p = os.path.join(d, "p5_stage0.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}가 없다 — Stage 0 게이트를 먼저 통과한다")
        with open(p, encoding="utf-8") as f:
            branch = next(csv.DictReader(f))["branch"]
    arms = arms or (GRID_ARMS_A if branch == "A" else GRID_ARMS_B)
    n_seeds = GRID_SEEDS_A if branch == "A" else GRID_SEEDS_B
    n_modes, iters = GRID_MODES, GRID_ITERS
    if quick:
        n_seeds, n_modes, iters, width, depth = 2, 2, 20, 16, 2
        arms = arms if arms else ("exact",)

    ref = _reference(d)
    shapes_ref = _reference_shapes(n_modes, p5n.CERTIFIED_GRID)
    rows, recs, diag = [], [], []
    for arm in arms:
        out = p5n.solve_p5_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                                  width=width, depth=depth, projection=arm,
                                  device=device, check_grid=GRID_CHECK_QUAD,
                                  resample_every=resample_every)
        w = out["w"]
        for m in range(1, n_modes + 1):
            target = ref[m]["lam_star"]
            cnt = {k: 0 for k in mt.OUTCOMES}
            for s in range(n_seeds):
                lam = float(out["lam"][s, m - 1])
                phi = out["shapes"][m - 1][s]
                macs = [_mass_mac(phi, R, w) for R in shapes_ref]
                j = int(np.nanargmax(macs)) + 1
                mac = float(macs[j - 1])
                e = abs(lam - target) / target if math.isfinite(lam) else float("nan")
                conv = _converged(out["hist"][m - 1], s)
                oc = ("non_converged" if not conv or not math.isfinite(e)
                      else "correct" if (j == m and mac >= mt.MAC_MIN
                                         and e <= mt.ELAM_MAX)
                      else "lower_mode_basin" if (j < m and mac >= mt.MAC_MIN)
                      else "spurious")
                cnt[oc] += 1
                recs.append({"problem": "P5", "arm": out["arm"], "mode": m,
                             "seed": s, "outcome": oc, "matched_mode": j,
                             "mac": mac, "e_lam": e, "Lam": lam,
                             "Lam_ref": target, "converged": conv,
                             "projected_mass_fraction":
                                 float(out["projected_mass_fraction"][s, m - 1]),
                             "Lam_check": float(out["lam_check"][s, m - 1]),
                             "quad_drift":
                                 float(out["quadrature_drift"][s, m - 1])})
            lo, hi = mt.wilson(cnt["correct"], n_seeds)
            pm = out["projected_mass_fraction"][:, m - 1]
            rows.append({"problem": "P5", "arm": out["arm"], "mode": m,
                         "n": n_seeds, "iters": iters,
                         **{k: cnt[k] for k in mt.OUTCOMES},
                         "p_correct": cnt["correct"] / n_seeds,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "seconds": float(out["stage_seconds"][m - 1]),
                         # **붕괴와 낮은-모드 착지를 가르는 열.** 질량비가 1에 가까운데
                         # Λ가 작으면 낮은 모드를, 0에 가까우면 사영이 함수를 지운 것이다.
                         "proj_mass_median": float(np.median(pm)),
                         "proj_mass_min": float(np.min(pm)),
                         # 두 구적차수의 상대차. 크면 그 셀의 Lam은 함수의
                         # Rayleigh 몫이 아니라 구적 인공물이다.
                         "check_grid": "x".join(map(str, out["check_grid"])),
                         "quad_drift_median":
                             float(np.median(out["quadrature_drift"][:, m - 1])),
                         "quad_drift_max":
                             float(np.max(out["quadrature_drift"][:, m - 1])),
                         "width": width, "depth": depth, "branch": branch})
            print(f"[p5 그리드] {arm:8s} 모드{m}: p_correct "
                  f"{cnt['correct'] / n_seeds:.2f}  basin {cnt['lower_mode_basin']}  "
                  f"spur {cnt['spurious']}  nc {cnt['non_converged']}  "
                  f"질량비중앙 {np.median(pm):.3e}")
        for m, sd in enumerate(out["span_diagnostics"], start=1):
            if sd is None:
                continue
            diag.append({"arm": out["arm"], "stage_mode": m, "n_seeds": n_seeds,
                         "iters": iters, **sd})
        # **arm이 끝날 때마다 저장한다.** 20시간짜리 실행이 종료 시에만 쓰면
        # 중간 실패로 전부 잃는다 — 실제로 그렇게 잃은 적이 있다.
        manifest.write_csv(os.path.join(d, "p5_neural.csv"), rows)
        manifest.write_csv(os.path.join(d, "p5_span_diagnostics.csv"), diag)
        manifest.write_jsonl(os.path.join(d, "p5_neural_records.jsonl"), recs)
        print(f"[p5 그리드] {arm} 저장 완료 — 누적 {len(rows)}행", flush=True)

    manifest.write_csv(os.path.join(d, "p5_neural.csv"), rows)
    manifest.write_csv(os.path.join(d, "p5_span_diagnostics.csv"), diag)
    manifest.write_jsonl(os.path.join(d, "p5_neural_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p5_neural.json"), manifest.build({
        "driver": "run_p5_neural.grid", "quick": bool(quick), "branch": branch,
        "arms": list(arms), "n_modes": n_modes, "n_seeds": n_seeds,
        "iters": iters, "architecture": [width, depth],
        "quadrature_grid": list(p5n.CERTIFIED_GRID),
        "check_quadrature_grid": list(GRID_CHECK_QUAD),
        "resample_every": resample_every,
        "quadrature_separation": "every cell is re-measured on a second "
                                 "quadrature order and the relative drift is "
                                 "reported; a large drift means the optimizer "
                                 "found a strain field the certified grid does "
                                 "not resolve, so that cell's Lam is a "
                                 "quadrature artifact rather than a Rayleigh "
                                 "quotient",
        "scoring": "unchanged from P1-P4: MAC >= 0.9, e_lam <= 0.05, four "
                   "outcomes, 95 % Wilson; the seed count is 20 rather than 50, "
                   "so the intervals are wider and the captions say so",
        "not_run": ["budget control", "initialization ladder", "fp32 axis"],
        "added": ADDED}))
    return {"rows": rows, "branch": branch, "outdir": d}


# ── v2 본실험 (사전규정 2026-09-06) ─────────────────────────────────────────
V2_RESAMPLE_EVERY = 250          # §2.1 훈련 격자 재추첨 주기
V2_CHECK_GRID = (40, 20, 15)     # §2.4 상시 계기 — 감사 격자보다 한 단계 위
V2_TAU_SEED = 9001               # §3.2 보정 전용 시드 기저(본실험 seed=0과 서로소)
V2_TAU_N_SEEDS = 8


def _geometry_record(geometry):
    """매니페스트의 기하 기록. 수치는 float로 남기고 종류 표지("kind": "L")는 문자열 그대로 둔다."""
    if geometry is None:
        return None
    return {k: v if isinstance(v, str) else float(v) for k, v in geometry.items()}


def tau_calibration(outdir=None, quick: bool = False, device=None,
                    width: int = 128, depth: int = 5, tag: str | None = None,
                    geometry=None, bc: str = "clamped_x0", audit_grid=None,
                    family_axes=None, heartbeat_every: int | None = None) -> dict:
    """§3.2 — 본실험을 보기 전에 τ를 1회 정한다.

    보정 런은 결과에 포함하지 않고, 보정 시드는 본실험에서 쓰지 않는다. 모드 1·3·6의
    꼬리 변화만 τ 결정에 쓰지만 순차 deflation이라 1–6을 모두 풀어야 모드 6에 닿는다.

    **새 구조(E16)는 `tag`를 준다.** 구조·감사 격자·격자족이 바뀌면 τ도 다시 정해야 하고,
    그 결과는 `manifest_p5_tau_<tag>.json`에 따로 쓴다 — 기준 구조의 `manifest_p5_tau.json`을
    덮으면 이후 기준 구조 실행이 조용히 다른 τ로 게이트된다. 기본값은 커밋된 동작이다."""
    import numpy as np
    from ..neural import p5_neural as p5n

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    launch = manifest.launch_provenance()
    n_modes, iters, n_seeds, snap = 6, GRID_ITERS, V2_TAU_N_SEEDS, 25
    if quick:
        # 게이트는 꼬리 10 %의 **연속 스냅샷**을 읽는다 — 스냅샷이 2개뿐이면
        # tail_variation이 inf를 돌려주고 τ를 정할 수 없다.
        n_modes, iters, n_seeds, width, depth, snap = 3, 20, 2, 16, 2, 2
    modes = tuple(m for m in TAU_CAL_MODES if m <= n_modes)

    audit = AUDIT_GRID if audit_grid is None else tuple(int(v) for v in audit_grid)
    out = p5n.solve_p5_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                              width=width, depth=depth, projection=TAU_CAL_ARM,
                              device=device, audit_grid=audit,
                              resample_every=V2_RESAMPLE_EVERY,
                              snapshot_every=snap, seed=V2_TAU_SEED,
                              geometry=geometry, bc=bc, family_axes=family_axes,
                              heartbeat_every=heartbeat_every)
    rows, vals = [], []
    for m in modes:
        for s in range(n_seeds):
            v = tail_variation(out["hist"][m - 1], s)
            rows.append({"mode": m, "seed": s, "tail_variation": v,
                         "Lam_audit": float(out["lam"][s, m - 1]),
                         "Lam_train": float(out["lam_train"][s, m - 1])})
            vals.append(v)
    finite = [v for v in vals if np.isfinite(v)]
    tau = tau_from_calibration(finite) if finite else float("nan")
    sfx = "" if tag is None else f"_{tag}"
    manifest.write_csv(os.path.join(d, f"p5_tau_calibration{sfx}.csv"), rows)
    manifest.write_json(os.path.join(d, f"manifest_p5_tau{sfx}.json"), manifest.build({
        "driver": "run_p5_neural.tau_calibration", "quick": bool(quick), "launch": launch,
        "tau": tau, "rule": "90th percentile of the calibration set, rounded UP "
                            "to the nearest 1/2/5 x 10^k; rounding down would "
                            "leave 10 % of the calibration set already failing",
        "n_finite": len(finite), "n_total": len(vals),
        "quantile_90": float(np.percentile(finite, 90)) if finite else None,
        "seed_base": V2_TAU_SEED, "n_seeds": n_seeds, "modes": list(modes),
        "arm": TAU_CAL_ARM, "iters": iters, "audit_grid": list(audit),
        "resample_every": V2_RESAMPLE_EVERY, "tag": tag,
        "geometry": _geometry_record(geometry),
        "bc": bc, "family_axes": None if family_axes is None
        else [list(a) for a in family_axes],
        "disjoint": "calibration seeds are drawn from base 9001 and the main grid "
                    "from base 0, so no training stream is shared",
        "added": "2026-09-06, v2 pre-specification, before any v2 main run"}))
    print(f"[p5 τ보정] {len(finite)}/{len(vals)}개 유한, "
          f"90분위 {np.percentile(finite, 90) if finite else float('nan'):.3e} "
          f"→ τ = {tau:.1e}", flush=True)
    return {"tau": tau, "rows": rows, "outdir": d}


HEARTBEAT_ITERS = 2000           # 모드 하나가 3.5시간이다 — 그 안이 보여야 한다


def _median_spread(out, mode: int):
    """그 모드 셀들의 사다리 꼬리폭 중앙값. 사다리를 끄면 None이다."""
    import numpy as np
    lad = out.get("lam_ladder")
    if lad is None:
        return None
    vals = [ladder_tail_spread([float(v) for v in lad[s, mode - 1]])
            for s in range(lad.shape[0])]
    vals = [v for v in vals if v == v]              # NaN 제외
    return float(np.median(vals)) if vals else None


def make_mode_writer(outdir, tag, *, condition, ref, tau, shapes_ref, audit,
                     width, depth, rows=None, recs=None):
    """모드가 끝날 때마다 **곁가지** 파일에 남긴다(2026-09-12 정전 대응).

    arm 단위 저장(v1 B5)은 21–48시간짜리 실행에 성기다. 정전으로 E2 (b′) 24.2시간과
    E3 53.3시간이 산출물 0바이트로 끝났다. 그렇다고 정규 경로를 고칠 수는 없다 —
    커밋된 실행과 같은 코드가 같은 표를 내야 하므로, 여기서 **같은 규칙을 따로 한 번
    더** 적용한다. 두 결과가 같음은 코드가 아니라 테스트가 지킨다
    (`test_partial_rows_agree_with_the_final_rows_on_the_shared_columns`).

    §3.6 구적분리(`Lam_check`·`quad_drift`)는 모든 모드가 끝난 뒤 한 번 도는
    후처리라 곁가지에는 **없다**. 그래서 이 파일은 정규 산출물의 부분집합이며,
    판정에 쓰지 않는다(사전규정 §6은 arm 단위 판정이다).
    """
    import datetime
    import math
    import time as _time

    import numpy as np

    # 누적 목록은 호출자가 arm을 가로질러 소유한다. arm마다 새로 만들면 둘째 arm이
    # 첫 arm의 곁가지 파일을 덮는다 — 정규 산출물이 v1 B5에서 겪은 바로 그 실패다.
    rows = [] if rows is None else rows
    recs = [] if recs is None else recs
    lam1 = float(ref[1]["lam_star"])
    t0 = _time.time()
    prog = os.path.join(outdir, f"p5_{tag}_progress.json")

    def _progress(complete: bool) -> dict:
        return {"tag": tag, "condition": condition, "modes_done": len(rows),
                "complete": complete, "elapsed_hours": (_time.time() - t0) / 3600,
                "updated_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "complete_means": "이 진행 파일이 가리키는 arm이 정규 산출물에 "
                                  "기록됐다는 뜻이다. 여러 arm을 한 프로세스에서 "
                                  "돌리면 arm마다 켜졌다 꺼진다.",
                "note": "곁가지 기록 — 판정은 완결된 arm의 정규 산출물로만 한다"}

    def _write(info):
        try:
            m, n_seeds = int(info["mode"]), int(info["n_seeds"])
            target = float(ref[m]["lam_star"])
            w, cnt = info["w"], {k: 0 for k in OUTCOMES_V2}
            for s in range(n_seeds):
                lam = float(info["lam"][s])
                macs = [_mass_mac(info["shapes"][s], R, w) for R in shapes_ref]
                j = int(np.nanargmax(macs)) + 1
                mac = float(macs[j - 1])
                e = abs(lam - target) / target if math.isfinite(lam) else float("nan")
                tv = tail_variation(info["hist"], s)
                conv = tv < tau
                oc = classify_v2(lam, lam1, j, m, mac, e, conv)
                cnt[oc] += 1
                recs.append({"problem": "P5", "arm": info["arm"],
                             "condition": condition, "mode": m, "seed": s,
                             "outcome": oc, "matched_mode": j, "mac": mac,
                             "e_lam": e, "Lam": lam,
                             "Lam_train": float(info["lam_train"][s]),
                             "Lam_ref": target, "converged": conv,
                             "tail_variation": tv, "tau": tau,
                             "lam_margin": lam / lam1 - 1.0,
                             "projected_mass_fraction": float(info["proj_mass"][s])})
            lo, hi = mt.wilson(cnt["correct"], n_seeds)
            pm = np.asarray(info["proj_mass"], dtype=float)
            rows.append({"problem": "P5", "arm": info["arm"],
                         "condition": condition, "mode": m, "n": n_seeds,
                         "iters": int(info["iters"]),
                         **{k: cnt[k] for k in OUTCOMES_V2},
                         "p_correct": cnt["correct"] / n_seeds,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "p_invalid": cnt["invalid_quadrature"] / n_seeds,
                         "tau": tau, "seconds": float(info["seconds"]),
                         "proj_mass_median": float(np.median(pm)),
                         "proj_mass_min": float(np.min(pm)),
                         "audit_grid": "x".join(map(str, audit)),
                         "width": width, "depth": depth})
            manifest.write_csv(
                os.path.join(outdir, f"p5_neural_{tag}_partial.csv"), rows)
            manifest.write_jsonl(
                os.path.join(outdir, f"p5_neural_{tag}_partial_records.jsonl"), recs)
            manifest.write_json(prog, _progress(False))
            print(f"[p5 곁가지] {info['arm']}/{condition} 모드{m} 저장 — "
                  f"누적 {len(rows)}행", flush=True)
        except Exception as exc:      # noqa: BLE001
            # **곁가지가 본 계산을 죽이면 본말이 뒤집힌다.** 21시간을 장부 때문에
            # 잃지 않도록 예외는 여기서 끊고 크게 찍기만 한다.
            print(f"[p5 곁가지] 중간저장 실패 — 계산은 계속한다: {exc!r}", flush=True)

    def _finish():
        """정규 산출물이 다 쓰인 뒤 호출한다 — 완결 표지는 매니페스트지만,
        진행 파일만 보고도 완결 여부를 알 수 있어야 오해가 없다."""
        try:
            manifest.write_json(prog, _progress(True))
        except Exception as exc:      # noqa: BLE001
            print(f"[p5 곁가지] 진행 파일 마감 실패: {exc!r}", flush=True)

    _write.finish = _finish
    return _write


def grid_v2(outdir=None, tau: float | None = None, arms=None, width: int = 128,
            depth: int = 5, quick: bool = False, device=None,
            control: bool = True, iters: int | None = None,
            tag: str = "v2", audit_grid=None, family_axes=None,
            check_grid=None, check_grids=None, gram_offdiag: bool = False,
            save_params: bool = False, normalize_input: bool = False,
            per_seed_span: bool = False,
            geometry=None, bc: str = "clamped_x0",
            ref_name: str = "p5_reference", fixed_train_grid=None,
            pair_scoring: bool = False, mc_points: int | None = None,
            mc_resample_every: int = 1, storage: str = "raw",
            batch_diag: bool = False, ref_level=None,
            resample_every: int | None = None, ladder_chunk: int | None = None,
            stage_from: dict | None = None, checkpoint_every: int | None = None,
            save_hist: bool = False, lr_milestones=None, fourier: dict | None = None,
            solver: str = "rayleigh", boundary_points: int = 600,
            n_modes: int | None = None, init_seed_offset: int = 0, traction_weight: float = 1.0) -> dict:
    """§5 본실험 — 훈련은 재표집 격자, 게이트·채점은 감사 격자.

    `pair_scoring`은 축퇴쌍이 있는 구조를 위한 군집 단위 정합이다(`match_clustered`). 끄면
    (기본) 기존 모드별 정합 코드를 그대로 탄다 — 커밋된 실행과 같은 경로다. 켜면 참조 모드
    형상을 고정 시작벡터로 푼다: 쌍 안의 참조 벡터는 임의 회전이라, 판정은 부분공간이라
    불변이어도 기록되는 진단이 실행마다 달라지지 않게 하기 위해서다.

    E15: `mc_points`를 주면 재표집 조건의 훈련 구적이 균등 몬테카를로가 되고(조건 이름 "mc",
    `mc_resample_every`회마다 새 표본), `storage`가 저장 규약을, `batch_diag`가 훈련 표본 위
    저장 스팬 비대각 계측을 정한다. 모두 기본값이면 커밋된 동작과 같다.

    `control`은 §5의 고정격자 대조군이다: arm (b)를 **훈련도 감사 격자에서** 돌려
    재표집 효과 자체를 결과로 남긴다.

    E21h: `stage_from`(npz·tag·arm·cond·n_stored·width·depth·iters)을 주면 앞 `n_stored`개 모드를 그 실행이
    저장한 망으로 두고(`solve_p5_neural(stored=)`) 새 단계만 학습·기록한다 — 행·기록·곁가지·스팬 진단은 새
    단계뿐이고, 망 파라미터는 전 모드다. `checkpoint_every`는 새 단계의 중간 망을 `p5_params_<tag>_ckpt.npz`에,
    `save_hist`는 새 단계의 감사 격자 Λ 이력을 `p5_hist_<tag>.npz`에 남긴다. 셋 다 기본값이면 커밋된 동작이다.

    E23: `lr_milestones`는 학습률 감소 시점을 반복 수로 못 박고, `fourier`는 학습하는 단계의 망에 고정 Fourier 특징을
    붙인다(`solve_p5_neural`). 둘 다 기본값이면 커밋된 동작이다.

    E25: `n_modes`는 풀 모드 수다(기본 None = 6). 저장 망 시작(`stage_from`)과 함께 한 단계만 다시 풀 때 쓴다.

    E27: `stage_from={"mixed": [단계별 출처 목록, …], "width", "depth", "iters"}`이면 저장 단계마다 시드를 출처(실행의
    망 파일·모드·Fourier 여부·정규화 여부)에 나눠 담는다(`p5_exploit.load_mixed_stages`). `init_seed_offset`은 학습하는
    단계의 초기화 난수에 더한다(재초기화 경로). 둘 다 기본값이면 커밋된 동작이다.

    E28: `traction_weight`는 강형식 해석기의 자유면 견인력 항 가중이다(기본값 1이면 커밋된 손실).

    E24: `solver="strong"`이면 변분형 대신 강형식 해석기(`p5_pinn.solve_p5_pinn`: 잔차 + 자유면 견인력 + 정규화 +
    직교성 벌점, 학습되는 λ)로 푼다. arm은 `("penalty",)` 하나이고 대조군·저장 망 시작은 다루지 않는다. 채점·기록은 같은
    규칙이고, 기록에 강형식 계측(학습된 λ, 잔차·견인력의 훈련/감사 값, 직교 결함)이 붙는다. 기본값이면 커밋된 동작이다."""
    import math

    import numpy as np

    from ..neural import p5_neural as p5n
    from .run_p5 import LADDER

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    launch = manifest.launch_provenance()
    if tau is None:
        p = os.path.join(d, "manifest_p5_tau.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}가 없다 — τ 보정을 먼저 돌린다(§3.2)")
        import json
        tau = float(json.load(open(p, encoding="utf-8"))["tau"])

    # **`()`와 `None`을 구별한다.** 빈 튜플이 falsy라 기본 3 arm으로 대체되면, 대조군만
    # 돌리려는 호출이 조용히 네 조건을 돈다(E12에서 실제로 그랬다). `None`만 기본값이다.
    arms = GRID_ARMS_A if arms is None else tuple(arms)
    # `iters`는 E2 예산 대조가 바꾸는 **유일한** 값이다(사전규정 E2). 기본값은 본실험.
    n_seeds, snap = GRID_SEEDS_A, 25
    n_modes_given = n_modes is not None
    n_modes = GRID_MODES if n_modes is None else int(n_modes)
    iters = GRID_ITERS if iters is None else int(iters)
    if quick:
        n_seeds, iters, width, depth, snap = 2, 20, 16, 2, 2
        # 명시한 모드 수는 quick에서도 둔다 — 단계 1만 푸는 실행(E28a)을 시험이 그대로 밟게 한다.
        n_modes = n_modes if n_modes_given else 2

    ref = _reference(d, ref_name)
    lam1 = float(ref[1]["lam_star"])
    # E3은 격자족을 올리므로 감사 격자도 함께 올라간다(사전규정 E3, v2 §2.2의 1.5배 규칙).
    audit = AUDIT_GRID if audit_grid is None else tuple(int(v) for v in audit_grid)
    check = V2_CHECK_GRID if check_grid is None else tuple(int(v) for v in check_grid)
    shapes_ref = _reference_shapes(n_modes, audit, geometry, bc,
                                   start_vector="ones" if pair_scoring else None,
                                   level=ref_level)
    clusters = (clusters_from_gaps([ref[m]["lam_star"] for m in range(1, n_modes + 1)])
                if pair_scoring else [(m,) for m in range(1, n_modes + 1)])
    rows, recs, diag = [], [], []

    # `resample_every`는 가우스 격자족의 재추첨 주기다(E17). 기본값은 본실험의 250회다.
    gauss_every = V2_RESAMPLE_EVERY if resample_every is None else int(resample_every)
    plan = [(a, mc_resample_every if mc_points else gauss_every,
             "mc" if mc_points else "resampled") for a in arms]
    if control:
        plan.append(("exact", None, "fixed_audit"))

    stored_ens, first = None, 1
    if stage_from is not None:
        if control:
            raise ValueError("stage_from은 재표집 조건만 다룬다(control=False)")
        import torch as _torch
        _dev = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        if "mixed" in stage_from:
            from .p5_exploit import load_mixed_stages
            stored_ens = load_mixed_stages(stage_from["mixed"], n_seeds=n_seeds, width=int(stage_from["width"]),
                                           depth=int(stage_from["depth"]), geometry=geometry, device=_dev)
            first = len(stage_from["mixed"]) + 1
        else:
            from .p5_exploit import load_ensembles
            stored_ens = load_ensembles(
                stage_from["npz"], stage_from["arm"], stage_from["cond"],
                n_modes=int(stage_from["n_stored"]), n_seeds=n_seeds, width=int(stage_from["width"]),
                depth=int(stage_from["depth"]), device=_dev)
            first = int(stage_from["n_stored"]) + 1

    part_rows, part_recs = [], []        # 곁가지 누적 — arm을 가로질러 산다
    ps_rows = []                         # 시드별 스팬 진단(별도 파일, 스키마 불변)
    for arm, resample, cond in plan:
        writer = make_mode_writer(d, tag, condition=cond, ref=ref, tau=tau,
                                  shapes_ref=shapes_ref, audit=audit, width=width,
                                  depth=depth, rows=part_rows, recs=part_recs)
        ckpt = None
        if checkpoint_every:
            ck_store = {}

            def ckpt(info, arm=arm, cond=cond, store=ck_store):
                # E21h 곁가지 — 새 단계의 중간 망. 쌓인 것을 매번 다시 써 정전에도 남는다.
                for name, v in info["params"].items():
                    store[f"{arm}_{cond}_m{info['mode']}_it{info['it']}_{name}"] = v
                np.savez_compressed(os.path.join(d, f"p5_params_{tag}_ckpt.npz"), **store)
        if solver == "strong":
            if control or stage_from is not None or arm != "penalty":
                raise ValueError("강형식은 penalty arm의 재표집 조건만 다룬다(control=False, stage_from=None)")
        if solver != "strong" and float(traction_weight) != 1.0:
            raise ValueError("traction_weight는 강형식 해석기에만 있다")
        if solver == "strong":
            from ..neural import p5_pinn
            out = p5_pinn.solve_p5_pinn(
                n_modes=n_modes, n_seeds=n_seeds, iters=iters, width=width, depth=depth,
                device=device, audit_grid=audit, resample_every=resample, snapshot_every=snap,
                family_axes=family_axes, geometry=geometry, bc=bc,
                mc_points=mc_points, boundary_points=boundary_points,
                on_mode_done=writer, heartbeat_every=HEARTBEAT_ITERS,
                check_grids=check_grids if check_grids is not None else [check],
                gram_offdiag=gram_offdiag, per_seed_span=per_seed_span,
                return_params=save_params, normalize_input=normalize_input,
                ladder_chunk=ladder_chunk, traction_weight=traction_weight)
        elif solver != "rayleigh":
            raise ValueError(f"solver는 'rayleigh' 또는 'strong'이어야 합니다: {solver!r}")
        else:
            out = p5n.solve_p5_neural(
                n_modes=n_modes, n_seeds=n_seeds, iters=iters, width=width,
                depth=depth, projection=arm, device=device,
                audit_grid=audit, check_grid=check,
                resample_every=resample, snapshot_every=snap, family_axes=family_axes,
                geometry=geometry, bc=bc,
                # 고정 조건의 훈련 격자. 기본값은 커밋된 동작(감사 격자 자체)이고,
                # E12는 그것을 감사 격자보다 성긴 격자로 바꿔 held-out을 유지한다.
                grid=((fixed_train_grid or audit) if resample is None
                      else p5n.CERTIFIED_GRID),
                on_mode_done=writer, heartbeat_every=HEARTBEAT_ITERS,
                check_grids=check_grids, gram_offdiag=gram_offdiag,
                per_seed_span=per_seed_span,
                return_params=save_params, normalize_input=normalize_input,
                mc_points=mc_points if resample else None,
                storage=storage if resample else "raw",
                batch_diag=batch_diag and resample is not None, ladder_chunk=ladder_chunk,
                stored=stored_ens,
                stored_iters=(None if stage_from is None else
                          [int(v) for v in stage_from["iters"]]
                          if isinstance(stage_from["iters"], (list, tuple)) else int(stage_from["iters"])),
                on_checkpoint=ckpt, checkpoint_every=checkpoint_every,
                lr_milestones=lr_milestones, fourier=fourier, init_seed_offset=init_seed_offset)
        w = out["w"]
        for m in range(first, n_modes + 1):
            target = ref[m]["lam_star"]
            cnt = {k: 0 for k in OUTCOMES_V2}
            for s in range(n_seeds):
                lam = float(out["lam"][s, m - 1])
                phi = out["shapes"][m - 1][s]
                if pair_scoring:
                    mc = match_clustered(phi, shapes_ref, w, clusters, target=m,
                                         earlier={mp: out["shapes"][mp - 1][s]
                                                  for mp in range(1, m)})
                    j, mac = mc["matched_mode"], mc["mac"]
                else:
                    macs = [_mass_mac(phi, R, w) for R in shapes_ref]
                    j = int(np.nanargmax(macs)) + 1
                    mac = float(macs[j - 1])
                e = abs(lam - target) / target if math.isfinite(lam) else float("nan")
                tv = tail_variation(out["hist"][m - 1], s)
                conv = tv < tau
                oc = classify_v2(lam, lam1, j, m, mac, e, conv)
                cnt[oc] += 1
                drift = float(out["quadrature_drift"][s, m - 1])
                lad = (None if out.get("lam_ladder") is None
                       else [float(v) for v in out["lam_ladder"][s, m - 1]])
                spr = None if lad is None else ladder_tail_spread(lad)
                # §2.6 — 원인(훈련 구적)과 증상(채점)이 걸치면 둘 다 명명한다.
                attr = ("e_algebraic|e_measurement"
                        if oc == "invalid_quadrature" or drift > DRIFT_FLAG
                        else "e_optimization" if oc == "non_converged"
                        else "")
                recs.append({"problem": "P5", "arm": out["arm"], "condition": cond,
                             "mode": m, "seed": s, "outcome": oc, "matched_mode": j,
                             "mac": mac, "e_lam": e, "Lam": lam,
                             "Lam_train": float(out["lam_train"][s, m - 1]),
                             "Lam_ref": target, "converged": conv,
                             "tail_variation": tv, "tau": tau,
                             # 하한까지의 여유. 참조 외삽 잔차(9.6e-4)보다 작은 음수
                             # 여유는 병리가 아니라 참조 불확실도일 수 있다.
                             "lam_margin": lam / lam1 - 1.0,
                             "projected_mass_fraction":
                                 float(out["projected_mass_fraction"][s, m - 1]),
                             "Lam_check": float(out["lam_check"][s, m - 1]),
                             "quad_drift": drift, "attribution": attr,
                             # 계측만 — `classify_v2`는 이 둘을 보지 않는다.
                             "lam_ladder": lad, "ladder_tail_spread": spr,
                             **({"cluster": list(mc["cluster"]), "dup_mac": mc["dup_mac"],
                                 "component_fractions": _component_fractions(phi, w)}
                                if pair_scoring else {}),
                             **({"storage": out["storage"],
                                 "batch_offdiag": float(out["batch_offdiag"][s, m - 1])}
                                if out.get("batch_offdiag") is not None else {}),
                             # E24 강형식 계측 — 변분형 기록에는 없다
                             **({key: float(out[key][s, m - 1]) for key in
                                 ("lam_param", "res_train", "res_audit", "bc_train", "bc_audit",
                                  "orth_train", "orth_audit")}
                                if out.get("lam_param") is not None else {})})
            lo, hi = mt.wilson(cnt["correct"], n_seeds)
            pm = out["projected_mass_fraction"][:, m - 1]
            dr = out["quadrature_drift"][:, m - 1]
            rows.append({"problem": "P5", "arm": out["arm"], "condition": cond,
                         "mode": m, "n": n_seeds, "iters": iters,
                         **{k: cnt[k] for k in OUTCOMES_V2},
                         "p_correct": cnt["correct"] / n_seeds,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "p_invalid": cnt["invalid_quadrature"] / n_seeds,
                         "tau": tau,
                         "seconds": float(out["stage_seconds"][m - 1]),
                         "proj_mass_median": float(np.median(pm)),
                         "proj_mass_min": float(np.min(pm)),
                         "audit_grid": "x".join(map(str, audit)),
                         "check_grid": "x".join(map(str, out["check_grid"])),
                         "quad_drift_median": float(np.median(dr)),
                         "quad_drift_max": float(np.max(dr)),
                         "ladder_spread_median": _median_spread(out, m),
                         "width": width, "depth": depth})
            print(f"[p5 v2] {arm:8s} {cond:12s} 모드{m}: correct "
                  f"{cnt['correct']}/{n_seeds}  invalid {cnt['invalid_quadrature']}  "
                  f"basin {cnt['lower_mode_basin']}  spur {cnt['spurious']}  "
                  f"nc {cnt['non_converged']}", flush=True)
        for m, sd in enumerate(out["span_diagnostics"], start=1):
            if sd is None or m < first:
                continue
            ps = sd.pop("per_seed", None)
            diag.append({"arm": out["arm"], "condition": cond, "stage_mode": m,
                         "n_seeds": n_seeds, "iters": iters, **sd})
            for row in (ps or []):
                ps_rows.append({"arm": out["arm"], "condition": cond,
                                "stage_mode": m, "iters": iters, **row})
        # **arm이 끝날 때마다 저장한다**(v1 B5).
        manifest.write_csv(os.path.join(d, f"p5_neural_{tag}.csv"), rows)
        if diag:                         # 모드 1만 푸는 실행에는 앞 스팬이 없어 행이 없다(E28a 단계 1)
            manifest.write_csv(os.path.join(d, f"p5_span_diagnostics_{tag}.csv"), diag)
        manifest.write_jsonl(os.path.join(d, f"p5_neural_{tag}_records.jsonl"), recs)
        if ps_rows:
            manifest.write_csv(os.path.join(d, f"p5_span_per_seed_{tag}.csv"), ps_rows)
        if save_params and out.get("params") is not None:
            # 망을 남기면 다음 재측정이 순전파 비용이 된다(E7은 이것이 없어 77 h 썼다).
            import numpy as _np
            _np.savez_compressed(
                os.path.join(d, f"p5_params_{tag}.npz"),
                **{f"{arm}_{cond}_m{m + 1}_{k}": v
                   for m, pm in enumerate(out["params"]) for k, v in pm.items()})
        if save_hist:
            np.savez_compressed(os.path.join(d, f"p5_hist_{tag}.npz"), snapshot_every=snap,
                                **{f"{arm}_{cond}_m{m}": np.asarray(out["hist"][m - 1])
                                   for m in range(first, n_modes + 1)})
        print(f"[p5 v2] {arm}/{cond} 저장 완료 — 누적 {len(rows)}행", flush=True)
        writer.finish()

    # §6 — invalid 비율이 한계를 넘은 arm은 1차 결과를 판정불가로 기록한다.
    verdicts = {}
    for arm in {r["arm"] for r in rows}:
        sub = [r for r in rows if r["arm"] == arm and r["condition"] == "resampled"]
        if not sub:
            continue
        inv = sum(r["invalid_quadrature"] for r in sub)
        tot = sum(r["n"] for r in sub)
        verdicts[arm] = {"p_invalid": inv / tot,
                         "primary_result": ("indeterminate"
                                            if inv / tot > INVALID_ARM_LIMIT
                                            else "usable")}
    manifest.write_json(os.path.join(d, f"manifest_p5_neural_{tag}.json"),
                        manifest.build({
        "driver": "run_p5_neural.grid_v2", "quick": bool(quick), "tag": tag,
        "launch": launch,
        "input_scaling": "per-axis [0,1]^3" if normalize_input else "raw",
        "geometry": _geometry_record(geometry),
        "bc": bc, "reference": ref_name,
        "reference_level": list(LADDER[-1] if ref_level is None else ref_level),
        "pair_scoring": bool(pair_scoring), "clusters": [list(c) for c in clusters],
        "mc_points": mc_points, "mc_resample_every": mc_resample_every if mc_points else None,
        "storage": storage, "batch_diag": bool(batch_diag), "ladder_chunk": ladder_chunk,
        "stage_from": (None if stage_from is None else
                       {**stage_from, "mixed": [[{**src, "npz": os.path.basename(src["npz"])} for src in st]
                                                for st in stage_from["mixed"]]} if "mixed" in stage_from else
                       {**stage_from, "npz": os.path.basename(stage_from["npz"])}),
        "init_seed_offset": int(init_seed_offset), "traction_weight": float(traction_weight),
        "checkpoint_every": checkpoint_every, "save_hist": bool(save_hist),
        "lr_milestones": None if lr_milestones is None else [int(v) for v in lr_milestones],
        "fourier": None if fourier is None else dict(fourier),
        "solver": solver,
        **({"boundary_points": int(boundary_points),
            "strong_form": {k_: out[k_] for k_ in ("weights", "eigenvalue", "loss_term_names")}}
           if solver == "strong" else {}),
        "fixed_train_grid": (None if fixed_train_grid is None
                             else list(fixed_train_grid)),
        "arms": list(arms), "control": bool(control), "n_modes": n_modes,
        "n_seeds": n_seeds, "iters": iters, "architecture": [width, depth],
        "audit_grid": list(audit), "check_quadrature_grid": list(check),
        "family_axes": None if family_axes is None
        else [list(a) for a in family_axes],
        "train_grid": (f"uniform Monte Carlo, {mc_points} points, redrawn every "
                       f"{mc_resample_every} iterations" if mc_points else
                       f"resampled Gauss family, redrawn every {gauss_every} iterations"),
        "resample_every": None if mc_points else gauss_every,
        "control_condition": "arm (b) trained on the audit grid itself, so the "
                             "resampling effect is a measured contrast rather "
                             "than an assumption",
        "tau": tau, "rayleigh_tol": RAYLEIGH_TOL,
        "classification_order": list(OUTCOMES_V2),
        "gate": "max relative change between consecutive audit-grid snapshots over "
                "the final 10 %, below tau; the training objective is not read "
                "because resampling makes it jump",
        "invalid_arm_limit": INVALID_ARM_LIMIT, "verdicts": verdicts,
        "not_run": ["fp32 axis", "budget control", "initialization ladder"],
        "added": "2026-09-06, v2 pre-specification"}))
    return {"rows": rows, "tau": tau, "verdicts": verdicts, "outdir": d}


# ── 사후 실패양식 귀인 (2026-09-10) ─────────────────────────────────────────
# **사전규정 밖이다.** v2 §6의 5분류와 문턱(MAC ≥ 0.9 ∧ e_λ ≤ 0.05)은 그대로 두고,
# 커밋된 기록만 다시 읽어 `spurious` 한 칸을 이름 있는 실패 양식으로 가른다. 새 계산은
# 없다. 이 절이 답하는 것은 "벽이 없어진 arm의 남은 실패는 무엇인가"이고, 그것을
# 알아야 실패를 e_optimization·e_approx 중 어디에 붙일지 말할 수 있다.
FAILURE_FORMS = ("basin", "accuracy_shortfall", "higher_mode", "unrecognizable",
                 "non_converged", "invalid_quadrature")
SENSITIVITY_THRESHOLDS = (0.05, 0.075, 0.10, 0.15)
SENSITIVITY_SCOPES = ("1", "2", "3", "4", "5", "6", "4-6")


def read_v2_records(outdir=None) -> list:
    """커밋된 본실험 기록을 읽는다. 사후 분석은 전부 이 파일 하나에서 나온다."""
    import json
    p = os.path.join(manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR), "p5_neural_v2_records.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — grid_v2를 먼저 돌린다(§5)")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def failure_form(rec) -> str:
    """`spurious`를 세 갈래로 가른다. 나머지 분류는 이름만 통일해 그대로 넘긴다.

    순서가 규칙의 일부다. **MAC 미달을 먼저 본다** — 형상을 알아보지 못한 셀에
    대해서는 "어느 모드를 잡았다"고 말할 수 없으므로, 최근접 모드 번호로 다시
    가르면 없는 정보를 지어내는 것이 된다."""
    oc = rec["outcome"]
    if oc == "lower_mode_basin":
        return "basin"
    if oc != "spurious":
        return oc
    if not (rec["mac"] >= mt.MAC_MIN):
        return "unrecognizable"
    if rec["matched_mode"] > rec["mode"]:
        return "higher_mode"
    return "accuracy_shortfall"


def correct_at(rec, e_lam_max: float) -> bool:
    """채점 사슬을 그대로 두고 e_λ 문턱만 갈아 끼운다(§6 민감도).

    문턱을 **바꾸자는 것이 아니다.** 사전규정 값 0.05에서 이 함수는 커밋된
    `correct`와 일치해야 하며, 회귀 테스트가 그것을 고정한다."""
    import math
    if rec["outcome"] == "invalid_quadrature" or not rec["converged"]:
        return False
    e = rec["e_lam"]
    return (rec["matched_mode"] == rec["mode"] and rec["mac"] >= mt.MAC_MIN
            and math.isfinite(e) and e <= e_lam_max)


def failure_attribution_rows(recs) -> list:
    """(arm, 조건, 모드)마다 실패 양식을 세고, `accuracy_shortfall` 셀의 방향과
    폭을 함께 남긴다.

    방향을 남기는 이유. 허용 시행함수의 Rayleigh 몫은 참조 고유값의 **상계**다.
    형상이 맞는데 Λ가 참조 위에 있으면 최소화가 아직 안 끝난 것이고, 아래에 있으면
    사영이나 구적이 하한을 깬 것이다 — 같은 칸에 담기지만 귀속이 정반대다."""
    keys, out = [], []
    for r in recs:
        k = (r["arm"], r["condition"], r["mode"])
        if k not in keys:
            keys.append(k)
    for arm, cond, mode in keys:
        sub = [r for r in recs if r["arm"] == arm and r["condition"] == cond
               and r["mode"] == mode]
        forms = [failure_form(r) for r in sub]
        acc = [r for r, f in zip(sub, forms) if f == "accuracy_shortfall"]
        row = {"problem": "P5", "arm": arm, "condition": cond, "mode": mode,
               "n": len(sub), "correct": forms.count("correct"),
               **{f: forms.count(f) for f in FAILURE_FORMS},
               "acc_above_reference": sum(1 for r in acc
                                          if r["Lam"] > r["Lam_ref"]),
               "acc_mac_min": min((r["mac"] for r in acc), default=float("nan")),
               "acc_e_lam_min": min((r["e_lam"] for r in acc), default=float("nan")),
               "acc_e_lam_max": max((r["e_lam"] for r in acc), default=float("nan"))}
        out.append(row)
    return out


def threshold_sensitivity_rows(recs) -> list:
    """사전규정 문턱을 기준으로 두고 그 주변에서 1차 결과가 얼마나 움직이는지."""
    out = []
    for arm in dict.fromkeys(r["arm"] for r in recs):
        for cond in dict.fromkeys(r["condition"] for r in recs
                                  if r["arm"] == arm):
            for scope in SENSITIVITY_SCOPES:
                ms = (range(4, 7) if scope == "4-6" else (int(scope),))
                sub = [r for r in recs if r["arm"] == arm
                       and r["condition"] == cond and r["mode"] in ms]
                if not sub:
                    continue
                for t in SENSITIVITY_THRESHOLDS:
                    k = sum(1 for r in sub if correct_at(r, t))
                    lo, hi = mt.wilson(k, len(sub))
                    out.append({"problem": "P5", "arm": arm, "condition": cond,
                                "mode_scope": scope, "e_lam_max": t,
                                "n": len(sub), "correct": k,
                                "p_correct": k / len(sub),
                                "wilson_lo": lo, "wilson_hi": hi,
                                "prespecified": t == mt.ELAM_MAX})
    return out


def failure_attribution(outdir=None) -> dict:
    """커밋된 기록에서 두 표를 만든다. GPU도, 새 실행도 필요 없다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    recs = read_v2_records(d)
    rows = failure_attribution_rows(recs)
    sens = threshold_sensitivity_rows(recs)
    manifest.write_csv(os.path.join(d, "p5_failure_attribution.csv"), rows)
    manifest.write_csv(os.path.join(d, "p5_threshold_sensitivity.csv"), sens)
    manifest.write_json(
        os.path.join(d, "manifest_p5_failure_attribution.json"), manifest.build({
            "driver": "run_p5_neural.failure_attribution",
            "source": "p5_neural_v2_records.jsonl", "n_records": len(recs),
            "posthoc": True,
            "prespecification": "unchanged — v2 §6 keeps MAC >= 0.9 and "
                                "e_lam <= 0.05; the primary result is the 0.05 row",
            "failure_forms": list(FAILURE_FORMS),
            "form_rule": "spurious splits three ways; the MAC gate is read first "
                         "because a cell whose shape matches no reference mode "
                         "cannot be said to have captured any particular one",
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "added": "2026-09-10, post-submission P5 analysis"}))
    return {"attribution": rows, "sensitivity": sens, "outdir": d}


# ── E3 감사 해상도 재표집 (사전규정 2026-09-10) ─────────────────────────────
# 대조군(45/60)이 재표집(16/60)보다 높은 것이 held-out이 아니어서인지 해상도 때문인지
# 가른다. 재표집을 유지하므로 격자 적응은 계속 차단되고 해상도만 대조군과 맞춰진다.
# 격자족을 올리면 감사 격자도 올려야 한다 — v2 §2.2의 1.5배 규칙이 깨지면 훈련·감사
# 분리 자체가 사라진다.
E3_FAMILY_AXES = ((28, 29, 30, 31, 32), (14, 15, 16), (11, 12, 13))
E3_AUDIT_GRID = (48, 24, 20)
# 드리프트 계기도 함께 올린다. 본실험의 검사격자 (40,20,15)는 E3의 감사 격자보다
# **거칠어서**, 그대로 쓰면 드리프트가 감사 격자의 부족이 아니라 검사격자의 부족을
# 잰다 — 계기의 의미가 뒤집힌다. 본실험의 비율(1.33, 1.33, 1.25)을 그대로 적용한다.
E3_CHECK_GRID = (64, 32, 25)

# ── E4 (사전규정 2026-09-14-p5-e4-prespecification.md) ──────────────────────
# E2 (b′)에서 **계기만** 바꾼다 — 훈련 격자족·예산·시드·아키텍처는 본실험 그대로다.
# 감사·검사 격자를 축마다 두 배로 올려, 모드 4의 상방 이탈이 감사 구적의 부족인지
# 함수 자체의 성질인지 가른다. 인증은 실행 전에 통과했다(제작해 최악 9.992e-15,
# 1.25배 격자 교차 8.216e-15, 문턱 1e-6). 실효 여유는 격자족 최대 (20,10,8)에 대해
# 3.0배로 v2 §2.2의 1.5배 규칙보다 강하다.
E4_AUDIT_GRID = (60, 30, 24)
E4_CHECK_GRID = (80, 40, 30)

# ── E5 (사전규정 2026-09-16-p5-e5-prespecification.md) ──────────────────────
# E3와 **훈련 격자족 하나만** 다르게 둔다 — arm·예산·감사 격자·τ·시드는 E3와 같고,
# 격자족만 본실험 기본값(16–20, 8–10, 6–8)이다. 그래서 감사 격자는 `E3_AUDIT_GRID`,
# 격자족은 `family_axes=None`(기본값)을 그대로 쓴다.
# 사다리 첫 칸이 E3의 검사격자와 같아야 드리프트가 같은 계기가 되어 두 실행이 비교된다.
E5_LADDER = ((64, 32, 25), (80, 40, 30), (96, 48, 36))

# 사다리 꼬리폭이 이 값 이하면 그 셀의 Λ는 차수에 대해 정착한 것으로 읽는다.
# E2+E4 실데이터 20셀에서 수렴 셀 최대 9.4e-05, 미수렴 셀 최소 5.3e-02 — 그 간극의
# 한가운데다. **E5에서는 계측만 하고 분류에 쓰지 않는다**(사전규정 §4).
LADDER_RESOLVED = 1e-3

# ── E6 (사전규정 2026-09-17-p5-e6-prespecification.md) ──────────────────────
# 커밋된 본실험 QR arm에 **사다리만** 더한다. 감사 격자를 0번 칸으로 두면 네 칸이
# 40,000회 쪽에서 조립한 것(E2의 감사·검사 + E4의 감사·검사)과 같아져, 꼬리폭을 두
# 예산에 대칭으로 적용할 수 있다. 20,000회 쪽에 세 칸이 없다는 것이 이 실행의 이유다.
E6_LADDER = (V2_CHECK_GRID, E4_AUDIT_GRID, E4_CHECK_GRID)

# ── E12 (사전규정 2026-09-26-p5-e12-prespecification.md) ────────────────────
# 고정 조건의 훈련 격자. 각 재표집 격자족의 **중앙 격자**이며 점수가 그 격자족의 평균과
# 정확히 같다(1,134점과 5,400점). 둘 다 감사 격자 (48,24,20)보다 성기므로 네 칸 모두에서
# 채점이 held-out으로 유지된다 — 본실험 대조군이 가졌던 교란이 여기서는 없다.
E12_FIXED_LOW = (18, 9, 7)
E12_FIXED_HIGH = (30, 15, 12)

# ── E16 정사각 단면 (사전규정 2026-09-30-p5-e16-prespecification.md) ─────────
# b = h = 0.10이라 굽힘 모드가 정확히 축퇴한 쌍으로 나온다(참조 쌍 안 간격 1e-9 이하).
# 격자족은 y·z가 같은 {7, 8, 9}다 — 정사각 단면의 최소 인증 격자 (16, 6, 6)보다 모두 촘촘하고
# 평균 점수 1,152가 기준 구조(1,134)와 맞는다. 감사 격자는 격자족 최대 (20, 9, 9)의 1.5배
# 올림이며 인증을 통과했다(제작해 최악 1.0e-11, 1.25배 격자 교차 1.0e-11). 사다리는 E6의
# 비율(4/3, 2, 8/3)을 y축 기준으로 두 축에 같이 적용한다.
E16_GEOMETRY = {"L": 1.0, "b": 0.10, "h": 0.10, "E": 1.0, "rho": 1.0, "nu": 0.29}
E16_REF = "p5_reference_e16_square"
E16_REF_LEVEL = (48, 12, 12)             # run_p5.SQUARE_LADDER[-1]
E16_FAMILY_AXES = ((16, 17, 18, 19, 20), (7, 8, 9), (7, 8, 9))
E16_AUDIT_GRID = (30, 14, 14)
E16_CHECK_GRID = (40, 19, 19)
E16_LADDER = ((40, 19, 19), (60, 28, 28), (80, 38, 38))
E16_TAU_TAG = "e16_square"


def e16_settings(arm: str) -> dict:
    """E16 arm 하나의 `grid_v2` 인자. τ는 넣지 않는다 — `e16_run`이 보정 파일에서 읽는다."""
    from .p5_stats import E16_TAGS
    if arm not in E16_TAGS:
        raise ValueError(f"E16 arm은 {sorted(E16_TAGS)} 중 하나여야 합니다: {arm!r}")
    return {"arms": (arm,), "control": False, "tag": E16_TAGS[arm],
            "geometry": dict(E16_GEOMETRY), "bc": "clamped_x0",
            "ref_name": E16_REF, "ref_level": E16_REF_LEVEL,
            "audit_grid": E16_AUDIT_GRID, "check_grid": E16_CHECK_GRID,
            "check_grids": list(E16_LADDER), "family_axes": E16_FAMILY_AXES,
            "pair_scoring": True, "gram_offdiag": True, "per_seed_span": True,
            "save_params": True}


def e16_tau(outdir=None, device=None) -> dict:
    """E16 τ 재보정(사전규정 §2.4) — 본실험 arm보다 먼저 한 번."""
    return tau_calibration(outdir=outdir, device=device, tag=E16_TAU_TAG,
                           geometry=dict(E16_GEOMETRY), audit_grid=E16_AUDIT_GRID,
                           family_axes=E16_FAMILY_AXES, heartbeat_every=HEARTBEAT_ITERS)


def e16_run(arm: str, outdir=None, device=None) -> dict:
    """E16 arm 하나. τ는 `manifest_p5_tau_e16_square.json`에서 읽고, 없으면 거부한다 —
    기준 구조의 τ로 조용히 게이트되는 일을 막는다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E16_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — e16_tau를 먼저 돌린다(사전규정 §2.4)")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e16_settings(arm))


# ── 후속 E17–E20 (사전규정 2026-10-02-p5-e17-e20-prespecification.md) ─────────
# 일곱 실행의 grid_v2 인자와 τ 출처. 기준 구조(E17·E20)는 기준 τ(5.0)를, 정사각 계열(E18·E19)은
# E16 τ(0.5)를 쓴다 — E18은 E16과 같은 게이트로 비교해야 하기 때문이다(사전규정 §2).
E18_GEOMETRY = {"L": 1.0, "b": 0.105, "h": 0.10, "E": 1.0, "rho": 1.0, "nu": 0.29}
E18_REF = "p5_reference_e18_near"
E17_MC_POINTS = 1134                     # 본실험 격자족 평균 점수(E15와 같다)
E20_MC_POINTS = 4 * E17_MC_POINTS        # 표본 비직교 O(1/√B)를 절반으로
FOLLOWUP_RUNS = ("e17_gauss1", "e17_mc250", "e18_diag", "e18_exact", "e19_budget", "e19_norm",
                 "e20r_b4")
_BASE_TAU = "manifest_p5_tau.json"


def followup_settings(name: str):
    """(grid_v2 인자, τ 매니페스트 파일명). 계기는 E15와 같다(사다리·Gram·시드별 스팬·파라미터)."""
    base = {"control": False, "tag": name, "check_grids": list(E6_LADDER),
            "gram_offdiag": True, "per_seed_span": True, "save_params": True}
    sq_tau = f"manifest_p5_tau_{E16_TAU_TAG}.json"
    if name == "e17_gauss1":
        return {**base, "arms": ("exact",), "resample_every": 1}, _BASE_TAU
    if name == "e17_mc250":
        return {**base, "arms": ("exact",), "mc_points": E17_MC_POINTS,
                "mc_resample_every": 250}, _BASE_TAU
    if name in ("e18_diag", "e18_exact"):
        k = e16_settings("diagonal" if name == "e18_diag" else "exact")
        k.update(tag=name, geometry=dict(E18_GEOMETRY), ref_name=E18_REF)
        return k, sq_tau
    if name in ("e19_budget", "e19_norm"):
        k = e16_settings("exact")
        k["tag"] = name
        if name == "e19_budget":
            k["iters"] = 2 * GRID_ITERS
        else:
            k["normalize_input"] = True
        return k, sq_tau
    if name == "e20r_b4":
        return {**base, "arms": ("diagonal",), "mc_points": E20_MC_POINTS,
                "mc_resample_every": 1, "storage": "orthogonalized",
                "batch_diag": True}, _BASE_TAU
    raise ValueError(f"후속 실행은 {FOLLOWUP_RUNS} 중 하나여야 합니다: {name!r}")


def followup_run(name: str, outdir=None, device=None) -> dict:
    """후속 실행 하나. τ 매니페스트가 없으면 거부한다."""
    import json
    kwargs, tau_file = followup_settings(name)
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, tau_file)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 이 실행의 τ를 먼저 정한다(사전규정 §2)")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **kwargs)



# ── E21 L-브래킷 (사전규정 2026-10-04-p5-e21-prespecification.md) ──────────────
# 상자가 아닌 첫 구조. 팔1 [0, 1] × [0, 0.2], 팔2 [0.8, 1] × [0.2, 0.7], 두께 0.1, x = 0 고정이며
# 재진입 모서리 (0.8, 0.2, z)에 응력 특이가 있다. 구적 격자 (n_L, n_W, n_T)는 팔의 길이 방향 단위
# 길이당 Gauss 점 수(올림), 폭, 두께의 점 수다. 최소 인증 훈련 격자 (12, 6, 6)부터 45개를 두어
# 평균 점수 1,269가 기준 구조(1,134)에 가깝다 — 기준 격자족을 그대로 쓰면 2,084점이다. 감사 격자는
# 격자족 최대 (16, 8, 8)의 1.5배 올림이며 인증을 통과했다. 사다리는 감사 격자의 4/3, 2, 8/3배다.
E21_GEOMETRY = {"kind": "L", "L": 1.0, "b": 0.20, "h": 0.10, "a": 0.50,
                "E": 1.0, "rho": 1.0, "nu": 0.29}
E21_REF = "p5_reference_e21_lbracket"
E21_REF_LEVEL = (80, 16, 8)              # run_p5.L_LADDER[-1]
E21_FAMILY_AXES = ((12, 13, 14, 15, 16), (6, 7, 8), (6, 7, 8))
E21_AUDIT_GRID = (24, 12, 12)
E21_CHECK_GRID = (32, 16, 16)
E21_LADDER = ((32, 16, 16), (48, 24, 24), (64, 32, 32))
E21_TAU_TAG = "e21_lbracket"


def e21_settings(arm: str) -> dict:
    """E21 arm 하나의 `grid_v2` 인자. τ는 넣지 않는다 — `e21_run`이 보정 파일에서 읽는다."""
    from .p5_stats import E21_TAGS
    if arm not in E21_TAGS:
        raise ValueError(f"E21 arm은 {sorted(E21_TAGS)} 중 하나여야 합니다: {arm!r}")
    return {"arms": (arm,), "control": False, "tag": E21_TAGS[arm],
            "geometry": dict(E21_GEOMETRY), "bc": "clamped_x0",
            "ref_name": E21_REF, "ref_level": E21_REF_LEVEL,
            "audit_grid": E21_AUDIT_GRID, "check_grid": E21_CHECK_GRID,
            "check_grids": list(E21_LADDER), "family_axes": E21_FAMILY_AXES,
            "pair_scoring": True, "gram_offdiag": True, "per_seed_span": True,
            "save_params": True}


def e21_tau(outdir=None, device=None) -> dict:
    """E21 τ 재보정(사전규정 §2.4) — 본실험 arm보다 먼저 한 번."""
    return tau_calibration(outdir=outdir, device=device, tag=E21_TAU_TAG,
                           geometry=dict(E21_GEOMETRY), audit_grid=E21_AUDIT_GRID,
                           family_axes=E21_FAMILY_AXES, heartbeat_every=HEARTBEAT_ITERS)


def e21_run(arm: str, outdir=None, device=None) -> dict:
    """E21 arm 하나. τ는 `manifest_p5_tau_e21_lbracket.json`에서 읽고, 없으면 거부한다 —
    다른 구조의 τ로 조용히 게이트되는 일을 막는다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — e21_tau를 먼저 돌린다(사전규정 §2.4)")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e21_settings(arm))




# ── E21b (사전규정 2026-10-06-p5-e21b-prespecification.md) ─────────────────────
# E21 정확 arm에서 훈련 격자족의 재추첨 주기만 1로 바꾼다(E17 가우스·1과 같은 조작). E21의 실패 셀은
# 훈련 격자에서 Rayleigh 하한 아래로 내려갔다 — 매 반복 다시 뽑으면 그 착취가 풀리는지를 본다.
# τ는 E21의 것(보조 분류에만 쓴다 — 판정량은 게이트 없는 정확도다).
E21B_TAG = "e21lb_exact_g1"


def e21b_settings() -> dict:
    return {**e21_settings("exact"), "tag": E21B_TAG, "resample_every": 1}


def e21b_run(outdir=None, device=None) -> dict:
    """E21b. τ는 `manifest_p5_tau_e21_lbracket.json`에서 읽고, 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e21b_settings())



# ── E21c (사전규정 2026-10-08-p5-e21c-prespecification.md) ─────────────────────
# E21 정확 arm의 격자족만 촘촘하게 한다 — 팔의 단위 길이당 16–20점으로 기준 구조와 같은 밀도다. E21b는
# 매 반복 재추첨으로도 훈련 격자 하한 위반이 남아 격자족 전체의 밀도 부족을 가리켰다. 감사 격자와 사다리는
# 1.5배 여유를 지키도록 함께 키운다(감사 (30, 15, 12)는 인증을 통과한다). τ는 E21의 것(보조).
E21C_TAG = "e21lb_exact_dense"
E21C_FAMILY_AXES = ((16, 17, 18, 19, 20), (8, 9, 10), (6, 7, 8))
E21C_AUDIT_GRID = (30, 15, 12)
E21C_CHECK_GRID = (40, 20, 16)
E21C_LADDER = ((40, 20, 16), (60, 30, 24), (80, 40, 32))


def e21c_settings() -> dict:
    return {**e21_settings("exact"), "tag": E21C_TAG, "family_axes": E21C_FAMILY_AXES,
            "audit_grid": E21C_AUDIT_GRID, "check_grid": E21C_CHECK_GRID,
            "check_grids": list(E21C_LADDER)}


def e21c_run(outdir=None, device=None) -> dict:
    """E21c. τ는 `manifest_p5_tau_e21_lbracket.json`에서 읽고, 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e21c_settings())




# ── E21d (사전규정 2026-10-09-p5-e21d-prespecification.md) ─────────────────────
# 착취의 자리(p5_exploit, 89f10a3)는 모서리 블록이 아니라 팔의 훈련 격자 절점 사이였고, 매 반복 재추첨(E21b)에서는
# 격자족 어느 격자에도 절점이 없는 공통 빈틈으로 옮겨 갔다. 절점을 고정하지 않는 구적으로 바꾼다 — E21b와
# 재추첨 주기(매 반복)는 같고 점 배치만 균등 몬테카를로다(상자의 e15b와 같은 조작, 표 11에서 60/60).
# 점 수는 E21 격자족의 평균(45개, 1,268.9점)이다. 2026-10-09 사용자 결정(모서리 블록 세분 대신).
E21D_TAG = "e21lb_exact_mc1"
E21D_MC_POINTS = 1269


def e21d_settings() -> dict:
    return {**e21_settings("exact"), "tag": E21D_TAG, "mc_points": E21D_MC_POINTS,
            "mc_resample_every": 1}


def e21d_run(outdir=None, device=None) -> dict:
    """E21d. τ는 `manifest_p5_tau_e21_lbracket.json`에서 읽고, 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e21d_settings())


# ── E21g (사전규정 2026-10-13-p5-e21g-prespecification.md) ─────────────────────
# 숨은 에너지는 막 변형이었고(사후 분해, 829f56c), 매 반복 MC의 모드 6 단계도 막 모드 8로 갔다. 두께 0.1 대 길이 1의
# 좌표 축척이 두께 방향으로 고른 함수를 고르게 한다는 후보를 입력 정규화(외접 상자로 나눔)로 시험한다. 주 판정은
# E21d에 정규화만 더한 조건(착취 없이 모드 6의 건너뜀만 남은 자리), 보조는 E21 정확 arm에 정규화만 더한 조건이다.
E21G_TAGS = {"mc": "e21lb_exact_mc1_norm", "gauss": "e21lb_exact_norm"}


def e21g_settings(arm: str) -> dict:
    if arm not in E21G_TAGS:
        raise ValueError(f"E21g 조건은 {sorted(E21G_TAGS)} 중 하나여야 합니다: {arm!r}")
    base = e21d_settings() if arm == "mc" else e21_settings("exact")
    return {**base, "tag": E21G_TAGS[arm], "normalize_input": True}


def e21g_run(arm: str, outdir=None, device=None) -> dict:
    """E21g 조건 하나. τ는 `manifest_p5_tau_e21_lbracket.json`에서 읽고, 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e21g_settings(arm))


# ── E21h (사전규정 2026-10-14-p5-e21h-prespecification.md) ─────────────────────
# E21d(원좌표)와 E21g-mc(정규화)의 모드 6 단계는 비틀림 모드 6에 이르지 못했다(시행공간은 모드 6을 담는다, E21f).
# 남은 두 후보를 모드 6 단계만 다시 풀어 시험한다 — 앞 모드 1–5는 그 실행이 저장한 망 그대로다(`grid_v2(stage_from=)`,
# 초기화·MC 표본 순서·사영이 순차 풀이와 같다). 일정 조건은 반복만 4배(StepLR 주기도 4배), 폭 조건은 새 단계의 폭만 256.
E21H_TAGS = {"sched_norm": "e21lb_s6_norm_x4", "sched_raw": "e21lb_s6_raw_x4",
             "wide_norm": "e21lb_s6_norm_w256"}
E21H_ITERS = 4 * GRID_ITERS
E21H_WIDTH = 256
E21H_CKPT_EVERY = 20_000


def e21h_settings(cond: str) -> dict:
    if cond not in E21H_TAGS:
        raise ValueError(f"E21h 조건은 {sorted(E21H_TAGS)} 중 하나여야 합니다: {cond!r}")
    base = e21d_settings() if cond == "sched_raw" else e21g_settings("mc")
    k = {**base, "tag": E21H_TAGS[cond], "save_hist": True,
         "stage_from": {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_{base['tag']}.npz"),
                        "tag": base["tag"], "arm": "exact", "cond": "mc", "n_stored": 5,
                        "width": 128, "depth": 5, "iters": GRID_ITERS}}
    if cond.startswith("sched"):
        k.update({"iters": E21H_ITERS, "checkpoint_every": E21H_CKPT_EVERY})
    else:
        k["width"] = E21H_WIDTH
    return k


def e21h_run(cond: str, outdir=None, device=None) -> dict:
    """E21h 조건 하나. τ는 E21의 것을 읽고, 앞 모드 1–5의 저장된 망이 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    k = e21h_settings(cond)
    npz = os.path.join(d, os.path.basename(k["stage_from"]["npz"]))
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz}가 없다 — 앞 모드의 저장된 망이 먼저다")
    k["stage_from"] = {**k["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **k)


# ── E23 (사전규정 2026-10-17-p5-e23-prespecification.md) ──────────────────────────
# E21h의 일정 조건은 단계를 4배로 늘리면서 학습률이 높은 구간도 4배로 늘렸다(사전규정 §5의 한계). lowtail은 감소
# 시점을 원래 일정의 6,666·13,332회에 묶고 반복만 80,000회로 늘린다 — 첫 19,998회는 E21g-mc 모드 6 단계와 같은
# 경로이고, 그 뒤는 0.16배 학습률로 이어 간다. fourier는 같은 20,000회 일정에서 새 단계의 입력에 고정 Fourier 특징을
# 붙인다(스펙트럼 편향 대응, Wang 외 2021). 둘 다 E21g-mc의 저장된 모드 1–5에서 모드 6 단계만 다시 푼다.
E23_TAGS = {"lowtail": "e21lb_s6_norm_x4lo", "fourier": "e21lb_s6_norm_ff"}
E23_MILESTONES = (GRID_ITERS // 3, 2 * (GRID_ITERS // 3))       # 6,666, 13,332
E23_FOURIER = {"features": 64, "sigma": 1.0, "seed": 2310}


def e23_settings(cond: str) -> dict:
    if cond not in E23_TAGS:
        raise ValueError(f"E23 조건은 {sorted(E23_TAGS)} 중 하나여야 합니다: {cond!r}")
    k = {**e21h_settings("sched_norm"), "tag": E23_TAGS[cond]}
    if cond == "lowtail":
        k["lr_milestones"] = E23_MILESTONES
    else:
        for name in ("iters", "checkpoint_every"):
            k.pop(name)
        k["fourier"] = dict(E23_FOURIER)
    return k


def e23_run(cond: str, outdir=None, device=None) -> dict:
    """E23 조건 하나. τ는 E21의 것을 읽고, 앞 모드 1–5의 저장된 망(E21g-mc)이 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    k = e23_settings(cond)
    npz = os.path.join(d, os.path.basename(k["stage_from"]["npz"]))
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz}가 없다 — 앞 모드의 저장된 망이 먼저다")
    k["stage_from"] = {**k["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **k)

# ── E24 (사전규정 2026-10-17-p5-e24-prespecification.md) ──────────────────────────
# 강형식 물리정보 해석기(`p5_pinn`)를 변분형의 두 조건에 그대로 얹는다. 상자는 본실험 조건(가우스 격자족 250회 재추첨,
# 원좌표, 감사 격자 30×15×12, E6 사다리, τ 5.0), L-브래킷은 E21g-mc 조건(균등 MC 1,269점 매 반복, 정규화 좌표, E21 감사
# 격자·사다리, τ 10)이다. 같은 시드 0–19, 같은 일정·망이고 바뀌는 것은 목적함수와 deflation 방식뿐이다.
E24_TAGS = {"box": "e24_pinn_box", "thick": "e24_pinn_thick", "lbracket": "e24_pinn_lb"}
# 파일럿 규칙(사전규정 §2.3): 보정 시드에서 기준 상자의 모드 1을 찾지 못하면 상자 조건을 두꺼운 상자(E13-a와 같은 설정)로 바꾼다.
E24_THICK = {"geometry": {"L": 1.0, "b": 0.2, "h": 0.10, "E": 1.0, "rho": 1.0, "nu": 0.29},
             "ref_name": "p5_reference_e13a_thick"}
E24_BOUNDARY_POINTS = 600
E24_PILOT_SEED = 9001          # 보정 전용 시드 기저(τ 보정과 같은 뜻) — 본실험 seed 0과 서로소
E24_PILOT_SEEDS = 4
E24_PILOT_MODES = 2


def e24_settings(structure: str) -> dict:
    if structure not in E24_TAGS:
        raise ValueError(f"E24 구조는 {sorted(E24_TAGS)} 중 하나여야 합니다: {structure!r}")
    common = {"arms": ("penalty",), "control": False, "solver": "strong",
              "boundary_points": E24_BOUNDARY_POINTS, "gram_offdiag": True, "per_seed_span": True,
              "save_params": True, "save_hist": True, "tag": E24_TAGS[structure]}
    if structure == "box":
        return {**common, "check_grids": list(E6_LADDER)}
    if structure == "thick":
        return {**common, "check_grids": list(E6_LADDER), "geometry": dict(E24_THICK["geometry"]),
                "ref_name": E24_THICK["ref_name"]}
    return {**e21g_settings("mc"), **common}


def _e24_tau_file(structure: str) -> str:
    return _BASE_TAU if structure in ("box", "thick") else f"manifest_p5_tau_{E21_TAU_TAG}.json"


def e24_run(structure: str, outdir=None, device=None) -> dict:
    """E24 구조 하나. τ는 그 변분형 조건의 것을 읽고(보조 분류용), 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, _e24_tau_file(structure))
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 그 구조의 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e24_settings(structure))


def e24_pilot(structure: str, outdir=None, device=None, quick: bool = False) -> dict:
    """E24 파일럿 — 보정 시드 4개(기저 9001), 모드 1–2, 단계당 20,000회. **판정에 쓰지 않는다.** 강형식이 이 조건에서
    모드 1을 찾는지(사전규정의 상자 구조 선택 규칙)와 계산 시간만 본다. 채점 규칙은 본실험과 같다(게이트 없는 정확)."""
    import math

    import numpy as np

    from ..neural import p5_pinn
    from .run_p5 import LADDER
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    launch = manifest.launch_provenance()
    k = e24_settings(structure)
    geometry = k.get("geometry")
    audit = tuple(k.get("audit_grid", AUDIT_GRID))
    n_seeds, n_modes, iters, width, depth = E24_PILOT_SEEDS, E24_PILOT_MODES, GRID_ITERS, 128, 5
    level = k.get("ref_level")
    if quick:
        n_seeds, iters, width, depth = 2, 6, 8, 2
        audit, level = (8, 5, 4), LADDER[0]
    mc = k.get("mc_points")
    out = p5_pinn.solve_p5_pinn(
        n_modes=n_modes, n_seeds=n_seeds, iters=iters, seed=E24_PILOT_SEED, width=width, depth=depth,
        device=device, geometry=geometry, audit_grid=audit,
        resample_every=(k.get("mc_resample_every", 1) if mc else V2_RESAMPLE_EVERY),
        mc_points=mc, family_axes=k.get("family_axes"), boundary_points=k["boundary_points"],
        normalize_input=bool(k.get("normalize_input")), heartbeat_every=None if quick else HEARTBEAT_ITERS)
    ref = _reference(d, k.get("ref_name", "p5_reference"))
    shapes_ref = _reference_shapes(n_modes + 2, audit, geometry, k.get("bc", "clamped_x0"), level=level)
    rows = []
    for m in range(1, n_modes + 1):
        for s in range(n_seeds):
            lam = float(out["lam"][s, m - 1])
            macs = [_mass_mac(out["shapes"][m - 1][s], R, out["w"]) for R in shapes_ref]
            j = int(np.nanargmax(macs)) + 1
            target = float(ref[m]["lam_star"])
            e = abs(lam - target) / target if math.isfinite(lam) else float("nan")
            rows.append({"structure": structure, "mode": m, "seed": s, "matched_mode": j,
                         "mac": float(macs[j - 1]), "e_lam": e, "Lam": lam, "Lam_ref": target,
                         "accurate": bool(j == m and macs[j - 1] >= 0.9 and e <= 0.05),
                         **{key: float(out[key][s, m - 1]) for key in
                            ("lam_param", "res_train", "res_audit", "bc_train", "bc_audit",
                             "orth_train", "orth_audit")},
                         "seconds": float(out["stage_seconds"][m - 1])})
    manifest.write_csv(os.path.join(d, f"p5_pilot_e24_{structure}.csv"), rows)
    manifest.write_json(os.path.join(d, f"manifest_p5_pilot_e24_{structure}.json"), manifest.build({
        "driver": "run_p5_neural.e24_pilot", "structure": structure, "launch": launch, "quick": bool(quick),
        "seed_base": E24_PILOT_SEED, "n_seeds": n_seeds, "n_modes": n_modes, "iters": iters,
        "settings": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in k.items()
                     if kk not in ("geometry",)},
        "geometry": _geometry_record(geometry), "role": "pilot on calibration seeds; not used for judgment"}))
    return {"rows": rows, "outdir": d}

# ── E25 (사전규정 2026-10-17-p5-e25-prespecification.md) ──────────────────────────
# T-브래킷 — 검증 프로토콜(원고 3.5절)이 정해진 뒤에 들인 구조에서 프로토콜을 전향적으로 시험한다. 원 설정(가우스 격자족
# 250회 재추첨, 원좌표)의 대각·정확 arm이 "전", 프로토콜 arm(매 반복 균등 MC, 정확 사영, 정규화 좌표)이 "후"이고, 프로토콜
# arm에서 단계가 건너뛰면 그 단계만 4배 일정으로 다시 푸는 처방(§2.4)을 정해 둔다. 격자족은 날개가 짧아 단위 길이당 13–17점
# (12점은 인증 1.27 × 10⁻⁶로 불합격), 감사 격자는 그 1.5배 (26, 12, 12)다(제작해 네 장 1.35 × 10⁻⁷).
E25_GEOMETRY = {"kind": "T", "L": 1.0, "b": 0.20, "h": 0.10, "a": 0.25, "E": 1.0, "rho": 1.0, "nu": 0.29}
E25_REF = "p5_reference_e25_tbracket"
E25_REF_LEVEL = (80, 16, 8)              # run_p5.L_LADDER[-1]
E25_FAMILY_AXES = ((13, 14, 15, 16, 17), (6, 7, 8), (6, 7, 8))
E25_AUDIT_GRID = (26, 12, 12)
E25_CHECK_GRID = (35, 16, 16)
E25_LADDER = ((35, 16, 16), (52, 24, 24), (70, 32, 32))
E25_MC_POINTS = 1651                     # 격자족 45개의 평균 점수 1,650.6
E25_TAGS = {"orig_diag": "e25tb_diag", "orig_exact": "e25tb_exact", "protocol": "e25tb_protocol"}
E25_REMEDY_MIN_STAGE = 4                 # 처방은 단계 4–6에만(그 앞은 예산 밖, 사전규정 §2.4)


def e25_settings(arm: str) -> dict:
    if arm not in E25_TAGS:
        raise ValueError(f"E25 arm은 {sorted(E25_TAGS)} 중 하나여야 합니다: {arm!r}")
    base = {"control": False, "tag": E25_TAGS[arm], "geometry": dict(E25_GEOMETRY), "bc": "clamped_x0",
            "ref_name": E25_REF, "ref_level": E25_REF_LEVEL, "audit_grid": E25_AUDIT_GRID,
            "check_grid": E25_CHECK_GRID, "check_grids": list(E25_LADDER), "family_axes": E25_FAMILY_AXES,
            "pair_scoring": True, "gram_offdiag": True, "per_seed_span": True, "save_params": True}
    if arm == "orig_diag":
        return {**base, "arms": ("diagonal",)}
    if arm == "orig_exact":
        return {**base, "arms": ("exact",)}
    return {**base, "arms": ("exact",), "mc_points": E25_MC_POINTS, "mc_resample_every": 1,
            "normalize_input": True, "save_hist": True}


def e25_remedy_settings(k: int, step: int) -> dict:
    """처방(사전규정 §2.4). step 1: 프로토콜 arm의 저장 모드 1–(k−1)에서 단계 k만 4배 일정으로 다시 푼다. step 2(k < 6):
    처방한 단계까지의 망(모드 1–k)에서 단계 k+1–6을 원래 일정으로 이어 푼다 — 저장 단계마다 뽑던 표본 수만큼 난수를 넘긴다."""
    if not E25_REMEDY_MIN_STAGE <= int(k) <= 6:
        raise ValueError(f"처방은 단계 {E25_REMEDY_MIN_STAGE}–6에만 정해 두었다: k = {k}")
    k = int(k)
    p = {kk: v for kk, v in e25_settings("protocol").items()}
    if step == 1:
        return {**p, "tag": f"e25tb_remedy_k{k}", "n_modes": k, "iters": 4 * GRID_ITERS,
                "checkpoint_every": 20_000,
                "stage_from": {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_{E25_TAGS['protocol']}.npz"),
                               "tag": E25_TAGS["protocol"], "arm": "exact", "cond": "mc", "n_stored": k - 1,
                               "width": 128, "depth": 5, "iters": GRID_ITERS}}
    if step == 2:
        if k == 6:
            raise ValueError("k = 6이면 이어 풀 단계가 없다")
        return {**p, "tag": f"e25tb_remedy_k{k}_rest",
                "stage_from": {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_e25tb_remedy_k{k}.npz"),
                               "tag": f"e25tb_remedy_k{k}", "arm": "exact", "cond": "mc", "n_stored": k,
                               "width": 128, "depth": 5, "iters": [GRID_ITERS] * (k - 1) + [4 * GRID_ITERS]}}
    raise ValueError(f"step은 1 또는 2다: {step!r}")


def _e25_tau(d) -> float:
    import json
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E25는 L-브래킷(E21)의 τ를 보조 분류에만 쓴다")
    return float(json.load(open(p, encoding="utf-8"))["tau"])


def e25_run(arm: str, outdir=None, device=None) -> dict:
    """E25 arm 하나. 참조(`p5_reference_e25_tbracket.csv`)와 E21 τ 파일이 없으면 거부한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    if not os.path.exists(os.path.join(d, f"{E25_REF}.csv")):
        raise FileNotFoundError(f"{E25_REF}.csv가 없다 — 참조 계산이 먼저다")
    return grid_v2(outdir=d, device=device, tau=_e25_tau(d), **e25_settings(arm))


def e25_remedy_run(k: int, step: int, outdir=None, device=None) -> dict:
    """E25 처방 한 단계. 앞 단계의 저장 망 파일이 출력 폴더에 없으면 거부한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    kw = e25_remedy_settings(k, step)
    npz = os.path.join(d, os.path.basename(kw["stage_from"]["npz"]))
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz}가 없다 — 처방의 앞 단계가 먼저다")
    kw["stage_from"] = {**kw["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=_e25_tau(d), **kw)


def e25_reference(outdir=None) -> dict:
    """E25 참조 — L-브래킷과 같은 사다리와 규칙(`run_p5.structure_reference`), 형상 인증은 감사 격자에서."""
    from . import run_p5
    return run_p5.structure_reference("e25_tbracket", {k: v for k, v in E25_GEOMETRY.items()},
                                      ladder=run_p5.L_LADDER, cert_grid=E25_AUDIT_GRID, outdir=outdir)

# ── E26 (사전규정 2026-10-18-p5-e26-prespecification.md) ──────────────────────────
# r2.0-interim 검토(2026-10-18)의 A1·A3. (a) 프로토콜 전/후를 프로토콜을 돌리지 않았던 네 구조로 넓힌다 — 원 설정과 같은
# 설정에 훈련 구적만 매 반복 균등 MC(격자족 평균 점수)로 바꾸고 정확 사영·원좌표를 둔다(기준 상자의 E15b와 같은 조작).
# (b) L-브래킷의 프로토콜 경로를 정규화 없이 세우기 위해, 원좌표 MC 실행(E21d)의 저장 모드 1–5에서 모드 6 단계를 고정
# Fourier 특징(E23과 같은 설정)으로 다시 푼다. (c) 재풀이 일관성 검사의 음성 대조 — 모드 6이 20/20 정확이었던 기준 구조의
# 원좌표 MC 실행(E15-b)에서 같은 재풀이를 한다(정답 단계에서 Λ가 허용오차보다 내려가지 않아야 검사가 쓸모 있다).
# (d) 프로토콜 경로를 원좌표로 바꾼 뒤의 전향 시험 — T-브래킷의 E25 프로토콜 arm(정규화)과 같고 정규화만 뺀다(같은 시드).
E26_TAGS = {"thick": "e26_thick_mc", "clamped2": "e26_cc_mc", "near": "e26_near_mc", "bilayer": "e26_bi_mc",
            "fourier_raw": "e21lb_s6_raw_ff", "fourier_base": "e15b_s6_raw_ff", "tbracket_raw": "e26_tb_raw_mc"}
E26_MC_POINTS = {"thick": 1134, "clamped2": 1134, "near": 1152, "bilayer": 1134}   # 각 격자족의 평균 점수
E26_TAU = {"thick": _BASE_TAU, "clamped2": _BASE_TAU, "near": "manifest_p5_tau_e16_square.json",
           "bilayer": "manifest_p5_tau_e22_bilayer.json", "fourier_raw": "manifest_p5_tau_e21_lbracket.json",
           "fourier_base": _BASE_TAU, "tbracket_raw": "manifest_p5_tau_e21_lbracket.json"}


def e26_settings(cond: str) -> dict:
    if cond not in E26_TAGS:
        raise ValueError(f"E26 조건은 {sorted(E26_TAGS)} 중 하나여야 합니다: {cond!r}")
    if cond == "fourier_raw":
        k = {**e21h_settings("sched_raw"), "tag": E26_TAGS[cond], "fourier": dict(E23_FOURIER)}
        for name in ("iters", "checkpoint_every"):
            k.pop(name)
        return k
    if cond == "tbracket_raw":
        k = {**e25_settings("protocol"), "tag": E26_TAGS[cond]}
        k.pop("normalize_input")
        return k
    if cond == "fourier_base":
        return {"arms": ("exact",), "control": False, "tag": E26_TAGS[cond], "mc_points": E17_MC_POINTS,
                "mc_resample_every": 1, "check_grids": list(E6_LADDER), "gram_offdiag": True, "per_seed_span": True,
                "save_params": True, "save_hist": True, "fourier": dict(E23_FOURIER),
                "stage_from": {"npz": os.path.join(manifest.P5_OUTDIR, "p5_params_e15b.npz"), "tag": "e15b",
                               "arm": "exact", "cond": "mc", "n_stored": 5, "width": 128, "depth": 5,
                               "iters": GRID_ITERS}}
    mc = {"mc_points": E26_MC_POINTS[cond], "mc_resample_every": 1, "arms": ("exact",), "control": False,
          "tag": E26_TAGS[cond], "gram_offdiag": True, "per_seed_span": True, "save_params": True}
    if cond == "thick":
        return {**mc, **{k_: (dict(v) if isinstance(v, dict) else v) for k_, v in E24_THICK.items()},
                "check_grids": list(E6_LADDER)}
    if cond == "clamped2":
        return {**mc, "bc": "two_face", "ref_name": "p5_reference_e13b_clamped2", "check_grids": list(E6_LADDER)}
    base = (e16_settings("exact") if cond == "near" else e22_settings("exact"))
    if cond == "near":
        base = {**base, "geometry": dict(E18_GEOMETRY), "ref_name": E18_REF}
    return {**base, **mc}


def e26_run(cond: str, outdir=None, device=None) -> dict:
    """E26 조건 하나. τ는 그 구조의 보정 파일에서 읽고(보조 분류용), fourier_raw는 E21d의 저장 망이 있어야 한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, E26_TAU[cond])
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 그 구조의 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    k = e26_settings(cond)
    if cond == "tbracket_raw" and not os.path.exists(os.path.join(d, f"{E25_REF}.csv")):
        raise FileNotFoundError(f"{E25_REF}.csv가 없다 — T-브래킷 참조가 먼저다")
    if "stage_from" in k:
        npz = os.path.join(d, os.path.basename(k["stage_from"]["npz"]))
        if not os.path.exists(npz):
            raise FileNotFoundError(f"{npz}가 없다 — 앞 모드의 저장된 망이 먼저다")
        k["stage_from"] = {**k["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **k)

# ── E26e (사전규정 2026-10-22-p5-e26e-prespecification.md) ───────────────────────────
# 프로토콜의 최적화 단계(재풀이 일관성 검사)를 E26b 첫 통과에 적용한다. 첫 통과에서 5/20 이상이 더 높은 모드에 자리 잡은
# 단계 k마다, 그 실행의 저장 모드 1–(k−1)에서 단계 k만 고정 Fourier 특징(E23과 같은 B)으로 다시 푼다(n_modes = k, 원래 일정).
# 이어지는 단계는 풀지 않는다 — 판독은 그 단계의 재풀이 검사(Λ 감소 > 0.05)와 정확의 전후다.
# 어느 단계를 다시 풀지는 `p5_stats.e26e_rule_stages`가 첫 통과 기록에서 정한다(5/20 이상이 더 높은 모드).


def e26e_settings(cond: str, k: int) -> dict:
    if cond not in ("thick", "clamped2", "near", "bilayer"):
        raise ValueError(f"E26e 구조는 E26b의 넷 중 하나여야 합니다: {cond!r}")
    k = int(k)
    if not 2 <= k <= 6:
        raise ValueError(f"재풀이 단계는 2–6이다: k = {k}")
    base = e26_settings(cond)
    return {**base, "tag": f"{base['tag']}_s{k}_ff", "n_modes": k, "fourier": dict(E23_FOURIER), "save_hist": True,
            "stage_from": {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_{base['tag']}.npz"), "tag": base["tag"],
                           "arm": "exact", "cond": "mc", "n_stored": k - 1, "width": 128, "depth": 5,
                           "iters": GRID_ITERS}}


def e26e_run(cond: str, k: int, outdir=None, device=None) -> dict:
    """E26e 재풀이 하나. τ는 그 구조의 것을 읽고, 첫 통과의 저장 망이 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, E26_TAU[cond])
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 그 구조의 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    k_ = e26e_settings(cond, k)
    npz = os.path.join(d, os.path.basename(k_["stage_from"]["npz"]))
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz}가 없다 — 첫 통과의 저장된 망이 먼저다")
    k_["stage_from"] = {**k_["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **k_)

# ── E27 (사전규정 2026-10-23-p5-e27-prespecification.md) ───────────────────────────
# 검토 r2(2026-10-23) §3.1·§3.2. (a) 양단 고정 상자에서 프로토콜을 끝까지 전파한다. 단계 1은 첫 통과(E26b), 단계 2는 첫
# 통과와 Fourier 재풀이(E26e) 가운데 시드마다 고른 것이다. 단계 3–6은 고른 앞 단계들에서 네 경로로 푼다 — 원 설정(first),
# Fourier 입력(ff), 고율 구간을 그대로 두고 저율 구간을 늘린 80,000회 일정(long, E23의 lowtail과 같은 감소 시점), 정규화
# 좌표(norm). 시드마다 사다리가 안정한(꼬리 폭 ≤ 10⁻³) 후보 가운데 감사 Λ가 가장 낮은 것을 고르고(`p5_stats.e27_select_stage`,
# 참조를 읽지 않는다), 고른 망이 다음 단계의 저장 단계가 된다(`p5_neural.MixedStoredStage`). 단계 3에는 재초기화(reinit)를
# 보조 경로로 더한다 — 고르는 데 쓰지 않는다. 표본 순서는 E26e와 같이 저장 단계마다 20,000회분 난수를 넘겨, 단계 k의 모든
# 경로가 첫 통과의 단계 k와 같은 표본을 본다. (b) 정사각 단면의 프로토콜 첫 통과(E26b와 같은 조작, 군집 채점).
E27_PATHS = ("first", "ff", "long", "norm")
E27_AUX_PATHS = {3: ("reinit",)}
E27_REINIT_OFFSET = 500
E27_CC_BASE = "e26_cc_mc"
E27_SELECTION = "p5_e27_cc_selection.json"
E27_SQ_TAG = "e27_sq_mc"
E27_SQ_MC_POINTS = 1152                  # 정사각 격자족(16–20, 7–9, 7–9)의 평균 점수


def e27_tag(k: int, path: str) -> str:
    return f"e27_cc_s{int(k)}_{path}"


def e27_candidates(k: int) -> dict:
    """단계 k에서 고를 후보 — 경로 이름 → 실행 태그(순서가 동률을 깬다)."""
    k = int(k)
    if k == 1:
        return {"first": E27_CC_BASE}
    if k == 2:
        return {"first": E27_CC_BASE, "ff": f"{E27_CC_BASE}_s2_ff"}
    return {p_: e27_tag(k, p_) for p_ in E27_PATHS}


def e27_path_spec(path: str) -> dict:
    """저장 단계로 되살릴 때의 망 구조와 입력 축척."""
    return {"fourier": dict(E23_FOURIER) if path == "ff" else None, "normalize": path == "norm"}


def e27_stage_sources(selection: dict, j: int, data_dir: str | None = None, candidates=None) -> list:
    """고른 단계 j의 출처 목록 — 경로마다 그 실행의 망 파일과 시드들. `candidates`는 단계 → {경로: 태그} 함수(기본 E27)."""
    d = manifest.P5_OUTDIR if data_dir is None else data_dir
    chosen = selection["stages"][str(int(j))]["choice"]
    cands = (e27_candidates if candidates is None else candidates)(j)
    groups = {}
    for s_, p_ in chosen.items():
        if p_ not in cands:
            raise ValueError(f"단계 {j}의 선택 {p_!r}는 후보가 아니다")
        groups.setdefault(p_, []).append(int(s_))
    return [{"npz": os.path.join(d, f"p5_params_{cands[p_]}.npz"), "arm": "exact", "cond": "mc", "mode": int(j),
             **e27_path_spec(p_), "seeds": sorted(groups[p_]), "path": p_}
            for p_ in cands if p_ in groups]


def e27_settings(k: int, path: str, selection: dict, data_dir: str | None = None) -> dict:
    k = int(k)
    if not 3 <= k <= 6:
        raise ValueError(f"E27 전파 단계는 3–6이다: k = {k}")
    if path not in E27_PATHS + E27_AUX_PATHS.get(k, ()):
        raise ValueError(f"단계 {k}의 경로는 {E27_PATHS + E27_AUX_PATHS.get(k, ())} 중 하나다: {path!r}")
    missing = [j for j in range(1, k) if str(j) not in selection.get("stages", {})]
    if missing:
        raise ValueError(f"앞 단계 {missing}의 선택이 없다 — 그 단계의 경로들이 먼저다")
    base = e26_settings("clamped2")
    kw = {**base, "tag": e27_tag(k, path), "n_modes": k, "save_hist": True,
          "stage_from": {"mixed": [e27_stage_sources(selection, j, data_dir) for j in range(1, k)],
                         "width": 128, "depth": 5, "iters": GRID_ITERS}}
    if path == "ff":
        kw["fourier"] = dict(E23_FOURIER)
    elif path == "long":
        kw.update({"iters": 4 * GRID_ITERS, "lr_milestones": E23_MILESTONES})
    elif path == "norm":
        kw["normalize_input"] = True
    elif path == "reinit":
        kw["init_seed_offset"] = E27_REINIT_OFFSET
    return kw


def _e27_selection(d: str) -> dict:
    import json
    p = os.path.join(d, E27_SELECTION)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — `e27_select_run`으로 앞 단계를 먼저 고른다")
    return json.load(open(p, encoding="utf-8"))


def e27_select_run(k: int, outdir=None) -> dict:
    """단계 k의 후보 기록에서 시드마다 고르고 선택 파일에 더한다(이미 있으면 같은지 확인만 한다)."""
    import json
    from . import p5_stats
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    k = int(k)
    recs = {p_: p5_stats.load(d, t) for p_, t in e27_candidates(k).items()}
    choice = p5_stats.e27_select_stage(recs, k)
    p = os.path.join(d, E27_SELECTION)
    sel = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"stages": {}}
    entry = {"choice": {str(s_): v for s_, v in sorted(choice.items())},
             "candidates": e27_candidates(k), "rule": p5_stats.E27_RULE}
    old = sel["stages"].get(str(k))
    if old is not None:
        if old["choice"] != entry["choice"]:
            raise ValueError(f"단계 {k}의 선택이 이미 있고 다르다 — 덮어쓰지 않는다")
        return sel
    sel["stages"][str(k)] = entry
    manifest.write_json(p, sel)
    manifest.write_json(os.path.join(d, "manifest_" + E27_SELECTION), manifest.build({
        "driver": "run_p5_neural.e27_select_run", "stage": k, "reads": sorted(e27_candidates(k).values()),
        "rule": p5_stats.E27_RULE, "prespecification": "docs/paper4-p5/2026-10-23-p5-e27-prespecification.md"}))
    return sel


def e27_run(k: int, path: str, outdir=None, device=None) -> dict:
    """E27 전파의 단계 k를 경로 하나로. τ는 기준 구조의 것(보조 분류), 앞 단계의 선택과 망 파일이 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, E26_TAU["clamped2"])
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    kw = e27_settings(k, path, _e27_selection(d), data_dir=d)
    for st in kw["stage_from"]["mixed"]:
        for src in st:
            if not os.path.exists(src["npz"]):
                raise FileNotFoundError(f"{src['npz']}가 없다 — 그 경로의 실행이 먼저다")
    return grid_v2(outdir=d, device=device, tau=tau, **kw)


def e27_square_settings() -> dict:
    """정사각 단면의 프로토콜 첫 통과 — E16 정확 arm의 설정에서 훈련 구적만 매 반복 균등 MC(격자족 평균 점수)로."""
    return {**e16_settings("exact"), "tag": E27_SQ_TAG, "mc_points": E27_SQ_MC_POINTS, "mc_resample_every": 1,
            "arms": ("exact",), "control": False}


def e27_square_run(outdir=None, device=None) -> dict:
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E16_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 정사각 단면의 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e27_square_settings())

# ── E28 (사전규정 2026-10-23-p5-e28-prespecification.md) ───────────────────────────
# 종합 검토(2026-10-23)와 그 응답의 세 요건. (a) 양단 고정 상자에서 참조를 실행 중 전혀 읽지 않는 end-to-end 프로토콜 — 단계
# 1부터 6까지 단계마다 네 경로(first·ff·long·norm)로 풀고 사다리가 안정한 후보 가운데 감사 Λ가 가장 낮은 해를 시드마다 골라
# 다음 단계로 넘긴다(E27a를 자료 전에 대체). 단계 1의 first는 첫 통과(E26b)의 단계 1과 같은 계산이라 그대로 쓴다. 단계 2·3에는
# 같은 예산 비교를 위해 재초기화 4회(reinit1–4, 초기화 난수 +500·r)를 보조로 돈다(고르는 데 쓰지 않는다). (b) L-브래킷 단계
# 6(E21d 저장 모드, 원좌표)에서 재초기화 4회 — "다시 시작해 최저 Λ를 고른다"는 단순 대안을 4배 일정과 같은 예산으로 비교한다.
# (c) 근축퇴 단면에서 단계 5의 선택(첫 통과 vs Fourier 재풀이)을 단계 6으로 전파한다(first·ff). (d) T-브래킷 원좌표 첫 통과
# (E26d, 120/120)의 단계 1–6을 Fourier로 다시 푼다 — 정답 단계에서 거짓 양성을 보는 음성 대조. (e) 두꺼운 상자의 강형식에서
# 자유면 견인력 가중을 10으로 — 원인(표면 일) → 처방 → 검증.
E28_PATHS = ("first", "ff", "long", "norm")
E28_REINIT = {2: (1, 2, 3, 4), 3: (1, 2, 3, 4)}
E28_REINIT_OFFSET = 500
E28_SELECTION = "p5_e28_cc_selection.json"
E28_NEAR_SELECTION = "p5_e28_nr_selection.json"
E28_LB_REINIT = (1, 2, 3, 4)
E28_TRACTION_WEIGHT = 10.0


def e28_tag(k: int, path: str) -> str:
    return f"e28_cc_s{int(k)}_{path}"


def e28_candidates(k: int) -> dict:
    """E28a 단계 k의 후보(경로 → 태그, 순서가 동률을 깬다). 단계 1의 first는 첫 통과(E26b)다."""
    k = int(k)
    return {p_: (E27_CC_BASE if (k == 1 and p_ == "first") else e28_tag(k, p_)) for p_ in E28_PATHS}


def e28_aux_paths(k: int) -> tuple:
    return tuple(f"reinit{r}" for r in E28_REINIT.get(int(k), ()))


def _apply_path(kw: dict, path: str) -> dict:
    """경로의 차이: Fourier 입력, 80,000회 저율 연장 일정, 정규화 좌표, 재초기화(+500·r)."""
    if path == "ff":
        kw["fourier"] = dict(E23_FOURIER)
    elif path == "long":
        kw.update({"iters": 4 * GRID_ITERS, "lr_milestones": E23_MILESTONES})
    elif path == "norm":
        kw["normalize_input"] = True
    elif path.startswith("reinit"):
        kw["init_seed_offset"] = E28_REINIT_OFFSET * int(path[len("reinit"):])
    elif path != "first":
        raise ValueError(f"모르는 경로: {path!r}")
    return kw


def e28_settings(k: int, path: str, selection: dict, data_dir: str | None = None) -> dict:
    k = int(k)
    if not 1 <= k <= 6:
        raise ValueError(f"E28a 단계는 1–6이다: k = {k}")
    allowed = tuple(p_ for p_ in E28_PATHS if not (k == 1 and p_ == "first")) + e28_aux_paths(k)
    if path not in allowed:
        raise ValueError(f"단계 {k}의 경로는 {allowed} 중 하나다: {path!r}")
    missing = [j for j in range(1, k) if str(j) not in selection.get("stages", {})]
    if missing:
        raise ValueError(f"앞 단계 {missing}의 선택이 없다 — 그 단계의 경로들이 먼저다")
    kw = {**e26_settings("clamped2"), "tag": e28_tag(k, path), "n_modes": k, "save_hist": True}
    if k >= 2:
        kw["stage_from"] = {"mixed": [e27_stage_sources(selection, j, data_dir, candidates=e28_candidates)
                                      for j in range(1, k)],
                            "width": 128, "depth": 5, "iters": GRID_ITERS}
    return _apply_path(kw, path)


def _select_into(path_json: str, k: int, candidates, d: str, prespec: str) -> dict:
    """단계 k의 후보 기록에서 시드마다 고르고 선택 파일에 더한다(이미 있으면 같은지 확인만 한다)."""
    import json
    from . import p5_stats
    recs = {p_: p5_stats.load(d, t) for p_, t in candidates(k).items()}
    choice = p5_stats.e27_select_stage(recs, k)
    p = os.path.join(d, path_json)
    sel = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"stages": {}}
    entry = {"choice": {str(s_): v for s_, v in sorted(choice.items())}, "candidates": candidates(k),
             "rule": p5_stats.E27_RULE}
    old = sel["stages"].get(str(k))
    if old is not None:
        if old["choice"] != entry["choice"]:
            raise ValueError(f"단계 {k}의 선택이 이미 있고 다르다 — 덮어쓰지 않는다")
        return sel
    sel["stages"][str(k)] = entry
    manifest.write_json(p, sel)
    manifest.write_json(os.path.join(d, "manifest_" + path_json), manifest.build({
        "driver": "run_p5_neural._select_into", "stage": int(k), "reads": sorted(candidates(k).values()),
        "rule": p5_stats.E27_RULE, "prespecification": prespec}))
    return sel


def e28_select_run(k: int, outdir=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    return _select_into(E28_SELECTION, int(k), e28_candidates, d,
                        "docs/paper4-p5/2026-10-23-p5-e28-prespecification.md")


def _read_selection(d: str, name: str) -> dict:
    import json
    p = os.path.join(d, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 앞 단계를 먼저 고른다")
    return json.load(open(p, encoding="utf-8"))


def _check_sources(kw: dict) -> dict:
    for st in kw.get("stage_from", {}).get("mixed", []):
        for src in st:
            if not os.path.exists(src["npz"]):
                raise FileNotFoundError(f"{src['npz']}가 없다 — 그 경로의 실행이 먼저다")
    return kw


def _tau_of(d: str, name: str) -> float:
    import json
    p = os.path.join(d, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — τ 보정이 먼저다")
    return float(json.load(open(p, encoding="utf-8"))["tau"])


def e28_run(k: int, path: str, outdir=None, device=None) -> dict:
    """E28a 단계 k를 경로 하나로. 단계 2 이상은 앞 단계의 선택과 망 파일이 있어야 한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, E26_TAU["clamped2"])
    sel = _read_selection(d, E28_SELECTION) if int(k) >= 2 else {"stages": {}}
    return grid_v2(outdir=d, device=device, tau=tau, **_check_sources(e28_settings(k, path, sel, data_dir=d)))


def e28_lb_settings(r: int) -> dict:
    """E28b — L-브래킷 단계 6 재초기화 r(1–4): E21d의 저장 모드 1–5에서 원래 일정·원좌표로, 초기화 난수 +500·r."""
    r = int(r)
    if r not in E28_LB_REINIT:
        raise ValueError(f"재초기화 번호는 {E28_LB_REINIT} 중 하나다: {r}")
    k = {**e21h_settings("sched_raw"), "tag": f"e28_lb_s6_reinit{r}"}
    for name in ("iters", "checkpoint_every"):
        k.pop(name)
    return _apply_path(k, f"reinit{r}")


def e28_lb_run(r: int, outdir=None, device=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    k = e28_lb_settings(r)
    npz = os.path.join(d, os.path.basename(k["stage_from"]["npz"]))
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz}가 없다 — E21d의 저장된 망이 먼저다")
    k["stage_from"] = {**k["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **k)


def e28_near_candidates(k: int) -> dict:
    """E28c — 근축퇴 단면. 단계 1–4는 첫 통과, 단계 5는 첫 통과와 Fourier 재풀이(E26e), 단계 6은 전파한 first·ff."""
    k = int(k)
    if k <= 4:
        return {"first": E26_TAGS["near"]}
    if k == 5:
        return {"first": E26_TAGS["near"], "ff": f"{E26_TAGS['near']}_s5_ff"}
    return {"first": "e28_nr_s6_first", "ff": "e28_nr_s6_ff"}


def e28_near_settings(path: str, selection: dict, data_dir: str | None = None) -> dict:
    if path not in ("first", "ff"):
        raise ValueError(f"E28c 경로는 first·ff다: {path!r}")
    missing = [j for j in range(1, 6) if str(j) not in selection.get("stages", {})]
    if missing:
        raise ValueError(f"앞 단계 {missing}의 선택이 없다")
    kw = {**e26_settings("near"), "tag": f"e28_nr_s6_{path}", "n_modes": 6, "save_hist": True,
          "stage_from": {"mixed": [e27_stage_sources(selection, j, data_dir, candidates=e28_near_candidates)
                                   for j in range(1, 6)], "width": 128, "depth": 5, "iters": GRID_ITERS}}
    return _apply_path(kw, path)


def e28_near_select_run(k: int, outdir=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    return _select_into(E28_NEAR_SELECTION, int(k), e28_near_candidates, d,
                        "docs/paper4-p5/2026-10-23-p5-e28-prespecification.md")


def e28_near_run(path: str, outdir=None, device=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, E26_TAU["near"])
    sel = _read_selection(d, E28_NEAR_SELECTION)
    return grid_v2(outdir=d, device=device, tau=tau, **_check_sources(e28_near_settings(path, sel, data_dir=d)))


def e28_tb_settings(k: int) -> dict:
    """E28d — T-브래킷 원좌표 첫 통과(E26d)의 단계 k를 같은 저장 모드에서 Fourier로 다시 푼다(단계 1은 저장 모드 없음)."""
    k = int(k)
    if not 1 <= k <= 6:
        raise ValueError(f"단계는 1–6이다: k = {k}")
    base = e26_settings("tbracket_raw")
    kw = {**base, "tag": f"e28_tb_s{k}_ff", "n_modes": k, "save_hist": True, "fourier": dict(E23_FOURIER)}
    if k >= 2:
        kw["stage_from"] = {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_{base['tag']}.npz"),
                            "tag": base["tag"], "arm": "exact", "cond": "mc", "n_stored": k - 1,
                            "width": 128, "depth": 5, "iters": GRID_ITERS}
    return kw


def e28_tb_run(k: int, outdir=None, device=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, E26_TAU["tbracket_raw"])
    if not os.path.exists(os.path.join(d, f"{E25_REF}.csv")):
        raise FileNotFoundError(f"{E25_REF}.csv가 없다 — T-브래킷 참조가 먼저다")
    kw = e28_tb_settings(k)
    if "stage_from" in kw:
        npz = os.path.join(d, os.path.basename(kw["stage_from"]["npz"]))
        if not os.path.exists(npz):
            raise FileNotFoundError(f"{npz}가 없다 — E26d의 저장된 망이 먼저다")
        kw["stage_from"] = {**kw["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **kw)


def e28_pinn_settings() -> dict:
    """E28e — 두꺼운 상자의 강형식(E24 thick과 같은 설정)에서 자유면 견인력 가중만 10으로."""
    return {**e24_settings("thick"), "tag": "e28_pinn_thick_w10", "traction_weight": E28_TRACTION_WEIGHT}


def e28_pinn_run(outdir=None, device=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, _e24_tau_file("thick"))
    return grid_v2(outdir=d, device=device, tau=tau, **e28_pinn_settings())

# ── E29 (사전규정 2026-10-30-p5-e29-prespecification.md) ───────────────────────────
# CMAME 요건(2026-10-30) §2.2 "한 단계 더". 순서 검사는 뒤 단계가 건너뛴 모드를 찾아야 울리므로 마지막 단계(6)를 볼 수
# 없었다. 첫 통과가 저장한 모드 1–6에서 단계 7만 그 첫 통과의 설정으로 푼다(n_modes = 7, 원래 20,000회 일정, 저장 단계가
# 뽑았을 표본만큼 난수를 넘김 — 7모드 첫 통과의 단계 7과 같은 계산). 단계 7은 보고하지 않는다. 단계 7의 출력은 저장
# 스팬 S_6 ⊇ S_5에 직교하므로 단계 6의 재풀이 경쟁자다. L-브래킷(E21d 원좌표 MC, 단계 6이 20시드 모두 모드 8)과 근축퇴
# 단면(E26b, 단계 6의 부정확은 모두 건너뛴 모드 5를 찾은 낮은 모드 — 마지막 단계의 음성 대조).
E29_TAGS = {"lbracket": "e29_lb_s7", "near": "e29_nr_s7"}
E29_TAU = {"lbracket": f"manifest_p5_tau_{E21_TAU_TAG}.json", "near": E26_TAU["near"]}
E29_STAGE = 7


def e29_settings(structure: str) -> dict:
    if structure not in E29_TAGS:
        raise ValueError(f"E29 구조는 {sorted(E29_TAGS)} 중 하나여야 합니다: {structure!r}")
    base = e21d_settings() if structure == "lbracket" else e26_settings("near")
    return {**base, "tag": E29_TAGS[structure], "n_modes": E29_STAGE, "save_hist": True,
            "stage_from": {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_{base['tag']}.npz"), "tag": base["tag"],
                           "arm": "exact", "cond": "mc", "n_stored": E29_STAGE - 1, "width": 128, "depth": 5,
                           "iters": GRID_ITERS}}


def e29_run(structure: str, outdir=None, device=None) -> dict:
    """E29 구조 하나. τ는 그 구조의 것을 읽고(보조 분류), 참조에 모드 7이 없거나 첫 통과의 저장 망이 없으면 거부한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, E29_TAU[structure])
    kw = e29_settings(structure)
    if E29_STAGE not in _reference(d, kw["ref_name"]):
        raise ValueError(f"{kw['ref_name']}에 모드 {E29_STAGE}의 참조가 없다")
    npz = os.path.join(d, os.path.basename(kw["stage_from"]["npz"]))
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz}가 없다 — 첫 통과의 저장된 망이 먼저다")
    kw["stage_from"] = {**kw["stage_from"], "npz": npz}
    return grid_v2(outdir=d, device=device, tau=tau, **kw)

# ── E30 (사전규정 2026-10-30-p5-e30-prespecification.md) — 단계적 배분 ─────────────────────────────────
# CMAME 요건(2026-10-30) §2.1의 대안(2.1′, 사용자 선택 A). 시드마다 참조 없이:
#   1회차 — 단계 1–6을 원좌표 첫 통과 설정(first)과 정규화 좌표(norm) 두 경로로 풀어 E27 규칙으로 고르고, 단계 7을 first로
#           더 푼다(보고하지 않는다, 단계 6의 경쟁자).
#   계획  — 단계 1–7 사슬의 순서 검사가 울린 가장 이른 단계 j*(1–6)를 시드마다 정한다(`p5_stats.e30_order_flags`).
#   2회차 — j*가 있는 시드만 단계 j*를 first·norm·Fourier(ff)·held-rate(long) 넷 가운데 고르고, 단계 j* + 1–6을 다시 두 경로,
#           단계 7을 first로 잇는다.
# 실행은 20시드 앙상블이라 2회차에도 모든 시드가 돈다. 단계 k에서 아직 j*에 이르지 않은 시드(울리지 않은 시드 포함)의 저장
# 단계는 1회차에서 고른 망으로 두므로 그 시드의 2회차 실행은 1회차를 되풀이하고, 결과는 1회차의 것을 쓴다. 비용은 시드마다
# 그 절차가 부른 실행만 센다(`p5_stats.e30`). 단계 1의 first는 첫 통과(E26b), 양단 고정의 단계 1 norm은 E28a의 것과 같은
# 계산이라 그대로 쓴다.
E30_STRUCTURES = {"cc": "clamped2", "nr": "near"}
E30_BASE_PATHS = ("first", "norm")
E30_ESCALATE = ("ff", "long")
E30_LOOKAHEAD = 7
E30_REUSE = {("cc", 1, "first"): E26_TAGS["clamped2"], ("cc", 1, "norm"): "e28_cc_s1_norm",
             ("nr", 1, "first"): E26_TAGS["near"]}
E30_PRESPEC = "docs/paper4-p5/2026-10-30-p5-e30-prespecification.md"
# 양단 고정 상자의 E13 참조는 모드 1–6뿐이다. 단계 7(보고하지 않는 경쟁자)의 기록이 채점할 모드 7의 Λ*는 같은 사다리·두 면
# 고정의 8모드 참조(`run_p5.structure_reference`)에서 읽는다. 단계 1–6은 첫 통과·E28a와 같은 참조로 채점한다.
E30_CC_REF7 = "p5_reference_e13b_clamped2_m8"


def _staged_spec(s: str) -> dict:
    """단계적 배분 정책(아래 `e30_*`)을 돌리는 구조 — 실험 이름(태그·파일 접두), 첫 통과 설정, 다시 쓰는 1회차 실행, 단계 7의
    참조(없으면 그 구조의 참조), τ 파일, 사전규정. E30(양단 고정·근축퇴)과 E31(정책을 고정한 보류 기하)이 같은 함수를 쓴다."""
    if s in E30_STRUCTURES:
        return {"exp": "e30", "base": lambda: e26_settings(E30_STRUCTURES[s]), "tau": E26_TAU[E30_STRUCTURES[s]],
                "ref7": E30_CC_REF7 if s == "cc" else None, "prespec": E30_PRESPEC,
                "reuse": {(k_, p_): t for (s_, k_, p_), t in E30_REUSE.items() if s_ == s}}
    if s in E31_STRUCTURES:
        return {"exp": "e31", "base": lambda: e31_settings(s), "tau": E31_TAU[s], "ref7": None,
                "prespec": E31_PRESPEC, "reuse": {(1, "first"): E31_FP_TAGS[s]}}
    raise ValueError(f"단계적 배분의 구조는 {sorted(E30_STRUCTURES) + list(E31_STRUCTURES)} 중 하나다: {s!r}")


def _e30_check(s: str, rnd: int, k: int) -> None:
    _staged_spec(s)
    if int(rnd) not in (1, 2) or not 1 <= int(k) <= E30_LOOKAHEAD:
        raise ValueError(f"E30 회차는 1·2, 단계는 1–{E30_LOOKAHEAD}다: r{rnd} s{k}")


def e30_tag(s: str, rnd: int, k: int, path: str) -> str:
    _e30_check(s, rnd, k)
    spec = _staged_spec(s)
    if int(rnd) == 1 and (int(k), path) in spec["reuse"]:
        return spec["reuse"][(int(k), path)]
    return f"{spec['exp']}_{s}_r{int(rnd)}_s{int(k)}_{path}"


def e30_selection_name(s: str, rnd: int) -> str:
    return f"p5_{_staged_spec(s)['exp']}_{s}_r{int(rnd)}_selection.json"


def e30_plan_name(s: str) -> str:
    return f"p5_{_staged_spec(s)['exp']}_{s}_plan.json"


def e30_paths(rnd: int, k: int, plan: dict | None = None) -> tuple:
    """회차 rnd 단계 k에서 도는 경로(순서가 동률을 깬다). 단계 7은 first뿐이다. 2회차는 j*가 k인 시드가 있을 때만 ff·long을 더한다."""
    if int(k) == E30_LOOKAHEAD:
        return ("first",)
    if int(rnd) == 1:
        return E30_BASE_PATHS
    return E30_BASE_PATHS + (E30_ESCALATE if int(k) in {int(v) for v in plan["flagged"].values()} else ())


def e30_active(rnd: int, k: int, seed: int, plan: dict | None = None) -> bool:
    """시드가 회차 rnd 단계 k의 결과를 쓰는가. 1회차는 모든 시드, 2회차는 j* ≤ k인 시드."""
    if int(rnd) == 1:
        return True
    j = plan["flagged"].get(str(int(seed)))
    return j is not None and int(j) <= int(k)


def e30_eligible(rnd: int, k: int, seed: int, plan: dict | None = None) -> tuple:
    """시드가 회차 rnd 단계 k에서 고를 수 있는 경로. 2회차의 j* 단계에서만 ff·long이 더해진다."""
    if int(k) == E30_LOOKAHEAD:
        return ("first",)
    if int(rnd) == 2 and plan["flagged"].get(str(int(seed))) == int(k):
        return E30_BASE_PATHS + E30_ESCALATE
    return E30_BASE_PATHS


def e30_chain_choice(s: str, j: int, seed: int, sel1: dict, sel2: dict | None, plan: dict | None) -> tuple:
    """시드의 사슬에서 단계 j를 차지하는 해 — (회차, 경로, 태그). 2회차에서 그 시드가 단계 j에 이르렀으면 2회차의 선택이다."""
    if sel2 is not None and e30_active(2, j, seed, plan):
        p_ = sel2["stages"][str(int(j))]["choice"][str(int(seed))]
        return 2, p_, e30_tag(s, 2, j, p_)
    p_ = sel1["stages"][str(int(j))]["choice"][str(int(seed))]
    return 1, p_, e30_tag(s, 1, j, p_)


def e30_stage_sources(s: str, j: int, sel1: dict, sel2: dict | None = None, plan: dict | None = None,
                      data_dir: str | None = None, n_seeds: int | None = None) -> list:
    """단계 j의 저장 출처 — (태그, 경로)마다 그 망 파일과 시드들(`p5_exploit.load_mixed_stages`의 꼴)."""
    d = manifest.P5_OUTDIR if data_dir is None else data_dir
    groups = {}
    for s_ in range(GRID_SEEDS_A if n_seeds is None else int(n_seeds)):
        _, p_, tag = e30_chain_choice(s, j, s_, sel1, sel2, plan)
        groups.setdefault((tag, p_), []).append(s_)
    return [{"npz": os.path.join(d, f"p5_params_{tag}.npz"), "arm": "exact", "cond": "mc", "mode": int(j),
             **e27_path_spec(p_), "seeds": sorted(v), "path": p_, "tag": tag} for (tag, p_), v in sorted(groups.items())]


def e30_settings(s: str, rnd: int, k: int, path: str, sel1: dict, sel2: dict | None = None,
                 plan: dict | None = None, data_dir: str | None = None) -> dict:
    _e30_check(s, rnd, k)
    k, rnd = int(k), int(rnd)
    spec = _staged_spec(s)
    if int(rnd) == 1 and (k, path) in spec["reuse"]:
        raise ValueError(f"{e30_tag(s, rnd, k, path)}는 이미 있는 실행을 그대로 쓴다")
    if path not in e30_paths(rnd, k, plan):
        raise ValueError(f"r{rnd} 단계 {k}의 경로는 {e30_paths(rnd, k, plan)} 중 하나다: {path!r}")
    if rnd == 2 and (plan is None or not plan["flagged"] or k < int(plan["j_min"])):
        raise ValueError(f"2회차 단계 {k}는 계획의 j_min 이후에만 돈다")
    need = [j for j in range(1, k) if str(j) not in sel1.get("stages", {})]
    if rnd == 2:
        need += [j for j in range(int(plan["j_min"]), k) if str(j) not in (sel2 or {}).get("stages", {})]
    if need:
        raise ValueError(f"앞 단계 {sorted(set(need))}의 선택이 없다")
    kw = {**spec["base"](), "tag": e30_tag(s, rnd, k, path), "n_modes": k, "save_hist": True}
    if spec["ref7"] is not None and k == E30_LOOKAHEAD:
        kw["ref_name"] = spec["ref7"]
    if k >= 2:      # 저장 단계마다 원 일정(20,000회)만큼 난수를 넘긴다 — 경로와 무관하게 단계 k가 첫 통과의 단계 k와 같은 표본을 본다
        kw["stage_from"] = {"mixed": [e30_stage_sources(s, j, sel1, sel2 if rnd == 2 else None, plan, data_dir)
                                      for j in range(1, k)], "width": int(kw.get("width", 128)),
                            "depth": int(kw.get("depth", 5)), "iters": int(kw.get("iters", GRID_ITERS))}
    return _apply_path(kw, path)


def _e30_files(d: str, s: str) -> tuple:
    import json
    load = lambda name: (json.load(open(os.path.join(d, name), encoding="utf-8"))
                         if os.path.exists(os.path.join(d, name)) else None)
    return load(e30_selection_name(s, 1)), load(e30_selection_name(s, 2)), load(e30_plan_name(s))


def e30_run(s: str, rnd: int, k: int, path: str, outdir=None, device=None) -> dict:
    """E30 실행 하나. 앞 단계의 선택(2회차는 계획도)과 그 망 파일이 있어야 한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, _staged_spec(s)["tau"])
    sel1, sel2, plan = _e30_files(d, s)
    if int(k) >= 2 and sel1 is None:
        raise FileNotFoundError(f"{e30_selection_name(s, 1)}가 없다 — 앞 단계를 먼저 고른다")
    if int(rnd) == 2 and plan is None:
        raise FileNotFoundError(f"{e30_plan_name(s)}가 없다 — 1회차의 계획이 먼저다")
    kw = e30_settings(s, rnd, k, path, sel1 or {"stages": {}}, sel2 or {"stages": {}}, plan, data_dir=d)
    if int(k) not in _reference(d, kw["ref_name"]):
        raise ValueError(f"{kw['ref_name']}에 모드 {k}의 참조가 없다")
    return grid_v2(outdir=d, device=device, tau=tau, **_check_sources(kw))


def e30_select_run(s: str, rnd: int, k: int, outdir=None) -> dict:
    """회차 rnd 단계 k의 후보 기록에서 시드마다 고르고 선택 파일에 더한다(이미 있으면 같은지 확인만 한다). 2회차에서 그
    단계에 이르지 않은 시드는 1회차의 선택을 적는다(그 시드의 2회차 실행은 1회차를 되풀이할 뿐 쓰지 않는다)."""
    import json
    from . import p5_stats
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    k, rnd = int(k), int(rnd)
    sel1, sel2, plan = _e30_files(d, s)
    if rnd == 2 and plan is None:
        raise FileNotFoundError(f"{e30_plan_name(s)}가 없다")
    cands = {p_: e30_tag(s, rnd, k, p_) for p_ in e30_paths(rnd, k, plan)}
    recs = {p_: p5_stats.load(d, t) for p_, t in cands.items()}
    seeds = sorted({int(r["seed"]) for r in recs["first"] if int(r["mode"]) == k})
    eligible = {s_: e30_eligible(rnd, k, s_, plan) for s_ in seeds}
    choice = p5_stats.e30_select_stage(recs, k, eligible)
    if rnd == 2:
        for s_ in seeds:
            if not e30_active(2, k, s_, plan):
                choice[s_] = sel1["stages"][str(k)]["choice"][str(s_)]
    name = e30_selection_name(s, rnd)
    p = os.path.join(d, name)
    sel = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"stages": {}}
    entry = {"choice": {str(s_): v for s_, v in sorted(choice.items())}, "candidates": cands,
             "eligible": {str(s_): list(v) for s_, v in sorted(eligible.items())}, "rule": p5_stats.E30_RULE}
    if rnd == 2:
        entry["active"] = [s_ for s_ in seeds if e30_active(2, k, s_, plan)]
    old = sel["stages"].get(str(k))
    if old is not None:
        if old["choice"] != entry["choice"]:
            raise ValueError(f"{name} 단계 {k}의 선택이 이미 있고 다르다 — 덮어쓰지 않는다")
        return sel
    sel["stages"][str(k)] = entry
    manifest.write_json(p, sel)
    manifest.write_json(os.path.join(d, "manifest_" + name), manifest.build({
        "driver": "run_p5_neural.e30_select_run", "structure": s, "round": rnd, "stage": k,
        "reads": sorted(cands.values()), "rule": p5_stats.E30_RULE, "prespecification": _staged_spec(s)["prespec"]}))
    return sel


def e30_plan_run(s: str, outdir=None) -> dict:
    """1회차 사슬(단계 1–7)에서 시드마다 순서 검사가 울린 단계와 가장 이른 j*(1–6)를 정해 계획 파일에 쓴다. 참조를 읽지 않는다."""
    from . import p5_stats
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    sel1, _, old = _e30_files(d, s)
    if sel1 is None or any(str(k) not in sel1["stages"] for k in range(1, E30_LOOKAHEAD + 1)):
        raise ValueError(f"1회차 단계 1–{E30_LOOKAHEAD}의 선택이 먼저다")
    recs = {}
    for k in range(1, E30_LOOKAHEAD + 1):
        for p_ in e30_paths(1, k):
            recs[(k, p_)] = {int(r["seed"]): r for r in p5_stats.load(d, e30_tag(s, 1, k, p_)) if int(r["mode"]) == k}
    flags, flagged = {}, {}
    for s_ in range(GRID_SEEDS_A):
        chain = {k: recs[(k, sel1["stages"][str(k)]["choice"][str(s_)])][s_] for k in range(1, E30_LOOKAHEAD + 1)}
        f = p5_stats.e30_order_flags(chain)
        flags[str(s_)] = f
        if f:
            flagged[str(s_)] = int(min(f))
    plan = {"flagged": flagged, "flags": flags, "j_min": min(flagged.values()) if flagged else None,
            "rule": p5_stats.E30_ORDER_RULE}
    if old is not None:
        if old != plan:
            raise ValueError(f"{e30_plan_name(s)}가 이미 있고 다르다 — 덮어쓰지 않는다")
        return old
    manifest.write_json(os.path.join(d, e30_plan_name(s)), plan)
    manifest.write_json(os.path.join(d, "manifest_" + e30_plan_name(s)), manifest.build({
        "driver": "run_p5_neural.e30_plan_run", "structure": s, "rule": p5_stats.E30_ORDER_RULE,
        "reads": sorted({e30_tag(s, 1, k, p_) for k, p_ in recs}), "prespecification": _staged_spec(s)["prespec"]}))
    return plan

# ── E31 (사전규정 2026-11-03-p5-e31-prespecification.md) — 정책을 고정한 보류 평가, 같은 예산 대조(B3) ─────────────────
# E30의 단계적 배분(4916ba6의 경로·선택 규칙·순서 검사·두 회차, 위의 `e30_*`)을 그대로 두고, 개발에 쓰지 않은 두 기하에서 돌린다
# (검토 R §10.3의 평가 집합, 사용자 결정 2026-11-03): 양단 고정 상자 h/L = 0.07(요건 §2.3(a)의 축, 개발 값 0.04)과 L-브래킷
# 팔 길이 a = 0.4(개발 값 0.5). 첫 통과 설정은 그 계열의 개발 첫 통과(양단 고정 E26b, L-브래킷 E21d)이고 기하·참조·태그만
# 바뀐다. L-브래킷만 둘이 더 바뀐다.
#   MC 점 수 — E21d 규칙(E21 격자족 45개의 평균 점수)을 a = 0.4에 적용한 1,210(1,210.07)이다.
#   감사 격자 — E21 규칙의 (24, 12, 12)는 a = 0.4에서 인증을 통과하지 못한다(제작해 오차 1.8 × 10⁻²). a = 0.5에서의 통과는
#   팔 2의 중심이 제작해의 진동 성분을 홀함수로 만드는 자리에 있어 오차가 상쇄된 덕이었다. 그래서 팔 길이 방향 점 수만 6씩
#   키운 후보 가운데 인증을 통과한 가장 작은 (42, 12, 12)를 쓴다(4.6 × 10⁻⁸). 사다리는 감사 격자의 4/3, 2, 8/3배(E21과 같은 비)다.
# 참조는 모드 1–10이라 단계 7도 같은 참조로 채점한다. τ는 그 계열의 것(보조 분류)이다. 단계 1의 first는 그 구조의 첫 통과다.
E31_PRESPEC = "docs/paper4-p5/2026-11-03-p5-e31-prespecification.md"
E31_STRUCTURES = ("cc07", "lb04")
E31_GEOMETRY = {"cc07": {"L": 1.0, "b": 0.20, "h": 0.07, "E": 1.0, "rho": 1.0, "nu": 0.29},
                "lb04": {**E21_GEOMETRY, "a": 0.40}}
E31_REF = {"cc07": "p5_reference_e31_cc07", "lb04": "p5_reference_e31_lb04"}
E31_REF_MODES = 10
E31_FP_TAGS = {"cc07": "e31_cc07_fp", "lb04": "e31_lb04_fp"}
E31_TAU = {"cc07": E26_TAU["clamped2"], "lb04": f"manifest_p5_tau_{E21_TAU_TAG}.json"}
E31_LB04_MC_POINTS = 1210
E31_LB04_AUDIT_GRID = (42, 12, 12)
E31_LB04_CHECK_GRID = (56, 16, 16)
E31_LB04_LADDER = ((56, 16, 16), (84, 24, 24), (112, 32, 32))


def e31_settings(s: str) -> dict:
    """보류 구조의 첫 통과 설정(`grid_v2` 인자, τ 없음). 단계적 배분의 실행은 여기에 태그·단계·경로를 얹는다(`e30_settings`)."""
    if s == "cc07":
        return {**e26_settings("clamped2"), "tag": E31_FP_TAGS[s], "geometry": dict(E31_GEOMETRY[s]),
                "ref_name": E31_REF[s], "save_hist": True}
    if s == "lb04":
        return {**e21d_settings(), "tag": E31_FP_TAGS[s], "geometry": dict(E31_GEOMETRY[s]), "ref_name": E31_REF[s],
                "mc_points": E31_LB04_MC_POINTS, "audit_grid": E31_LB04_AUDIT_GRID,
                "check_grid": E31_LB04_CHECK_GRID, "check_grids": list(E31_LB04_LADDER), "save_hist": True}
    raise ValueError(f"E31 구조는 {E31_STRUCTURES} 중 하나다: {s!r}")


def e31_reference(s: str, outdir=None) -> dict:
    """보류 구조의 참조(모드 1–10) — 그 계열과 같은 사다리·규칙(`run_p5.structure_reference`). 형상 인증은 양단 고정이 기본 격자,
    L-브래킷이 그 구조의 감사 격자에서 한다(E13-b·E21과 같다)."""
    from . import run_p5
    if s == "cc07":
        return run_p5.structure_reference("e31_cc07", dict(E31_GEOMETRY[s]), faces="two_face", ladder=run_p5.LADDER,
                                          n_modes=E31_REF_MODES, outdir=outdir)
    if s == "lb04":
        return run_p5.structure_reference("e31_lb04", dict(E31_GEOMETRY[s]), faces="x0", ladder=run_p5.L_LADDER,
                                          n_modes=E31_REF_MODES, cert_grid=E31_LB04_AUDIT_GRID, outdir=outdir)
    raise ValueError(f"E31 구조는 {E31_STRUCTURES} 중 하나다: {s!r}")


def e31_fp_run(s: str, outdir=None, device=None) -> dict:
    """보류 구조의 첫 통과(단계 1–6, 20시드). 참조와 그 계열의 τ 파일이 없으면 거부한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, E31_TAU[s])
    kw = {**e31_settings(s), "n_modes": 6}
    if 6 not in _reference(d, kw["ref_name"]):
        raise ValueError(f"{kw['ref_name']}에 모드 6의 참조가 없다")
    return grid_v2(outdir=d, device=device, tau=tau, **kw)


# B3(검토 R §6.4) — 같은 예산의 단순 전략. E30-cc와 같은 구조·설정에서 held-rate 일정(학습률 감소 6,666·13,332회, 그 뒤 유지,
# `long` 경로와 같은 감소 시점)을 단계마다 49,400회(E30-cc 시드 평균 비용 비 2.47 × 20,000회)로 여섯 단계에 고르게 쓴다. 한
# 실행이고 검사·후보 선택·재풀이가 없다. 참조는 E30-cc 단계 1–6과 같다.
B3_TAG = "e31_b3_cc_hr"
B3_ITERS = 49_400


def b3_settings() -> dict:
    return {**e26_settings("clamped2"), "tag": B3_TAG, "n_modes": 6, "iters": B3_ITERS,
            "lr_milestones": E23_MILESTONES, "save_hist": True}


def b3_run(outdir=None, device=None) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, E26_TAU["clamped2"])
    kw = b3_settings()
    if 6 not in _reference(d, kw["ref_name"]):
        raise ValueError(f"{kw['ref_name']}에 모드 6의 참조가 없다")
    return grid_v2(outdir=d, device=device, tau=tau, **kw)

# ── E31r (사전규정 2026-11-07-p5-e31r-prespecification.md) — E31-lb04 단계 6의 사후 재풀이 판독 ─────────────────
# E31-lb04는 단계 6이 20시드 모두 모드 7에 섰고, 모드 6과의 간격(2.9%)이 t보다 좁아 울리지 않아 2회차가 없었다. 울렸다면
# (가상 계획: 20시드 모두 j* = 6) 2회차가 단계 6에 더했을 ff·long을 1회차 사슬 위에서 그대로 푼다. first·norm은 1회차 실행을
# 그대로 쓴다(같은 계산이다, E30 결과 문서 §9). 등록 판정이 아니고, 태그는 E31의 2회차와 섞이지 않게 따로 둔다. 가상 계획은
# 파일로 쓰지 않는다. 단계 7은 다시 풀지 않는다.
E31R_PRESPEC = "docs/paper4-p5/2026-11-07-p5-e31r-prespecification.md"
E31R_STRUCTURE = "lb04"
E31R_STAGE = 6
E31R_PATHS = E30_ESCALATE


def e31r_tag(path: str) -> str:
    if path not in E31R_PATHS:
        raise ValueError(f"E31r 경로는 {E31R_PATHS} 중 하나다: {path!r}")
    return f"e31r_{E31R_STRUCTURE}_s{E31R_STAGE}_{path}"


def e31r_plan(n_seeds: int = GRID_SEEDS_A) -> dict:
    """가상 계획 — 모든 시드가 단계 6에서 울렸다고 둔다."""
    return {"flagged": {str(s_): E31R_STAGE for s_ in range(int(n_seeds))}, "j_min": E31R_STAGE}


def e31r_settings(path: str, sel1: dict, data_dir: str | None = None) -> dict:
    """2회차 j* = 6의 그 경로 설정(`e30_settings`)에 태그만 바꾼다. 저장 단계 1–5는 1회차 사슬이다."""
    tag = e31r_tag(path)
    kw = e30_settings(E31R_STRUCTURE, 2, E31R_STAGE, path, sel1, {"stages": {}}, e31r_plan(), data_dir=data_dir)
    kw["tag"] = tag
    return kw


def e31r_run(path: str, outdir=None, device=None) -> dict:
    """E31r 실행 하나. E31-lb04의 1회차 선택(단계 1–6)과 울림 없는 계획이 있어야 한다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    tau = _tau_of(d, _staged_spec(E31R_STRUCTURE)["tau"])
    sel1, sel2, plan = _e30_files(d, E31R_STRUCTURE)
    if sel1 is None or any(str(j) not in sel1["stages"] for j in range(1, E31R_STAGE + 1)):
        raise FileNotFoundError(f"{e30_selection_name(E31R_STRUCTURE, 1)}에 단계 1–{E31R_STAGE}의 선택이 없다")
    if plan is None or plan["flagged"] or sel2 is not None:
        raise ValueError("E31r는 울림이 없어 2회차가 없던 E31-lb04의 사후 판독이다")
    kw = e31r_settings(path, sel1, data_dir=d)
    if E31R_STAGE not in _reference(d, kw["ref_name"]):
        raise ValueError(f"{kw['ref_name']}에 모드 {E31R_STAGE}의 참조가 없다")
    return grid_v2(outdir=d, device=device, tau=tau, **_check_sources(kw))

# ── E21e (사전규정 2026-10-09-p5-e21e-prespecification.md) ─────────────────────
# E21c는 기준 구조의 밀도(팔의 단위 길이당 16–20점)로 모드 3만 고쳤다. 기준 구조에서 깊은 모드를 살린
# 밀도는 그보다 높은 고해상 격자족(E3, 평균 5,400점)이다. 같은 점수의 격자족을 L에 그대로 쓴다 — 팔의
# 단위 길이당 28–32점, 폭 14–16점, 두께 11–13점(45개, 평균 9,836점 — E21c의 4.7배). 감사 격자는 1.5배 규칙대로
# (48, 24, 20)이고 인증을 통과한다(제작해 최악 3.1e-15). 사다리는 같은 4/3·2·8/3배다. τ는 E21의 것(보조).
E21E_TAG = "e21lb_exact_hires"
E21E_AUDIT_GRID = (48, 24, 20)
E21E_LADDER = ((64, 32, 27), (96, 48, 40), (128, 64, 53))
# 사다리 최상단은 78만 점이라 한 번에 재면 약 180 GB다(dgx02 실측: 10만 점 22.9 GB, 33만 점 75.8 GB).
# 묶어 잰다 — 점마다의 상태는 점끼리 독립이라 같은 사다리다(`solve_p5_neural(ladder_chunk=)`). 10만 점 묶음의
# 사다리 경로를 dgx01에서 끝까지 돌리니 텐서 최대 47.9 GB, 시스템 사용 최대 105.7/119 GiB였다 — 5만 점으로
# 줄이고 사다리 직전·칸 사이에 할당자 캐시를 비운다(사전규정 보완 2026-10-09, 신경 자료 전).
E21E_LADDER_CHUNK = 50_000


def e21e_settings() -> dict:
    return {**e21_settings("exact"), "tag": E21E_TAG, "family_axes": E3_FAMILY_AXES,
            "audit_grid": E21E_AUDIT_GRID, "check_grid": E21E_LADDER[0],
            "check_grids": list(E21E_LADDER), "ladder_chunk": E21E_LADDER_CHUNK}


def e21e_run(outdir=None, device=None) -> dict:
    """E21e. τ는 `manifest_p5_tau_e21_lbracket.json`에서 읽고, 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E21_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — E21 τ 보정이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e21e_settings())

# ── E22 두 층 이종재료 (사전규정 2026-10-04-p5-e22-prespecification.md) ────────────
# E13-a 상자(h = 0.10)를 두께 방향 두 층으로 나눈다. z < h/2는 E = 1(E13-a와 같은 재료), 위는 E = 0.1.
# 밀도·포아송비는 같아 질량 내적(사영·MAC)이 균질과 같고 강성 불연속만 달라진다. 구적 격자
# (n_x, n_y, n_T)의 n_T는 층마다의 Gauss 점 수다. 격자족은 평면 축이 E13-a와 같고(16–20, 8–10) 층마다
# 3·4점이라 두께 평균 7점·평균 1,134점이 E13-a와 같다(30개). 감사 격자는 격자족 최대 (20, 10, 4)의
# 1.5배 올림이며 인증을 통과했다. 계면을 가로지르는 Gauss는 같은 점 수로 인증을 지나지 못한다.
E22_GEOMETRY = {"kind": "bilayer", "L": 1.0, "b": 0.20, "h": 0.10, "E": 1.0, "E_top": 0.1,
                "rho": 1.0, "nu": 0.29}
E22_REF = "p5_reference_e22_bilayer"
E22_REF_LEVEL = (48, 16, 8)              # run_p5.LADDER[-1]
E22_FAMILY_AXES = ((16, 17, 18, 19, 20), (8, 9, 10), (3, 4))
E22_AUDIT_GRID = (30, 15, 6)
E22_CHECK_GRID = (40, 20, 8)
E22_LADDER = ((40, 20, 8), (60, 30, 12), (80, 40, 16))
E22_TAU_TAG = "e22_bilayer"


def e22_settings(arm: str) -> dict:
    """E22 arm 하나의 `grid_v2` 인자. τ는 `e22_run`이 보정 파일에서 읽는다."""
    from .p5_stats import E22_TAGS
    if arm not in E22_TAGS:
        raise ValueError(f"E22 arm은 {sorted(E22_TAGS)} 중 하나여야 합니다: {arm!r}")
    return {"arms": (arm,), "control": False, "tag": E22_TAGS[arm],
            "geometry": dict(E22_GEOMETRY), "bc": "clamped_x0",
            "ref_name": E22_REF, "ref_level": E22_REF_LEVEL,
            "audit_grid": E22_AUDIT_GRID, "check_grid": E22_CHECK_GRID,
            "check_grids": list(E22_LADDER), "family_axes": E22_FAMILY_AXES,
            "pair_scoring": True, "gram_offdiag": True, "per_seed_span": True,
            "save_params": True}


def e22_tau(outdir=None, device=None) -> dict:
    """E22 τ 재보정 — 본실험 arm보다 먼저 한 번."""
    return tau_calibration(outdir=outdir, device=device, tag=E22_TAU_TAG,
                           geometry=dict(E22_GEOMETRY), audit_grid=E22_AUDIT_GRID,
                           family_axes=E22_FAMILY_AXES, heartbeat_every=HEARTBEAT_ITERS)


def e22_run(arm: str, outdir=None, device=None) -> dict:
    """E22 arm 하나. τ는 `manifest_p5_tau_e22_bilayer.json`에서 읽고, 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    p = os.path.join(d, f"manifest_p5_tau_{E22_TAU_TAG}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — e22_tau를 먼저 돌린다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **e22_settings(arm))


# ── E1 오라클 적합 상한 (사전규정 2026-09-10) ───────────────────────────────
# 묻는 것: 모드 4의 `accuracy_shortfall`(형상 맞음, Λ가 참조보다 5-9 % 위)이 최적화가
# 덜 끝난 것인가, 시행공간이 그 정확도를 못 내는 것인가. deflation을 쓰지 않고 참조
# 모드에 직접 적합시킨 뒤 Rayleigh 몫을 잰다. **단측이다** — 적합이 실패해도 그것이
# 공간의 한계라는 뜻은 아니다(FEM 참조가 C⁰이라 매끄러운 시행함수가 못 따라간다).
E1_MODE = 4
E1_ITERS = 20000
E1_SEEDS = 20
E1_ELAM_RULE = 0.05      # 중앙값이 이보다 작으면 e_optimization으로 귀속


def reference_states(mode: int, grids, n_modes: int, geometry, level, faces: str = "x0") -> list:
    """한 번 푼 FEM 참조 모드의 12성분 상태를 여러 격자에서(E21f). 고정 시작벡터로 푼다."""
    from ..problems import p5_block as p5
    r = p5.solve(*level, n_modes=n_modes, geometry=geometry, faces=faces, start_vector="ones")
    v = r["vec_full"][:, mode - 1]
    out = []
    for grid in grids:
        pts, _ = p5._tensor_grid(grid, geometry)
        u = p5.eval_mode_at(r["mesh"], v, pts).T
        gr = p5.eval_grad_at(r["mesh"], v, pts)
        out.append(np.concatenate([u, gr.transpose(1, 2, 0).reshape(9, -1)], axis=0))
    return out


def reference_state(mode: int, grid, n_modes: int = 6, geometry=None, level=None,
                    faces: str = "x0"):
    """참조 모드의 12성분 상태 (u₀u₁u₂, ∂uᵢ/∂xⱼ)를 구적점에서. `_state_one`과 같은 배치.

    `geometry`를 주면(E21f) 그 구조의 참조 사다리 단계 `level`에서 고정 시작벡터로 푼다. 기본값은 커밋된
    상자 경로 그대로다."""
    if geometry is not None:
        return reference_states(mode, [grid], n_modes, geometry, level, faces)[0]
    from ..problems import p5_block as p5
    from .run_p5 import LADDER
    pts, _ = p5._tensor_grid(grid)
    r = p5.solve(*LADDER[-1], n_modes=n_modes)
    v = r["vec_full"][:, mode - 1]
    u = p5.eval_mode_at(r["mesh"], v, pts).T                 # (3, n_q)
    g = p5.eval_grad_at(r["mesh"], v, pts)                   # (n_q, 3, 3)
    return np.concatenate([u, g.transpose(1, 2, 0).reshape(9, -1)], axis=0)


# ── E10 (사전규정 2026-09-22-p5-e10-prespecification.md) ────────────────────
LOCK_MASS_MIN = 0.95      # 변위는 맞다
LOCK_ENERGY_MAX = 0.50    # 변형률은 틀리다


def locking_signature(mac_mass: float, mac_energy: float) -> bool:
    """얇은 구조 잠김의 서명 — **변위는 맞는데 변형률이 틀린** 셀.

    h/L = 0.04에서 굽힘 강성은 h³, 기생 전단·막 강성은 h에 비례하므로 기생 변형률의
    상대 벌점이 (L/h)² = 625다. 전역 MLP가 Kirchhoff형 운동학을 맞추지 못하면 변위는
    거의 맞는데 변형률 에너지만 폭증한다.

    **둘 다 걸어야 한다.** 에너지 쪽만 보면 완전 붕괴 셀(질량 MAC 0.000, E1 시드 5)까지
    잠김으로 잘못 센다. 문턱 0.95는 커밋된 E1의 실패 셋(0.956·0.986·0.988)이 전부 위,
    붕괴 셀이 아래라 둘을 가르는 자리다."""
    return bool(mac_mass >= LOCK_MASS_MIN and mac_energy < LOCK_ENERGY_MAX)


def oracle_fit(outdir=None, mode: int = E1_MODE, n_seeds: int = E1_SEEDS,
               iters: int = E1_ITERS, width: int = 128, depth: int = 5,
               lr: float = 2e-3, quick: bool = False, device=None,
               normalize_input: bool = False, tag: str = "", geometry=None,
               ref_name: str = "p5_reference", ref_level=None, fit_grid=None,
               check_grids=None, deflate=None, prespec=None, save_params: bool = False) -> dict:
    """§E1 — 참조 모드에 변위와 변형률 **둘 다** 적합시키고 Rayleigh 몫을 잰다.

    E21f(2026-10-10)의 선택 인자 — 기본값이면 커밋된 상자 경로 그대로다. `geometry`·`ref_name`·`ref_level`은
    다른 구조의 참조, `fit_grid`는 맞추는 격자(기본 감사 격자), `check_grids`는 맞춘 격자 밖에서 Rayleigh 몫을
    다시 재는 격자들(첫 격자에서는 FEM과의 MAC도), `deflate`={npz, cond, n, grid, width, depth}는 다른 실행이
    저장한 모드 1–n의 원시 상태로 맞춘 함수를 그 격자에서 정확 사영한 뒤의 Rayleigh 몫(`Lam_deflated`)이다.

    E31q(2026-11-11): `save_params`는 맞춘 망을 `p5_params_<tag>.npz`(키 `exact_mc_m{mode}_…`, `p5_exploit.load_stage`가 읽는
    꼴)로 남긴다 — 다른 저장 집합으로 다시 사영할 때 다시 맞추지 않도록. 기본은 남기지 않는다(커밋된 호출 그대로)."""
    import torch
    from torch.func import vmap

    from ..neural import p5_neural as p5n

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    ref = _reference(d, ref_name)
    lam_ref = float(ref[mode]["lam_star"])
    if quick:
        n_seeds, iters, width, depth = 2, 20, 16, 2

    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = p5n.P5_GEOMETRY if geometry is None else geometry
    nu, E, rho = float(g["nu"]), float(g["E"]), float(g["rho"])
    fgrid = AUDIT_GRID if fit_grid is None else tuple(int(v) for v in fit_grid)
    cgrids = [tuple(int(v) for v in cg) for cg in (check_grids or [])]
    pts, w = p5n.quadrature(fgrid, dev, g)
    # E10: 망이 보는 좌표만 축별로 나눈다. 곱해지는 x는 물리 좌표로 남으므로 하드 BC는
    # 그대로다(솔버 변인이지 계기가 아니다).
    sc = (torch.tensor([g["L"], g["b"], g["h"]], device=dev)
          if normalize_input else None)
    if geometry is None:
        S_ref = torch.as_tensor(reference_state(mode, AUDIT_GRID), device=dev)
        S_ref_check = None
    else:
        states = reference_states(mode, [fgrid] + cgrids[:1], max(6, mode), g, tuple(ref_level))
        S_ref = torch.as_tensor(states[0], device=dev)
        S_ref_check = torch.as_tensor(states[1], device=dev) if cgrids else None

    # **참조 자체의 바닥.** 같은 감사 격자에서 FEM 모드의 Rayleigh 몫이 Λ*에서 얼마나
    # 떨어지는지. 적합이 이보다 잘할 수는 없으므로 결과를 읽는 분모다.
    lam_femref = float(p5n.rq_from_state(S_ref, w, nu, E, rho))
    floor = abs(lam_femref - lam_ref) / lam_ref

    m_ref = float(p5n._mass_ip(S_ref, S_ref, w))
    a_ref = float(p5n.energy_ip(S_ref, S_ref, w, nu, E))
    # 초기화는 본실험의 이 모드 스테이지와 같은 시드를 쓴다(k = mode − 1).
    ens = p5n.EnsembleVecMLP3D(n_seeds, width, depth, seed=mode - 1, device=dev)
    opt = torch.optim.Adam(list(ens.params.values()), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 3, 1),
                                            gamma=0.4)

    def loss_fit(p):
        """스케일·부호에 불변인 두 MAC의 합. 시행함수와 참조의 정규화가 서로 무관하므로
        잔차를 그대로 쓰면 스케일만 맞추다 끝난다."""
        S = p5n._state_one(p, ens.base, pts, sc)
        mm = p5n._mass_ip(S, S_ref, w) ** 2 / (p5n._mass_ip(S, S, w) * m_ref)
        ae = p5n.energy_ip(S, S_ref, w, nu, E) ** 2 / (
            p5n.energy_ip(S, S, w, nu, E) * a_ref)
        return (1.0 - mm) + (1.0 - ae)

    t0 = time.perf_counter()
    for it in range(iters):
        r = vmap(loss_fit)(ens.params)
        opt.zero_grad(set_to_none=True)
        r.sum().backward()
        opt.step(); sched.step()
    secs = time.perf_counter() - t0
    if save_params:
        if not tag:
            raise ValueError("save_params는 태그가 있어야 한다(p5_params_<tag>.npz)")
        np.savez_compressed(os.path.join(d, f"p5_params_{tag}.npz"),
                            **{f"exact_mc_m{int(mode)}_{k_}": v.detach().cpu().numpy() for k_, v in ens.params.items()})

    rows = []
    with torch.no_grad():
        S = vmap(lambda p: p5n._state_one(p, ens.base, pts, sc))(ens.params)
        extra = [{} for _ in range(n_seeds)]
        for i, cg in enumerate(cgrids):
            pc, wc = p5n.quadrature(cg, dev, g)
            Sc = vmap(lambda p: p5n._state_one(p, ens.base, pc, sc))(ens.params)
            Ec = p5n.young_at(pc, g)
            for s in range(n_seeds):
                lc = float(p5n.rq_from_state(Sc[s], wc, nu, Ec, rho))
                extra[s].update({f"Lam_check_{i}": lc, f"e_lam_check_{i}": abs(lc - lam_ref) / lam_ref,
                                 f"check_grid_{i}": "x".join(map(str, cg))})
                if i == 0 and S_ref_check is not None:
                    extra[s]["mac_check_0"] = float(
                        p5n._mass_ip(Sc[s], S_ref_check, wc) ** 2
                        / (p5n._mass_ip(Sc[s], Sc[s], wc) * p5n._mass_ip(S_ref_check, S_ref_check, wc)))
        if deflate:
            from .p5_exploit import load_ensembles
            ensD = load_ensembles(deflate["npz"], "exact", deflate["cond"], n_modes=int(deflate["n"]),
                                  n_seeds=n_seeds, width=int(deflate["width"]),
                                  depth=int(deflate["depth"]), device=dev)
            pd_, wd = p5n.quadrature(tuple(int(v) for v in deflate["grid"]), dev, g)
            # E25: 저장 망이 정규화 좌표로 학습됐으면(`deflate["normalize"]`) 같은 축척으로 잰다. 없으면 커밋된 원좌표 경로다.
            sd_ = (torch.tensor(p5n.input_scale(g), device=dev) if deflate.get("normalize") else None)
            Rd = [p5n.state_batch(e, pd_, sd_, None, g) for e in ensD]
            Sd = vmap(lambda p: p5n._state_one(p, ens.base, pd_, sc))(ens.params)
            Pd = vmap(lambda s_, *q: p5n._project_exact(s_, torch.stack(q, 0), wd))(Sd, *Rd)
            Ed = p5n.young_at(pd_, g)
            for s in range(n_seeds):
                extra[s]["Lam_deflated"] = float(p5n.rq_from_state(Pd[s], wd, nu, Ed, rho))
        for s in range(n_seeds):
            lam = float(p5n.rq_from_state(S[s], w, nu, E, rho))
            mac = float(p5n._mass_ip(S[s], S_ref, w) ** 2
                        / (p5n._mass_ip(S[s], S[s], w) * m_ref))
            # **에너지 MAC**(E10). 손실은 처음부터 둘을 다 썼는데 기록이 질량 쪽만
            # 남겨, 잠김 가설이 예측하는 "변위는 맞고 변형률은 틀린" 간극을 볼 수 없었다.
            mac_e = float(p5n.energy_ip(S[s], S_ref, w, nu, E) ** 2
                          / (p5n.energy_ip(S[s], S[s], w, nu, E) * a_ref))
            rows.append({"problem": "P5", "experiment": "E1", "mode": mode,
                         "seed": s, "Lam_fit": lam, "Lam_ref": lam_ref,
                         "e_lam": abs(lam - lam_ref) / lam_ref, "mac": mac,
                         "mac_energy": mac_e,
                         "locking_signature": locking_signature(mac, mac_e),
                         "iters": iters, "grid": "x".join(map(str, fgrid)),
                         "width": width, "depth": depth, **extra[s]})
    med = float(np.median([r["e_lam"] for r in rows]))
    verdict = ("e_optimization" if med < E1_ELAM_RULE else "indeterminate")
    suf = f"_{tag}" if tag else ""
    manifest.write_csv(os.path.join(d, f"p5_e1_oracle_fit{suf}.csv"), rows)
    manifest.write_json(os.path.join(d, f"manifest_p5_e1{suf}.json"), manifest.build({
        "input_scaling": "per-axis [0,1]^3" if normalize_input else "raw",
        "locking_signature_rule":
            f"mass MAC >= {LOCK_MASS_MIN} and energy MAC < {LOCK_ENERGY_MAX}",
        "n_locking_signature": sum(1 for r in rows if r["locking_signature"]),
        "driver": "run_p5_neural.oracle_fit", "quick": bool(quick),
        "prespecification": prespec or "2026-09-11-p5-followup-prespecification.md, E1",
        "mode": mode, "n_seeds": n_seeds, "iters": iters,
        "architecture": [width, depth], "audit_grid": list(fgrid),
        "geometry": _geometry_record(g), "reference": ref_name,
        "ref_level": None if ref_level is None else list(ref_level), "fit_grid": list(fgrid),
        "check_grids": [list(cg) for cg in cgrids],
        "deflate": None if not deflate else {k: (list(v) if isinstance(v, tuple) else v)
                                             for k, v in deflate.items()},
        "launch": manifest.launch_provenance(),
        "params": f"p5_params_{tag}.npz" if save_params else None,
        "loss": "(1 - mass MAC) + (1 - energy MAC) against the FEM reference mode",
        "e_lam_median": med, "rule": E1_ELAM_RULE, "verdict": verdict,
        "one_sided": "a median at or above the rule is indeterminate, not evidence "
                     "that the trial space cannot hold the mode: the FEM reference "
                     "is C0 and a smooth trial function cannot match it exactly",
        "reference_floor_e_lam": floor, "Lam_fem_on_audit_grid": lam_femref,
        "seconds": secs, "added": "2026-09-10, E1"}))
    print(f"[p5 E1] mode {mode}: e_lam 중앙값 {med:.4f} (규칙 {E1_ELAM_RULE}) → "
          f"{verdict} | 참조 바닥 {floor:.4f} | {secs / 3600:.2f} h", flush=True)
    return {"rows": rows, "e_lam_median": med, "verdict": verdict,
            "reference_floor": floor, "outdir": d}



# ── E21f (사전규정 2026-10-10-p5-e21f-prespecification.md) ─────────────────────
# E21d의 모드 6 단계는 20셀 모두 z-지배인 모드 6·7을 건너뛰고 면내 모드 8(Λ ≈ 1.596)에 갔다. 시행공간이 모드 6을
# 담는지 지도 적합으로 묻는다. 맞춘 망을 E21d가 저장한 모드 1–5로 감사 격자에서 정확 사영한 Rayleigh 몫이 E21d
# 모드 6 단계의 Λ보다 낮으면, 더 낮은 함수가 시행공간에 있었는데 최적화가 놓친 것이다. 대조는 E21d가 실제로 찾은
# 모드 8. 맞추는 격자는 E21c 격자족의 중앙(2,079점), 감사 격자와 사다리 최상단은 맞춘 격자 밖이다.
E21F_MODES = (6, 8)
E21F_FIT_GRID = (18, 9, 7)


def e21f_settings(mode: int) -> dict:
    return {"mode": int(mode), "tag": f"e21f_m{int(mode)}", "geometry": dict(E21_GEOMETRY),
            "ref_name": E21_REF, "ref_level": E21_REF_LEVEL, "fit_grid": E21F_FIT_GRID,
            "check_grids": [E21_AUDIT_GRID, E21_LADDER[-1]],
            "deflate": {"npz": os.path.join(manifest.P5_OUTDIR, "p5_params_e21lb_exact_mc1.npz"),
                        "cond": "mc", "n": 5, "grid": E21_AUDIT_GRID, "width": 128, "depth": 5}}


def e21f_run(outdir=None, device=None) -> dict:
    """E21f — 모드 6과 대조 모드 8. 저장 모드는 출력 폴더의 E21d 파라미터에서 읽는다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    out = {}
    for m in E21F_MODES:
        k = e21f_settings(m)
        k["deflate"] = {**k["deflate"], "npz": os.path.join(d, "p5_params_e21lb_exact_mc1.npz")}
        out[m] = oracle_fit(outdir=d, device=device, **k)
    return out


# ── E31p (사전규정 2026-11-08-p5-e31p-prespecification.md) — E31-lb04 단계 6의 표현 판독 ──────────────────────
# E31-lb04(L-브래킷 a 0.4)의 단계 6은 20시드 모두 모드 6(면외 굽힘, Λ 1.632)을 건너뛰고 모드 7(면내 횡, 1.680)에 섰다.
# E21f와 같은 절차(원고 4.4절)로 시행공간이 모드 6을 담는지 묻는다: 모드 6에 맞춘 망을 E31-lb04 첫 통과가 저장한 모드
# 1–5로 감사 격자에서 정확 사영한 Rayleigh 몫이 같은 시드 단계 6의 Λ보다 낮은가. 대조는 단계 6이 실제로 찾은 모드 7.
# 맞추는 격자는 E21f와 같은 단위 길이당 점 수, 다시 재는 격자는 그 구조의 감사 격자와 사다리 최상단이다. 사후 판독이고
# E31의 등록 판정은 바뀌지 않는다.
E31P_PRESPEC = "docs/paper4-p5/2026-11-08-p5-e31p-prespecification.md"
E31P_MODES = (6, 7)


def e31p_settings(mode: int) -> dict:
    s = "lb04"
    return {"mode": int(mode), "tag": f"e31p_{s}_m{int(mode)}", "geometry": dict(E31_GEOMETRY[s]),
            "ref_name": E31_REF[s], "ref_level": E21_REF_LEVEL, "fit_grid": E21F_FIT_GRID,
            "check_grids": [E31_LB04_AUDIT_GRID, E31_LB04_LADDER[-1]], "prespec": E31P_PRESPEC,
            "deflate": {"npz": os.path.join(manifest.P5_OUTDIR, f"p5_params_{E31_FP_TAGS[s]}.npz"),
                        "cond": "mc", "n": 5, "grid": E31_LB04_AUDIT_GRID, "width": 128, "depth": 5}}


def e31p_run(mode: int, outdir=None, device=None) -> dict:
    """E31p 하나(모드 6 또는 대조 모드 7). 저장 모드는 출력 폴더의 E31-lb04 첫 통과 파라미터에서 읽는다."""
    if int(mode) not in E31P_MODES:
        raise ValueError(f"E31p 모드는 {E31P_MODES} 중 하나다: {mode!r}")
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    k = e31p_settings(mode)
    k["deflate"] = {**k["deflate"], "npz": os.path.join(d, f"p5_params_{E31_FP_TAGS['lb04']}.npz")}
    if mode not in _reference(d, k["ref_name"]):
        raise ValueError(f"{k['ref_name']}에 모드 {mode}의 참조가 없다")
    if not os.path.exists(k["deflate"]["npz"]):
        raise FileNotFoundError(f"{k['deflate']['npz']}가 없다 — E31-lb04 첫 통과가 먼저다")
    return oracle_fit(outdir=d, device=device, **k)


# ── E31q (사전규정 2026-11-11-p5-e31q-prespecification.md) — E31p의 적합을 다시 하고 망을 남긴다 ───────────────────
# r6.5 검토 P1: E31p는 맞춘 망을 첫 통과의 저장 모드 1–5로 사영했는데 원고의 결론은 1회차에서 고른 사슬에 관한 것이다. 맞춘 망이
# 남아 있지 않아 같은 설정·같은 시드로 다시 맞추고 망을 남긴다(태그·사전규정·`save_params`만 다르다). 1회차 사슬의 저장 모드로
# 다시 사영하는 판독은 `p5_exploit.e31q_probe_chain`이다. 사후 판독이고 E31·E31p의 판정은 바뀌지 않는다.
E31Q_PRESPEC = "docs/paper4-p5/2026-11-11-p5-e31q-prespecification.md"
E31Q_MODES = E31P_MODES


def e31q_tag(mode: int) -> str:
    if int(mode) not in E31Q_MODES:
        raise ValueError(f"E31q 모드는 {E31Q_MODES} 중 하나다: {mode!r}")
    return f"e31q_lb04_m{int(mode)}"


def e31q_settings(mode: int) -> dict:
    return {**e31p_settings(mode), "tag": e31q_tag(mode), "prespec": E31Q_PRESPEC, "save_params": True}


def e31q_run(mode: int, outdir=None, device=None) -> dict:
    """E31q 적합 하나(모드 6 또는 대조 모드 7) — E31p와 같은 입력 검사."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    k = e31q_settings(mode)
    k["deflate"] = {**k["deflate"], "npz": os.path.join(d, f"p5_params_{E31_FP_TAGS['lb04']}.npz")}
    if mode not in _reference(d, k["ref_name"]):
        raise ValueError(f"{k['ref_name']}에 모드 {mode}의 참조가 없다")
    if not os.path.exists(k["deflate"]["npz"]):
        raise FileNotFoundError(f"{k['deflate']['npz']}가 없다 — E31-lb04 첫 통과가 먼저다")
    return oracle_fit(outdir=d, device=device, **k)


# ── E32 (사전규정 2026-11-09-p5-e32-prespecification.md) ─────────────────────────────────────────────────────────────
# 대수 층 보강(CMAME r6.1 독립 논리 검토 §2.3). 원고 6.2절 — 저장 관행(Rowan 외 [2]: 저장할 때 감사 격자에서 직교화하고 쓸 때
# 자기 계수로 뺀다, 식 (3))의 직교성은 매 반복 다시 뽑는 표본 위에서 남지 않는다 — 은 기준 구조에서만 쟀다.
# (a) T-브래킷: 원좌표 첫 통과(E26d — 정확 사영·원시 저장·매 반복 균등 MC 1,651점)와 같은 설정·같은 시드·같은 표본 순서에서
#     사영 규칙과 저장 규약만 저장 관행으로 바꾼다. 단계 1·2는 저장 함수가 하나 이하라 두 규칙이 같다.
# (b) 기준 구조 B = 283(1,134의 1/4): 저장 관행과, 같은 표본의 정확 사영 대조. B 1,134(E15)·4,536(E20)과 함께 드리프트의
#     B^{−1/2}와 사영 질량의 손실을 세 점으로 읽는다. 계기는 E15·E20과 같다(사다리·Gram·시드별 스팬·파라미터).
E32_PRESPEC = "docs/paper4-p5/2026-11-09-p5-e32-prespecification.md"
E32_TAGS = {"tb_storage": "e32_tb_orth_mc", "b283_storage": "e32_b283_orth_mc", "b283_exact": "e32_b283_exact_mc"}
E32_B_SMALL = E17_MC_POINTS // 4         # 283 — 표본 비직교 O(1/√B)를 두 배로
E32_TAU = {"tb_storage": E26_TAU["tbracket_raw"], "b283_storage": _BASE_TAU, "b283_exact": _BASE_TAU}


def e32_settings(cond: str) -> dict:
    """E32 실행 하나의 `grid_v2` 인자. τ는 넣지 않는다 — `e32_run`이 그 구조의 보정 파일에서 읽는다(보조 분류용)."""
    if cond not in E32_TAGS:
        raise ValueError(f"E32 조건은 {sorted(E32_TAGS)} 중 하나여야 합니다: {cond!r}")
    if cond == "tb_storage":
        return {**e26_settings("tbracket_raw"), "tag": E32_TAGS[cond], "arms": ("diagonal",),
                "storage": "orthogonalized", "batch_diag": True}
    k, _ = followup_settings("e20r_b4")
    k = {**k, "tag": E32_TAGS[cond], "mc_points": E32_B_SMALL}
    if cond == "b283_exact":
        # E15-b와 같다: 원시 저장·정확 사영, batch_diag는 원시 출력의 비대각을 남기는 계측으로 둔다.
        k.update(arms=("exact",), storage="raw")
    return k


def e32_run(cond: str, outdir=None, device=None) -> dict:
    """E32 실행 하나. τ 보정 파일과 참조가 출력 폴더에 없으면 거부한다."""
    import json
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    k = e32_settings(cond)
    p = os.path.join(d, E32_TAU[cond])
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}가 없다 — 그 구조의 τ 보정이 먼저다")
    ref = k.get("ref_name", "p5_reference")
    if not os.path.exists(os.path.join(d, f"{ref}.csv")):
        raise FileNotFoundError(f"{ref}.csv가 없다 — 참조 계산이 먼저다")
    tau = float(json.load(open(p, encoding="utf-8"))["tau"])
    return grid_v2(outdir=d, device=device, tau=tau, **k)

# ── E2·E3 판정 (사전규정의 규칙을 코드로 박는다) ─────────────────────────────
# 결과를 보고 규칙을 고르는 일이 없도록, 판정을 손이 아니라 이 함수가 한다.
# 규칙 자체는 2026-09-11-p5-followup-prespecification.md에 있고 여기서 옮기지 않는다.
E2_SHORTFALL_RULE = 0.05     # QR 모드 4 e_λ 중앙값이 이보다 작으면 e_optimization
E2_SHORTFALL_BASE = 0.058    # 20,000회에서의 같은 중앙값. 이 위면 예산이 원인이 아니다


def e2_verdict(recs, arm: str) -> dict:
    """E2 판정. 규칙 1(벽)과 규칙 2(근접실패)는 서로 독립으로 매긴다."""
    sub = [r for r in recs if arm in r["arm"] and r["condition"] == "resampled"]
    deep = [r for r in sub if r["mode"] >= 4]
    n_correct = sum(1 for r in deep if r["outcome"] == "correct")
    out = {"arm": arm, "n_4_6": len(deep), "correct_4_6": n_correct,
           "wall_robust": n_correct == 0}
    m4 = [r["e_lam"] for r in sub if r["mode"] == 4]
    if m4:
        med = float(np.median(m4))
        out["mode4_e_lam_median"] = med
        out["mode4_correct"] = sum(1 for r in sub
                                   if r["mode"] == 4 and r["outcome"] == "correct")
        out["shortfall"] = ("e_optimization" if med < E2_SHORTFALL_RULE
                            else "not_budget" if med >= E2_SHORTFALL_BASE
                            else "indeterminate")
    return out


def e3_reference_counts(outdir=None) -> dict:
    """E3이 비교할 두 기준을 **커밋된 기록에서** 읽는다 — 하드코딩하지 않는다."""
    recs = [r for r in read_v2_records(outdir) if r["mode"] >= 4]
    pick = lambda cond, a: sum(  # noqa: E731
        1 for r in recs if r["condition"] == cond and a in r["arm"]
        and r["outcome"] == "correct")
    n = sum(1 for r in recs if r["condition"] == "fixed_audit")
    return {"control": pick("fixed_audit", "exact"),
            "resampled": pick("resampled", "exact"), "n": n}


def e3_verdict(correct_4_6: int, n: int = 60, outdir=None) -> dict:
    """E3 판정 — 대조군 구간 안이면 해상도, 재표집 구간 안이면 격자 적응, 둘 다
    아니면 두 요인이 함께 작용한 것으로 기록하고 분리 불가 공시를 유지한다."""
    base = e3_reference_counts(outdir)
    ctl, res, nb = base["control"], base["resampled"], base["n"]
    # 사전규정의 문구는 "**구간 안**"이다 — 이 조건의 비율이 기준 arm의 Wilson 구간에
    # 들어가는지를 본다. 두 구간의 겹침으로 읽으면 훨씬 느슨해져서, 어느 쪽과도 잘
    # 구별되지 않는 결과가 한쪽으로 분류된다.
    p_hat = correct_4_6 / n
    inside = lambda k: mt.wilson(k, nb)[0] <= p_hat <= mt.wilson(k, nb)[1]  # noqa: E731
    in_ctl, in_res = inside(ctl), inside(res)
    reading = ("resolution" if in_ctl and not in_res
               else "grid_adaptation" if in_res and not in_ctl
               else "both")
    return {"correct_4_6": correct_4_6, "n": n, "p_correct": p_hat,
            "wilson": mt.wilson(correct_4_6, n), "reference": base,
            "inside_control": in_ctl, "inside_resampled": in_res,
            "reading": reading}
