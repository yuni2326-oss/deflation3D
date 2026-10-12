"""P5 원고 그림 — CSV·JSONL → PNG. 계산은 하지 않는다.

`render/figures.py`와 같은 규약이다: import 시점에 Agg 강제, 축 라벨은 영문,
저장은 `docs/_generated/fig/paper4/`에 `p5_` 접두로(2026-10-02까지 `fig/paper2/`).

**색은 마지막에 정한다.** 형태를 먼저 고르고(벽 그림은 식별 + 불확실도라 구간 점도표,
사다리 그림은 연속축 위의 두 부류라 로그-로그 선도), 색은 검증된 범주형 팔레트에서
고정 순서로 배정했다. arm 다섯은 슬롯 1–5이고 (a″)를 (a′) 위에 두는 행 순서로 검증했다
(worst adjacent ΔE 9.1 protan / 22.9 normal; (a″)를 (a′) 아래에 두면 주황–분홍이 12.9로
정상시 하한 15를 못 넘는다). 인쇄·흑백을 위해 색에만 의존하지 않는다 — 벽 그림은 마커 모양,
사다리 그림은 실선/파선, 전이 그림은 흑백 모양만으로 같은 정보를 싣는다.

**3D 도감(`fig_modes3d`)은 FEM을 풀지 않는다.** 형상과 성격은 `run_p5.mode_atlas`가 참조
사다리 최상단에서 풀어 쓴 산출물을 읽는다. 색은 변위 크기가 아니라 **지배 성분 부류**다 —
형상은 기하가 이미 보이고, 원고의 논지(성분 공유)는 부류가 싣기 때문이다. 부류 색은 arm
슬롯과 겹치지 않도록 슬롯 7(보라)·6(초록)을 쓰며 전 쌍 검증을 통과했다(ΔE 27.9 deutan).
그림 1 확장(2026-10-09)은 L-브래킷의 x-지배 모드에 빨강(#e34948)을 더한다. 세 색 전 쌍 검증은 정상시
최악 ΔE 33.1로 통과하지만 빨강↔초록의 protan ΔE가 7.2로 6–8 하한대라, 칸마다 성격 라벨(예: "axial (x)")을
글자로 직접 다는 것이 보조 부호다.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import to_rgb     # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection   # noqa: E402

from .. import metrics as mt             # noqa: E402
from ..drivers import manifest           # noqa: E402

DATA_DIR = os.path.join("docs", "_generated", "data", "paper4")
FIG_DIR = os.path.join("docs", "_generated", "fig", "paper4")

# 검증된 범주형 팔레트(고정 순서, 순환 금지). 슬롯 1–4.
C_DIAG, C_EXACT, C_QR, C_FIXED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_REORTH = "#e87ba4"                      # 슬롯 5 — (a″), E11
C_ZDOM, C_YDOM, C_XDOM = "#4a3aa7", "#008300", "#e34948"   # 도감의 성분 부류
INK, INK_MUTED = "#0b0b0b", "#52514e"
BAND, CLAMP, CLAMP_EDGE, OUTLINE = "#e1e0d9", "#6f6d68", "#3b3a37", "#9a9892"
LADDER_RESOLVED = 1e-3

# 그림 2가 읽는 네 구적 차수. 두 실행의 망이 비트 동일이라 한 함수의 수렴 연구가 된다.
LADDER_POINTS = (5400, 12000, 43200, 96000)


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _jsonl(data_dir, name):
    with open(os.path.join(data_dir, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _deep(records, arm=None, cond=None):
    return [r for r in records
            if (arm is None or arm in r["arm"])
            and (cond is None or r["condition"] == cond)
            and r["mode"] >= 4]


def _spread(vals):
    """사다리 끝 세 칸의 상대 폭(원고 Eq. 5)."""
    tail = vals[-3:]
    return (max(tail) - min(tail)) / abs(tail[-1])


ARM_DIAG = "p5_neural_3d[diagonal projection]"
ARM_EXACT = "p5_neural_3d[exact projection]"
ARM_QR = "p5_neural_3d[qr projection]"
ARM_REORTH = "p5_neural_3d[reorth diagonal projection]"


def _cells(records, arm, cond):
    """arm 이름은 **완전일치**로 고른다 — "diagonal"은 "reorth diagonal"의 부분문자열이다."""
    return [r for r in records if r["arm"] == arm and r["condition"] == cond
            and r["mode"] >= 4]


def fig_wall(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """모드 4–6 성공률과 Wilson 구간 — (a) 사전등록 채점, (b) 사다리 필터.

    막대가 아니라 **구간 점도표**다. 읽어야 할 것이 높이가 아니라 구간의 서로소 여부이고,
    막대는 0/60을 아무것도 없는 것으로 그려 버린다. (a″)는 E11 실행이며 자체 사다리를
    가진다. 본실험 네 조건의 사다리는 비트 동일 재측정(E7·E6)에서 온다."""
    main = _jsonl(data_dir, "p5_neural_v2_records.jsonl")
    e11 = _jsonl(data_dir, "p5_neural_e11_reorth_diag_records.jsonl")
    lad = {}
    for f in ("p5_neural_e7_ladder_a_records.jsonl",
              "p5_neural_e7_ladder_b_records.jsonl",
              "p5_neural_e6_budget_ladder_records.jsonl"):
        for r in _jsonl(data_dir, f):
            lad[(r["arm"], r["condition"], r["mode"], r["seed"])] = r
    lad = list(lad.values()) + e11

    # 행 순서 = 검증된 인접 순서. (a″)를 (a′) 바로 위에 둔다.
    arms = [(ARM_REORTH, "resampled", "(a″) Re-orth. diagonal", C_REORTH, "P", e11),
            (ARM_DIAG, "resampled", "(a′) Diagonal", C_DIAG, "o", main),
            (ARM_EXACT, "resampled", "(b) Exact", C_EXACT, "s", main),
            (ARM_QR, "resampled", "(b′) QR", C_QR, "^", main),
            (ARM_EXACT, "fixed_audit", "(b) Fixed grid", C_FIXED, "D", main)]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55), sharey=True)
    for ax, mode in zip(axes, ("pre", "filtered")):
        for i, (arm, cond, label, color, marker, recs) in enumerate(arms):
            if mode == "pre":
                cells = _cells(recs, arm, cond)
            else:
                cells = [r for r in _cells(lad, arm, cond)
                         if r.get("ladder_tail_spread") is not None
                         and r["ladder_tail_spread"] <= LADDER_RESOLVED]
            n = len(cells)
            k = sum(1 for r in cells if r["outcome"] == "correct")
            lo, hi = mt.wilson(k, n) if n else (0.0, 0.0)
            y = len(arms) - 1 - i
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round",
                    zorder=2)
            ax.plot([k / n if n else 0], [y], marker=marker, ms=9, color=color,
                    markeredgecolor="white", markeredgewidth=1.6, zorder=3)
            ax.text(hi + 0.025, y, f"{k}/{n}", va="center", ha="left",
                    fontsize=8.5, color=INK)
        ax.set_xlim(-0.04, 1.06)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([a[2] for a in arms][::-1], fontsize=9)
        ax.set_xlabel("Correct fraction, modes 4–6", fontsize=9)
        ax.set_title("(a) Prespecified scoring" if mode == "pre"
                     else "(b) Restricted to resolved cells",
                     fontsize=9.5, color=INK, pad=7)
        ax.tick_params(axis="x", labelsize=8.5, colors=INK_MUTED, length=3)
        ax.tick_params(axis="y", length=0, labelsize=9, colors=INK)
        ax.grid(axis="x", color="#e6e6e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#cfcfca")
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_wall.png")


def fig_ladder(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """모드 4의 구적 수렴 — (a) 차수에 따른 Λ, (b) 판정이 쓰는 꼬리 폭.

    두 실행(감사 (30,15,12)와 (60,30,24))의 훈련 격자 Λ가 비트 동일이므로 같은 함수를
    네 차수에서 잰 것이 된다.

    (a)는 Λ를 참조로 나눠 그린다 — 원값으로 그리면 해상 셀 아홉이 참조선과 겹쳐
    구별되지 않고 주석이 데이터선을 가린다. (b)는 판정이 실제로 읽는 양을 직접 보인다.
    (a)만으로는 해상 셀이 한 줄로 뭉쳐 "얼마나 잘 수렴했는지"가 보이지 않기 때문이다.
    색과 **선종류·마커**가 같은 이분을 이중으로 싣는다(흑백 인쇄)."""
    a = {r["seed"]: r for r in _jsonl(data_dir, "p5_neural_e2_qr_records.jsonl")
         if r["mode"] == 4}
    b = {r["seed"]: r for r in
         _jsonl(data_dir, "p5_neural_e4_audit_x2_records.jsonl") if r["mode"] == 4}
    lam_ref = float(a[0]["Lam_ref"])

    cells = []
    for s_ in sorted(a):
        vals = [a[s_]["Lam"], a[s_]["Lam_check"], b[s_]["Lam"], b[s_]["Lam_check"]]
        cells.append((s_, vals, _spread(vals)))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})

    n_res = sum(1 for _, _, sp in cells if sp <= LADDER_RESOLVED)
    for _, vals, sp in cells:
        res = sp <= LADDER_RESOLVED
        ax1.plot(LADDER_POINTS, [v / lam_ref for v in vals],
                 color=C_QR if res else C_EXACT, ls="-" if res else "--",
                 lw=1.6, alpha=0.85, marker="o" if res else "^", ms=4.5,
                 markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax1.axhline(1.0, color=INK_MUTED, lw=1.0, ls=":", zorder=2)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xticks(LADDER_POINTS)
    ax1.set_xticklabels([f"{p:,}" for p in LADDER_POINTS])
    ax1.set_xlabel("Quadrature points", fontsize=9)
    ax1.set_ylabel("Λ / Λ*", fontsize=9)
    ax1.set_title("(a) Eigenvalue across four orders", fontsize=9.5, color=INK, pad=7)
    handles = [plt.Line2D([], [], color=C_QR, ls="-", marker="o", ms=4.5, lw=1.6,
                          label=f"Resolved ({n_res})"),
               plt.Line2D([], [], color=C_EXACT, ls="--", marker="^", ms=4.5, lw=1.6,
                          label=f"Unresolved ({len(cells) - n_res})")]
    ax1.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")

    order = sorted(cells, key=lambda c: c[2])
    for i, (_, _, sp) in enumerate(order):
        res = sp <= LADDER_RESOLVED
        ax2.plot([i], [sp], marker="o" if res else "^", ms=7,
                 color=C_QR if res else C_EXACT, markeredgecolor="white",
                 markeredgewidth=1.2, ls="none", zorder=3)
    ax2.axhline(LADDER_RESOLVED, color=INK_MUTED, lw=1.2, ls="--", zorder=2)
    ax2.text(len(order) - 0.4, LADDER_RESOLVED * 1.6, "Resolved threshold 10⁻³",
             fontsize=8, color=INK_MUTED, ha="right")
    ax2.set_yscale("log")
    ax2.set_xlabel("Cells, ordered by tail spread", fontsize=9)
    ax2.set_ylabel("Ladder tail spread s", fontsize=9)
    ax2.set_title("(b) The quantity the rule reads", fontsize=9.5, color=INK, pad=7)
    ax2.set_xticks([])

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
        ax.grid(color="#e6e6e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#cfcfca")
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_ladder.png")


# ─────────────────────────────────────────────────────────────────────────────
# 그림 — 세 구조의 참조 모드형상(3D)
# ─────────────────────────────────────────────────────────────────────────────

ATLAS = (("baseline", "Baseline", "clamped–free", "h/L = 0.04", "p5_neural_v2.csv", "e14base"),
         ("e13a_thick", "E13-a", "clamped–free", "h/L = 0.10", "p5_neural_e13a_thick.csv", "e14a"),
         ("e13b_clamped2", "E13-b", "clamped–clamped", "h/L = 0.04",
          "p5_neural_e13b_clamped2.csv", "e14b"),
         ("e21_lbracket", "L-bracket", "clamped–free", "re-entrant corner",
          "p5_neural_e21lb_diag.csv", "e21lb_diag"),
         ("e22_bilayer", "Two layers", "clamped–free", r"$E_\mathrm{top}/E$ = 0.1",
          "p5_neural_e22bi_diag.csv", "e22bi_diag"))
# 상자 밖 두 구조(그림 1 확장, 2026-10-09)는 구조마다 따로 쓴 도감을 읽는다(`run_p5.extra_atlas`).
# 기존 세 행의 산출물(`p5_mode_shapes.csv`·`p5_mode_character.json`)은 그대로다.
ATLAS_EXTRA = ("e21_lbracket", "e22_bilayer")
ATLAS_ROW_LIMITS = {"e21_lbracket": ((-0.04, 1.04), (-0.12, 0.82), (-0.12, 0.22))}
ATLAS_ROW_HEIGHT = {"e21_lbracket": 1.25}       # 행 높이(기존 행 = 1). 팔2가 y로 0.7까지 뻗는다
ATLAS_ROW_ZOOM = {"e21_lbracket": 1.05}         # 기존 행은 1.42 — L은 칸을 넘치지 않게 줄인다
ATLAS_ROW_VIEW = {"e21_lbracket": (40, -60)}    # (고도각, 방위각) — 위에서 보아 L 모양과 면내 모드가 보이게
ATLAS_ROW_IN = 1.372                             # 기존 행 높이(인치) — 세 행 판의 4.45 in에서
# L-브래킷 블록(팔1·모서리·팔2)끼리 맞닿은 면은 안쪽 면이라 그리지 않는다.
L_INNER_FACES = ({"x1"}, {"x0", "y1"}, {"y0"})
ATLAS_AMP = 0.08                 # 최대 변위를 길이의 8 %로(모드는 임의 정규화)
ATLAS_VIEW = (24, -60)           # (고도각, 방위각), 정사영
ATLAS_LIMITS = ((-0.04, 1.04), (-0.12, 0.32), (-0.12, 0.22))
ATLAS_LIGHT = (-0.35, -0.55, 1.0)


def predicted_wall(dominants) -> int | None:
    """성분 공유 규칙: 저장 집합에 지배 성분을 공유하는 쌍이 처음 생기는 단계 k*의 다음 모드.
    E13 예측(2026-09-24, 49460de)이 쓴 규칙이다."""
    for k in range(2, len(dominants) + 1):
        if len(set(dominants[:k])) < k:
            return k + 1
    return None


def _read_csv(data_dir, name):
    with open(os.path.join(data_dir, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _diag_counts(data_dir, name):
    rows = [r for r in _read_csv(data_dir, name)
            if r["arm"] == ARM_DIAG and r["condition"] == "resampled"]
    out = {int(r["mode"]): int(r["correct"]) for r in rows}
    if sorted(out) != list(range(1, 7)):
        raise ValueError(f"{name}: 대각 arm 모드 1–6이 아니다: {sorted(out)}")
    return out


def _span_medians(data_dir, tag):
    rows = _read_csv(data_dir, f"p5_span_per_seed_{tag}.csv")
    return {k: float(np.median([float(r["max_offdiag"]) for r in rows if int(r["k"]) == k]))
            for k in (2, 3)}


def _block_faces(P, skip=frozenset()):
    """구조점 (nx, ny, nz, 3)의 경계면을 사각형 목록으로. `skip`의 면("x0", "x1", "y0", "y1")은
    뺀다 — 다중 블록에서 블록끼리 맞닿은 안쪽 면이다. 기본값이면 기존 순서 그대로다."""
    nx, ny, nz, _ = P.shape
    F = []
    for k in (0, nz - 1):
        for i in range(nx - 1):
            for j in range(ny - 1):
                F.append([P[i, j, k], P[i + 1, j, k], P[i + 1, j + 1, k], P[i, j + 1, k]])
    for j, name in ((0, "y0"), (ny - 1, "y1")):
        if name in skip:
            continue
        for i in range(nx - 1):
            for k in range(nz - 1):
                F.append([P[i, j, k], P[i + 1, j, k], P[i + 1, j, k + 1], P[i, j, k + 1]])
    for i, name in ((0, "x0"), (nx - 1, "x1")):
        if name in skip:
            continue
        for j in range(ny - 1):
            for k in range(nz - 1):
                F.append([P[i, j, k], P[i, j + 1, k], P[i, j + 1, k + 1], P[i, j, k + 1]])
    return np.array(F)


def _shaded(F, color):
    """양면 Lambert 음영. 색은 범주(성분 부류)이므로 명도 변조가 값을 뜻하지 않는다."""
    light = np.array(ATLAS_LIGHT) / np.linalg.norm(ATLAS_LIGHT)
    n = np.cross(F[:, 2] - F[:, 0], F[:, 3] - F[:, 1])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    sh = 0.62 + 0.38 * np.abs(n @ light)
    return np.clip(np.array(to_rgb(color))[None, :] * sh[:, None], 0.0, 1.0)


def _box_edges(L, b, h):
    c = np.array([[x, y, z] for x in (0, L) for y in (0, b) for z in (0, h)])
    return [(c[a], c[q]) for a in range(8) for q in range(a + 1, 8)
            if np.count_nonzero(np.abs(c[a] - c[q]) > 0) == 1]


def _outline_edges(g):
    """변형 전 윤곽. 상자는 열두 모서리, L-브래킷은 L자 둘레, 두 층 구조는 상자에 계면 사각형을 더한다."""
    if g.get("kind") == "L":
        L, b, h, a = (float(g[k]) for k in ("L", "b", "h", "a"))
        ring = [(0.0, 0.0), (L, 0.0), (L, b + a), (L - b, b + a), (L - b, b), (0.0, b)]
        E = [((p[0], p[1], z), (q[0], q[1], z)) for z in (0.0, h)
             for p, q in zip(ring, ring[1:] + ring[:1])]
        E += [((x, y, 0.0), (x, y, h)) for x, y in ring]
        return [(np.array(p_), np.array(q_)) for p_, q_ in E]
    E = _box_edges(g["L"], g["b"], g["h"])
    if g.get("kind") == "bilayer":
        z = 0.5 * float(g["h"])
        c = [(0.0, 0.0), (g["L"], 0.0), (g["L"], g["b"]), (0.0, g["b"])]
        E += [(np.array([p[0], p[1], z]), np.array([q[0], q[1], z]))
              for p, q in zip(c, c[1:] + c[:1])]
    return E


def _atlas_rows(data_dir):
    """행마다 (구조 기록, [(블록 격자 크기, 안쪽 면, {모드: (n, 6) 배열})]). 기존 세 구조는 공용 도감,
    상자 밖 두 구조는 구조별 도감에서 읽는다."""
    with open(os.path.join(data_dir, "p5_mode_character.json"), encoding="utf-8") as f:
        atlas = json.load(f)
    grid = tuple(atlas["render_grid"])
    shared = {}
    for r in _read_csv(data_dir, "p5_mode_shapes.csv"):
        shared.setdefault((r["structure"], int(r["mode"])), []).append(
            [float(r[k]) for k in ("x", "y", "z", "ux", "uy", "uz")])
    rows = {}
    for tag, *_ in ATLAS:
        if tag not in ATLAS_EXTRA:
            rows[tag] = (atlas["structures"][tag],
                         [(grid, frozenset(), {m: np.array(shared[(tag, m)]) for m in range(1, 5)})])
            continue
        with open(os.path.join(data_dir, f"p5_mode_atlas_{tag}.json"), encoding="utf-8") as f:
            s = json.load(f)
        per = {}
        for r in _read_csv(data_dir, f"p5_mode_shapes_{tag}.csv"):
            per.setdefault((int(r["block"]), int(r["mode"])), []).append(
                [float(r[k]) for k in ("x", "y", "z", "ux", "uy", "uz")])
        inner = L_INNER_FACES if s["geometry"].get("kind") == "L" else (frozenset(),)
        rows[tag] = (s, [(tuple(bk["n"]), frozenset(inner[i]),
                          {m: np.array(per[(i, m)]) for m in range(1, 5)})
                         for i, bk in enumerate(s["render_blocks"])])
    return rows


def _sci(v):
    e = int(np.floor(np.log10(abs(v))))
    return f"{v:.3f}" if e == 0 else f"{v / 10 ** e:.3f}×10$^{{{e}}}$"



def fig_modes3d(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """다섯 구조 × 모드 1–4의 참조 모드형상. 테두리는 성분 공유 규칙이 벽을 예측한 모드."""
    rows_data = _atlas_rows(data_dir)
    cls_color = {"z": C_ZDOM, "y": C_YDOM, "x": C_XDOM}
    weights = [ATLAS_ROW_HEIGHT.get(t, 1.0) for t, *_ in ATLAS]
    top_in, bottom_in = 0.31, 0.02
    H = ATLAS_ROW_IN * sum(weights) + top_in + bottom_in
    fy = lambda inch: inch / H
    fig = plt.figure(figsize=(7.48, H))
    ncol = 4
    left, right = 0.118, 0.998
    cw = (right - left) / ncol
    top = 1.0 - fy(top_in)
    y_cursor = top
    used = set()
    for (tag, name, bc, hl, csv_name, span_tag), wgt in zip(ATLAS, weights):
        s, blocks = rows_data[tag]
        g, faces, modes = s["geometry"], s["faces"], s["modes"]
        dc = _diag_counts(data_dir, csv_name)
        sm = _span_medians(data_dir, span_tag)
        wall = predicted_wall([m["dominant"] for m in modes[:4]])
        rh = fy(ATLAS_ROW_IN * wgt)
        y0 = y_cursor - rh
        y_cursor = y0
        fig.text(0.006, y0 + rh * 0.60, name, fontsize=8.6, weight="bold", color=INK,
                 va="center")
        fig.text(0.006, y0 + rh * 0.60 - fy(0.156), f"{bc}\n{hl}", fontsize=6.8,
                 color=INK_MUTED, va="top", linespacing=1.3)
        (x0, x1), (y0l, y1l), (z0, z1) = ATLAS_ROW_LIMITS.get(tag, ATLAS_LIMITS)
        for ci in range(ncol):
            m = ci + 1
            mi = modes[ci]
            used.add(mi["dominant"])
            ax = fig.add_axes([left + ci * cw - 0.03, y0 + rh * 0.10, cw + 0.06, rh * 0.72],
                              projection="3d")
            ax.patch.set_alpha(0.0)
            F = []
            amp = max(float(np.abs(sh[m][:, 3:]).max()) for _, _, sh in blocks)
            for n, inner, sh in blocks:
                A = sh[m]
                X = A[:, :3].reshape(*n, 3)
                U = A[:, 3:].reshape(*n, 3)
                F.append(_block_faces(X + U / amp * ATLAS_AMP, inner))
            F = np.concatenate(F)
            ax.add_collection3d(Poly3DCollection(
                F, facecolors=_shaded(F, cls_color[mi["dominant"]]), edgecolors="none",
                linewidths=0))
            if g.get("kind") == "bilayer":
                # 재료 계면(두께 중앙)을 보이는 두 옆면(y = 0, x = L)에 변형된 위치로 긋는다. 면 색 위에서
                # 보이도록 흰 선이고, mplot3d가 깊이로 다시 매기는 면 순서보다 위에 오도록 zorder를 준다.
                for n, _inner, sh in blocks:
                    A = sh[m]
                    Dm = (A[:, :3] + A[:, 3:] / amp * ATLAS_AMP).reshape(*n, 3)
                    for line in (Dm[:, 0, n[2] // 2], Dm[-1, :, n[2] // 2]):
                        ax.plot(*line.T, color="#ffffff", lw=0.7, zorder=20)
            for p_, q_ in _outline_edges(g):
                ax.plot(*zip(p_, q_), color=OUTLINE, lw=0.4, ls=(0, (2, 2)))
            for xf in ([0.0] if faces == "x0" else [0.0, g["L"]]):
                pad = 0.03
                ax.add_collection3d(Poly3DCollection(
                    [np.array([[xf, -pad, -pad], [xf, g["b"] + pad, -pad],
                               [xf, g["b"] + pad, g["h"] + pad], [xf, -pad, g["h"] + pad]])],
                    facecolors=CLAMP, edgecolors=CLAMP_EDGE, linewidths=0.4, alpha=0.9))
            ax.set_xlim(x0, x1); ax.set_ylim(y0l, y1l); ax.set_zlim(z0, z1)
            ax.set_box_aspect((x1 - x0, y1l - y0l, z1 - z0), zoom=ATLAS_ROW_ZOOM.get(tag, 1.42))
            ax.set_proj_type("ortho")
            view = ATLAS_ROW_VIEW.get(tag, ATLAS_VIEW)
            ax.view_init(elev=view[0], azim=view[1])
            ax.set_axis_off()

            x_text = left + ci * cw + 0.008
            onset = m == wall
            fig.text(x_text, y0 + rh - fy(0.096), f"Mode {m}", fontsize=7.4, color=INK,
                     weight="bold", va="top")
            fig.text(x_text, y0 + rh - fy(0.254), f"{mi['character']} ({mi['dominant']})",
                     fontsize=6.6, color=INK_MUTED, va="top")
            fig.text(left + (ci + 1) * cw - 0.008, y0 + rh - fy(0.096), f"diag. {dc[m]}/20",
                     fontsize=7.0, ha="right", va="top", color=INK,
                     weight="bold" if onset else "normal")
            lines = [f"Λ* = {_sci(mi['lam_star'])}"]
            if m >= 3:
                lines.append(f"stored 1–{m - 1}: max|C| = {sm[m - 1]:.3g}")
            fig.text(x_text, y0 + fy(0.041), "\n".join(lines), fontsize=6.4,
                     color=INK_MUTED, va="bottom", linespacing=1.3)
            if onset:
                fig.patches.append(FancyBboxPatch(
                    (left + ci * cw + 0.002, y0 + fy(0.027)), cw - 0.004, rh - fy(0.053),
                    boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                    fill=False, lw=0.9, ec=INK, zorder=10))
    handles = [Patch(color=C_ZDOM, label="z-dominant (bending, torsion)"),
               Patch(color=C_YDOM, label="y-dominant (lateral)")]
    if "x" in used:
        handles.append(Patch(color=C_XDOM, label="x-dominant (axial)"))
    handles.append(Patch(facecolor=CLAMP, edgecolor=CLAMP_EDGE, label="clamped face"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.112, 1.0), ncol=len(handles),
               fontsize=7.0, frameon=False, handlelength=1.2, columnspacing=1.6)
    return _save(fig, outdir, "p5_fig_modes3d.png")


# ─────────────────────────────────────────────────────────────────────────────
# 그림 — 전이의 시드 수준 관계(E14)
# ─────────────────────────────────────────────────────────────────────────────

TRANSITION_MARKERS = ("o", "s", "^")        # 본실험, E13-a, E13-b — 흑백에서 모양만으로
TRANSITION_NAMES = ("Baseline", "E13-a", "E13-b")


def fig_transition(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """(a) 등록된 추정량 — 로지스틱 적합과 시드 군집 부트스트랩 띠, 관측 120점.
    (b) 사후 관찰 — 단계 k = 2의 결과 부류별 시드 비대각. 곡선과 띠는 `p5_stats`가 쓴
    산출물에서 읽는다(렌더러는 적합하지 않는다)."""
    from ..drivers import p5_stats
    with open(os.path.join(data_dir, "p5_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    cur, reg = stats["e14_curve"], stats["e14_registered"]
    obs = p5_stats.e14_observations(data_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.45),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.fill_between(cur["x"], cur["lo"], cur["hi"], color=BAND, lw=0, zorder=1)
    ax1.plot(cur["x"], cur["fit"], color=INK, lw=1.6, zorder=3)
    for gi, mk in enumerate(TRANSITION_MARKERS):
        sel = obs["structure"] == gi
        yy = np.where(obs["y"][sel] > 0, 1.07, -0.07) + (gi - 1) * 0.025
        ax1.plot(obs["x"][sel], yy, ls="none", marker=mk, ms=4.6, color=INK_MUTED,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                 label=TRANSITION_NAMES[gi])
    # 추정치는 캡션에 쓴다(형식 규약 §4: 그림 안에는 데이터만).
    ax1.text(0.13, 1.07, "Correct", fontsize=7.4, color=INK_MUTED, va="center")
    ax1.text(0.13, -0.07, "Not\ncorrect", fontsize=7.4, color=INK_MUTED, va="center")
    ax1.set_xlim(-3.35, 0.12); ax1.set_ylim(-0.17, 1.17)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlabel("Stored-span log$_{10}$ max |C$_{ij}$| at stage k", fontsize=9)
    ax1.set_ylabel("P(correct at mode k + 1)", fontsize=9)
    ax1.set_title("(a) Registered estimator", fontsize=9.5, color=INK, pad=7)
    ax1.legend(fontsize=7.8, frameon=False, loc="lower left", handletextpad=0.3)

    k2 = obs["k"] == 2
    cls_mk = {"correct": ("o", INK, INK), "lower_mode_basin": ("x", INK, INK),
              "other": ("o", "white", INK_MUTED)}
    rows = []
    for gi, (tag, _) in enumerate(p5_stats.E14_STRUCTURES):
        out = {(r["mode"], r["seed"]): r["outcome"] for r in p5_stats.load(data_dir, tag)}
        sel = k2 & (obs["structure"] == gi)
        for x, sd in zip(obs["x"][sel], obs["seed"][sel]):
            o = out[(3, int(sd))]
            rows.append((gi, x, o if o in ("correct", "lower_mode_basin") else "other"))
    offs = {"correct": 0.22, "other": 0.0, "lower_mode_basin": -0.22}
    for cls, (mk, fc, ec) in cls_mk.items():
        pts = [(x, 2 - gi + offs[cls]) for gi, x, c in rows if c == cls]
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, ls="none", marker=mk, ms=5.2 if mk == "o" else 5.6,
                     markerfacecolor=fc, markeredgecolor=ec,
                     markeredgewidth=0.9 if mk == "x" else 0.8, color=ec, zorder=3,
                     label={"correct": "Correct", "lower_mode_basin": "Lower-mode basin",
                            "other": "Other failure"}[cls])
    ax2.set_yticks([2, 1, 0]); ax2.set_yticklabels(TRANSITION_NAMES, fontsize=9)
    ax2.set_ylim(-0.5, 2.62)
    ax2.set_xlim(-3.35, 0.12)
    ax2.set_xlabel("Stored-span log$_{10}$ max |C$_{ij}$| at stage k = 2", fontsize=9)
    ax2.set_title("(b) Outcome class at mode 3 (post hoc)", fontsize=9.5, color=INK, pad=7)
    ax2.legend(fontsize=7.6, frameon=False, loc="upper left", ncol=3, handletextpad=0.2,
               columnspacing=0.9, bbox_to_anchor=(0.0, 1.02))
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
        ax.grid(color="#e6e6e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#cfcfca")
    ax2.tick_params(axis="y", length=0, colors=INK)
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_transition.png")


ALL = {"p5_fig_wall": fig_wall, "p5_fig_ladder": fig_ladder,
       "p5_fig_modes3d": fig_modes3d, "p5_fig_transition": fig_transition}
