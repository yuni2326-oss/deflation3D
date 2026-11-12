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
C_SLOT6 = "#008300"                       # 슬롯 6 — 그림 6(c)의 원좌표 Fourier
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
                     else "(b) Restricted to stable cells",
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
                          label=f"Stable ({n_res})"),
               plt.Line2D([], [], color=C_EXACT, ls="--", marker="^", ms=4.5, lw=1.6,
                          label=f"Unstable ({len(cells) - n_res})")]
    ax1.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")

    order = sorted(cells, key=lambda c: c[2])
    for i, (_, _, sp) in enumerate(order):
        res = sp <= LADDER_RESOLVED
        ax2.plot([i], [sp], marker="o" if res else "^", ms=7,
                 color=C_QR if res else C_EXACT, markeredgecolor="white",
                 markeredgewidth=1.2, ls="none", zorder=3)
    ax2.axhline(LADDER_RESOLVED, color=INK_MUTED, lw=1.2, ls="--", zorder=2)
    ax2.text(len(order) - 0.4, LADDER_RESOLVED * 1.6, "Stability threshold 10⁻³",
             fontsize=8, color=INK_MUTED, ha="right")
    ax2.set_yscale("log")
    ax2.set_xlabel("Cells, ordered by tail spread", fontsize=9)
    ax2.set_ylabel("Ladder tail spread s", fontsize=9)
    ax2.set_title("(b) Ladder tail spread per cell", fontsize=9.5, color=INK, pad=7)
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
         ("e13a_thick", "Thick block", "clamped–free", "h/L = 0.10", "p5_neural_e13a_thick.csv", "e14a"),
         ("e13b_clamped2", "C–C block", "clamped–clamped", "h/L = 0.04",
          "p5_neural_e13b_clamped2.csv", "e14b"),
         ("e21_lbracket", "L-bracket", "clamped–free", "re-entrant corner",
          "p5_neural_e21lb_diag.csv", "e21lb_diag"),
         ("e22_bilayer", "Two layers", "clamped–free", r"$E_\mathrm{top}/E$ = 0.1",
          "p5_neural_e22bi_diag.csv", "e22bi_diag"),
         ("e25_tbracket", "T-bracket", "clamped–free", "flange at free end",
          "p5_neural_e25tb_diag.csv", "e25tb_diag"))
# 상자 밖 두 구조(그림 1 확장, 2026-10-09)는 구조마다 따로 쓴 도감을 읽는다(`run_p5.extra_atlas`).
# 기존 세 행의 산출물(`p5_mode_shapes.csv`·`p5_mode_character.json`)은 그대로다.
ATLAS_EXTRA = ("e21_lbracket", "e22_bilayer", "e25_tbracket")
ATLAS_ROW_LIMITS = {"e21_lbracket": ((-0.04, 1.04), (-0.12, 0.82), (-0.12, 0.22)),
                    "e25_tbracket": ((-0.04, 1.04), (-0.37, 0.57), (-0.12, 0.22))}
ATLAS_ROW_HEIGHT = {"e21_lbracket": 1.25, "e25_tbracket": 1.25}   # 행 높이(기존 행 = 1)
ATLAS_ROW_ZOOM = {"e21_lbracket": 1.05, "e25_tbracket": 1.05}     # 기존 행은 1.42 — 칸을 넘치지 않게 줄인다
ATLAS_ROW_VIEW = {"e21_lbracket": (40, -60), "e25_tbracket": (40, -60)}   # 위에서 보아 평면 모양이 보이게
ATLAS_ROW_IN = 1.372                             # 기존 행 높이(인치) — 세 행 판의 4.45 in에서
# L-브래킷 블록(팔1·모서리·팔2)끼리 맞닿은 면은 안쪽 면이라 그리지 않는다.
L_INNER_FACES = ({"x1"}, {"x0", "y1"}, {"y0"})
T_INNER_FACES = ({"x1"}, {"x0", "y0", "y1"}, {"y1"}, {"y0"})     # 줄기·접합·날개(−)·날개(+), t_blocks 순서
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
    """대각 arm 모드 1–6의 correct. 파일이 아직 없으면(E25 실행 전) None — 그림은 칸에 "–"를 쓴다."""
    if not os.path.exists(os.path.join(data_dir, name)):
        return None
    rows = [r for r in _read_csv(data_dir, name)
            if r["arm"] == ARM_DIAG and r["condition"] == "resampled"]
    out = {int(r["mode"]): int(r["correct"]) for r in rows}
    if sorted(out) != list(range(1, 7)):
        raise ValueError(f"{name}: 대각 arm 모드 1–6이 아니다: {sorted(out)}")
    return out


def _span_medians(data_dir, tag):
    if not os.path.exists(os.path.join(data_dir, f"p5_span_per_seed_{tag}.csv")):
        return None
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
    if g.get("kind") in ("L", "T"):
        L, b, h, a = (float(g[k]) for k in ("L", "b", "h", "a"))
        ring = ([(0.0, 0.0), (L, 0.0), (L, b + a), (L - b, b + a), (L - b, b), (0.0, b)] if g["kind"] == "L" else
                [(0.0, 0.0), (L - b, 0.0), (L - b, -a), (L, -a), (L, b + a), (L - b, b + a), (L - b, b), (0.0, b)])
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
        if tag in ATLAS_EXTRA and not os.path.exists(os.path.join(data_dir, f"p5_mode_atlas_{tag}.json")):
            continue                      # 구조별 도감이 아직 없으면 그 행을 건너뛴다(T-브래킷은 참조 뒤에 만든다)
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
        inner = {"L": L_INNER_FACES, "T": T_INNER_FACES}.get(s["geometry"].get("kind"), (frozenset(),))
        rows[tag] = (s, [(tuple(bk["n"]), frozenset(inner[i]),
                          {m: np.array(per[(i, m)]) for m in range(1, 5)})
                         for i, bk in enumerate(s["render_blocks"])])
    return rows


def _sci(v):
    e = int(np.floor(np.log10(abs(v))))
    return f"{v:.3f}" if e == 0 else f"{v / 10 ** e:.3f}×10$^{{{e}}}$"



def fig_modes3d(data_dir: str = DATA_DIR, outdir: str = FIG_DIR, detector: bool = True) -> str:
    """여섯 구조 × 모드 1–4의 참조 모드형상(r2에서 T-브래킷 추가). 테두리는 성분 공유 규칙이 벽을 예측한 모드.

    `detector=False`(CMAME r6.1)는 대각 탐지기 주석(칸마다 "diag. n/20", 저장 스팬의 max|C|, 예측 벽 테두리)을 빼고
    형상·성격·참조 고유값만 남긴 판을 `p5_fig_modes3d_plain.png`로 쓴다. 본문이 대각 규칙을 주장하지 않게 되어
    탐지기 수치는 보충자료(Note S4)로만 간다. 기본값이면 커밋된 그림 그대로다."""
    rows_data = _atlas_rows(data_dir)
    atlas_rows = [row for row in ATLAS if row[0] in rows_data]
    cls_color = {"z": C_ZDOM, "y": C_YDOM, "x": C_XDOM}
    weights = [ATLAS_ROW_HEIGHT.get(t, 1.0) for t, *_ in atlas_rows]
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
    for (tag, name, bc, hl, csv_name, span_tag), wgt in zip(atlas_rows, weights):
        s, blocks = rows_data[tag]
        g, faces, modes = s["geometry"], s["faces"], s["modes"]
        dc = _diag_counts(data_dir, csv_name) if detector else None
        sm = _span_medians(data_dir, span_tag) if detector else None
        wall = predicted_wall([m["dominant"] for m in modes[:4]]) if detector else None
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
            if detector:
                fig.text(left + (ci + 1) * cw - 0.008, y0 + rh - fy(0.096),
                         "diag. –" if dc is None else f"diag. {dc[m]}/20",
                         fontsize=7.0, ha="right", va="top", color=INK,
                         weight="bold" if onset else "normal")
            lines = [f"Λ* = {_sci(mi['lam_star'])}"]
            if m >= 3 and sm is not None:
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
    return _save(fig, outdir, "p5_fig_modes3d.png" if detector else "p5_fig_modes3d_plain.png")


def fig_modes3d_plain(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """그림 3(CMAME r6.1) — 탐지기 주석 없는 참조 모드형상."""
    return fig_modes3d(data_dir, outdir, detector=False)


# ─────────────────────────────────────────────────────────────────────────────
# 그림 — 전이의 시드 수준 관계(E14)
# ─────────────────────────────────────────────────────────────────────────────

TRANSITION_MARKERS = ("o", "s", "^")        # 본실험, E13-a, E13-b — 흑백에서 모양만으로
TRANSITION_NAMES = ("Baseline", "Thick block", "Clamped–clamped block")


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


# ─────────────────────────────────────────────────────────────────────────────
# 그림 — L-브래킷 모드 6 단계의 궤적(E21h, 사전규정 2026-10-14 §3.1의 보조 판독)
# ─────────────────────────────────────────────────────────────────────────────

# 세 조건과 그 색·선종류. 색은 검증된 범주형 팔레트의 슬롯 1–3을 고정 순서로 쓴다(세 색 ΔE 9.2 deutan·
# 27.6 normal 통과; 초록은 배경 대비 2.74라 선 끝에 조건 이름을 글자로 단다). 선종류가 같은 구분을 이중으로 싣는다.
STAGE6 = (("e21lb_s6_norm_x4", "Fourfold schedule, normalized", C_DIAG, "-"),
          ("e21lb_s6_raw_x4", "Fourfold schedule, raw", C_EXACT, "--"),
          ("e21lb_s6_norm_w256", "Width 256, normalized", C_QR, ":"),
          # r2(E23): 높은 학습률 구간을 묶은 80,000회 일정과 Fourier 입력(20,000회). 다섯 색 순서는 검증기 통과
          # (CVD 최악 인접 ΔE 9.1, 정상시 19.6). 대비 경고는 선종류와 범례가 보조 부호다.
          ("e21lb_s6_norm_x4lo", "High rate held, normalized", C_FIXED, "-."),
          ("e21lb_s6_norm_ff", "Fourier input, normalized", C_REORTH, (0, (3, 1, 1, 1))),
          # r2(E26a): 원좌표 Fourier 입력(표 6의 (viii)). 슬롯 6을 고정 순서로 더했다 — 여섯 색 검증 통과(CVD 최악 인접
          # ΔE 9.1, 정상시 19.6, 대비 경고는 선종류·범례가 보조 부호).
          ("e21lb_s6_raw_ff", "Fourier input, raw", C_SLOT6, (0, (5, 1, 1, 1, 1, 1))))
STAGE6_EIG_TOL = 0.05            # (c)의 문턱 = (1 + 0.05)·λ6 — 채점 규칙의 고유값 기준(§3.4)
STAGE6_SMOOTH = 40               # (c)는 시드마다 40 스냅숏(1,000회) 후행 중앙값으로 학습 잡음을 누른 뒤 센다
STAGE6_REDUCTIONS = (26_666, 53_332)   # 반복 4배 일정의 StepLR 감소 시점(80,000 // 3의 배수)


def stage6_hist(data_dir: str, tag: str):
    """E21h 단계의 감사 격자 Λ 이력 — (반복, Λ[스냅숏, 시드]). `grid_v2(save_hist=)`가 커밋한 것을 읽기만 한다."""
    z = np.load(os.path.join(data_dir, f"p5_hist_{tag}.npz"))
    lam = np.asarray(z["exact_mc_m6"], dtype=float)
    return np.arange(lam.shape[0]) * int(z["snapshot_every"]), lam


def _trailing_median(lam, window: int):
    """시드마다 직전 `window` 스냅숏(자기 포함)의 중앙값. 처음 몇 스냅숏은 가진 만큼으로."""
    out = np.empty_like(lam)
    for i in range(lam.shape[0]):
        out[i] = np.median(lam[max(0, i - window + 1):i + 1], axis=0)
    return out


def stage6_counts(data_dir: str, tag: str, threshold: float, smooth: int | None = None):
    """반복마다 Λ가 `threshold` 아래인 시드 수. `smooth`를 주면 시드별 후행 중앙값 위에서 센다."""
    its, lam = stage6_hist(data_dir, tag)
    if smooth:
        lam = _trailing_median(lam, smooth)
    return its, (lam < threshold).sum(axis=1)


def _lbracket_reference(data_dir: str, modes=(6, 7, 8)) -> dict:
    return {int(r["mode"]): float(r["lam_star"])
            for r in _read_csv(data_dir, "p5_reference_e21_lbracket.csv") if int(r["mode"]) in modes}


def fig_stage6(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """모드 6 단계의 궤적 — (a)(b) 반복 4배 일정의 시드별 감사 격자 Λ(정규화·원좌표), (c) 문턱 아래 시드 수.

    형태: 연속축(반복) 위의 궤적이라 선도. 시드별 선은 얇게, 중앙값은 굵게 그리고, 참조 고유값 λ6·λ7·λ8은
    점선, 학습률 감소 시점은 파선으로 둔다. 초기 반복의 큰 Λ는 축 밖으로 둔다 — 원고의 논지는 모드 8 근처
    (Λ ≈ 1.6)에서 모드 6(1.288)으로 내려가는 시점이기 때문이다. (c)의 문턱은 채점 규칙의 고유값 기준
    (1.05·λ6)이고, 스냅숏마다의 학습 잡음이 문턱 근처에서 수를 흔들어 시드별 1,000회 후행 중앙값 위에서 센다.
    폭 256 조건은 20,000회 실행이다. 참조선 라벨은 궤적이 지나지 않는 왼쪽 끝(초기 Λ > 1.6)에 둔다.
    r2: (c)에 표 6의 (viii)(원좌표 Fourier, E26a)을 더했다 — (c)는 표 6의 (iii)–(viii)이다."""
    ref = _lbracket_reference(data_dir)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, (tag, label, color, _), title in zip(axes[:2], STAGE6[:2],
                                                ("(a) Normalized coordinates", "(b) Raw coordinates")):
        its, lam = stage6_hist(data_dir, tag)
        for s_ in range(lam.shape[1]):
            ax.plot(its, lam[:, s_], color=color, lw=0.6, alpha=0.45, zorder=2)
        ax.plot(its, np.median(lam, axis=1), color=color, lw=1.9, zorder=3)
        for m, lv in ref.items():
            ax.axhline(lv, color=INK_MUTED, lw=1.0, ls=":", zorder=1)
            ax.text(1_500, lv - 0.004, f"Mode {m}", fontsize=7.5, color=INK_MUTED, ha="left", va="top")
        ax.set_ylim(1.25, 1.70)
        ax.set_title(title, fontsize=9.5, color=INK, pad=7)
    axes[0].set_ylabel("Audit-grid Λ", fontsize=9)
    axes[1].set_yticklabels([])

    ax = axes[2]
    thr = (1 + STAGE6_EIG_TOL) * ref[6]
    for tag, label, color, ls in STAGE6:
        if not os.path.exists(os.path.join(data_dir, f"p5_hist_{tag}.npz")):
            continue                # E23 이력이 생기기 전에는 세 조건만 그린다
        its, n = stage6_counts(data_dir, tag, thr, smooth=STAGE6_SMOOTH)
        ax.plot(its, n, color=color, ls=ls, lw=1.8, label=label, zorder=3)
    ax.set_ylim(-0.5, 21)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel(r"Seeds within 5% of $\lambda_6$", fontsize=9)
    ax.set_title("(c) Seeds that reached mode 6", fontsize=9.5, color=INK, pad=7)
    short = {"Fourfold schedule, normalized": "Fourfold, normalized", "Fourfold schedule, raw": "Fourfold, raw",
             "High rate held, normalized": "High rate held", "Fourier input, normalized": "Fourier, normalized",
             "Width 256, normalized": "Width 256", "Fourier input, raw": "Fourier, raw"}
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [short.get(t, t) for t in labels], fontsize=7.3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.6)

    for ax in axes:
        for x in STAGE6_REDUCTIONS:
            ax.axvline(x, color="#9a9892", lw=1.0, ls="--", zorder=1)
        ax.set_xlim(0, 80_000)
        ax.set_xticks([0, 20_000, 40_000, 60_000, 80_000])
        ax.set_xticklabels(["0", "20,000", "40,000", "60,000", "80,000"])
        ax.set_xlabel("Iteration", fontsize=9)
        ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
        ax.grid(color="#e6e6e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#cfcfca")
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_stage6.png")


# ── 개념도(r2): 해석기의 흐름과 세 층, 검증 프로토콜 흐름도 ──────────────────────────
# 자료를 읽지 않는 도식이다. 층 색은 검증된 범주형 팔레트의 슬롯을 고정 순서로 쓰고(계기 = 슬롯 1, 대수 = 슬롯 2,
# 최적화 = 슬롯 3, 표현 = 회색), 색만으로 구별하지 않도록 층마다 이름을 글자로 단다.
LAYER_COLORS = {"instrument": C_DIAG, "algebra": C_EXACT, "optimization": C_QR, "representation": "#8a8984"}


def _tint(hex_color: str, t: float = 0.86):
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def _box(ax, x, y, w, h, text, edge, face="white", size=8.2, weight="normal", style="round,pad=0.012,rounding_size=0.018"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.1, edgecolor=edge, facecolor=face,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight,
            zorder=4, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color=INK_MUTED, style="-|>", lw=1.1, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def fig_pipeline(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """그림 1(r2) — 순차 신경 고유해석의 흐름(왼쪽)과 세 층·전제 조건(오른쪽). `data_dir`는 쓰지 않는다(도식)."""
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # (a) 흐름 — 학습 경로는 세로, 측정 경로는 오른쪽 가지
    ax.text(0.0, 0.985, "(a) Sequential solver, stage k", fontsize=9.5, color=INK, weight="bold", va="top")
    X, W = 0.03, 0.25
    _box(ax, X, 0.77, W, 0.10, "Training quadrature\n(Gauss family or Monte Carlo)", C_DIAG, size=7.8)
    _box(ax, X, 0.56, W, 0.12, "Stage network u = d(x) N(x)\nminimizes the Rayleigh quotient\nor the strong-form residual",
         C_QR, size=7.8)
    _box(ax, X, 0.36, W, 0.10, "Output stored as a column of Φ\n(raw, or orthogonalized)", C_EXACT, size=7.8)
    # r6.1: 대각 규칙은 저장 관행의 뺄셈(식 (3))으로만 남고 QR은 정확 사영의 한 계산법이라 상자에서 뺐다.
    _box(ax, X, 0.14, W, 0.12, "Deflation for stage k + 1\n(exact projection, diagonal\nsubtraction, or penalty)",
         C_EXACT, size=7.8)
    _arrow(ax, X + W / 2, 0.77, X + W / 2, 0.68)
    _arrow(ax, X + W / 2, 0.56, X + W / 2, 0.46)
    _arrow(ax, X + W / 2, 0.36, X + W / 2, 0.26)
    ax.annotate("", xy=(X, 0.62), xytext=(X, 0.21), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.1, connectionstyle="arc3,rad=-0.55"))
    ax.text(X - 0.026, 0.415, "next\nstage", fontsize=7.2, color=INK_MUTED, ha="center", va="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    M = 0.32
    _box(ax, M, 0.585, 0.135, 0.07, "Audit grid,\nnever trained on", C_DIAG, size=7.4)
    _box(ax, M, 0.43, 0.135, 0.07, "Quadrature ladder,\nfour orders", C_DIAG, size=7.4)
    _arrow(ax, X + W, 0.62, M, 0.62)
    _arrow(ax, M + 0.0675, 0.585, M + 0.0675, 0.50)
    ax.text(M + 0.0675, 0.405, "every reported Λ,\nshape, MAC, history,\nGram matrix", fontsize=6.9,
            color=INK_MUTED, ha="center", va="top")
    # (b) 층 — 아래에서 위로 진단한다
    ax.text(0.515, 0.985, "(b) Layers of a higher-mode failure", fontsize=9.5, color=INK, weight="bold", va="top")
    rows = [("optimization", "Optimization", "Does the stage reach the mode?",
             "stage schedule, input embedding, coordinate scaling"),
            ("algebra", "Algebra", "Is the stored span removed?",
             "orthogonality drift max|C|, exact projection in the quadrature of use"),
            ("instrument", "Instrument", "Is the trained network measurable?",
             "audit grid, ladder, gate, audit/training ratio"),
            ("representation", "Representation and formulation (preconditions)",
             "Can the trial space hold the mode, and does the objective select it?",
             "representability probe; objective and boundary treatment")]
    BX, BW, BH, y = 0.545, 0.445, 0.17, 0.765
    for key, name, q, tools in rows:
        c = LAYER_COLORS[key]
        ax.add_patch(FancyBboxPatch((BX, y), BW, BH, boxstyle="round,pad=0.008,rounding_size=0.014",
                                    linewidth=1.2, edgecolor=c, facecolor=_tint(c), zorder=3))
        ax.text(BX + 0.014, y + BH - 0.045, name, fontsize=8.6, weight="bold", color=INK, zorder=4)
        ax.text(BX + 0.014, y + BH - 0.095, q, fontsize=7.7 if len(q) < 50 else 7.0, color=INK, zorder=4)
        ax.text(BX + 0.014, y + 0.025, tools, fontsize=7.0, color=INK_MUTED, zorder=4, style="italic")
        y -= 0.205
    ax.annotate("", xy=(0.525, 0.905), xytext=(0.525, 0.18), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.0))
    ax.text(0.512, 0.54, "diagnosed from the bottom up", rotation=90, fontsize=7.0, color=INK_MUTED,
            va="center", ha="center")
    return _save(fig, outdir, "p5_fig_pipeline.png")


def fig_protocol(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """그림 3(r2) — a posteriori 검증 프로토콜의 흐름도. 검사는 계기 층부터 위로 읽고, 실패한 검사는 그 층의 처방을
    가리킨다. 최적화 층은 참조 없는 재풀이 검사다: 저장 모드에서 다른 경로(Fourier 입력, 긴 일정, 정규화 좌표)로 다시
    풀어 감사 Λ가 가장 낮은 해를 지키고, 감소가 허용오차를 넘으면 첫 풀이가 준최적이었다는 증거다(한쪽 방향, 원고
    명제 2 — 건너뜀과 덜 수렴한 혼합은 가르지 못한다). 표현 가능성 탐침도 한쪽 방향이라 낮은 함수를 찾지 못하면 판정 보류다.
    끝 상자는 "더 개선을 찾지 못했다"이지 정확성 인증이 아니다(종합 검토 2026-10-23). r4.1(검토 2026-10-30 §13.3–13.5):
    울림은 "더 낮은 허용 값을 찾았다"로 쓰고 오차 하한은 명제 2의 조건부 해석으로 두며, 하한 게이트는 하한이 있을 때만,
    안정한 해가 없는 단계 위에서 푼 단계도 판정 보류다. `data_dir`는 쓰지 않는다."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    dec = "round4,pad=0.012"
    CX, CW = 0.03, 0.54
    mid = CX + CW / 2
    RX, RW = 0.69, 0.29
    _box(ax, CX + 0.08, 0.88, CW - 0.16, 0.07, "Train stage k on its training quadrature", INK_MUTED, size=8.2)
    checks = [("instrument", 0.70, "Instrument: audit grid, ladder spread s ≤ 10⁻³,\nlower bound if available, "
               "audit/training ratio near 1?"),
              ("algebra", 0.52, "Algebra: Gram matrix recorded (cond G, drift max|C|);\nsubtraction exact in "
               "the quadrature of use?"),
              ("optimization", 0.34, "Optimization: do re-solves along other paths (Fourier\ninput, longer schedule, "
               "normalized coordinates) stay above (1 − t)Λ?")]
    for key, y, text in checks:
        c = LAYER_COLORS[key]
        _box(ax, CX, y, CW, 0.11, text, c, face=_tint(c), style=dec, size=7.8)
    _box(ax, CX + 0.04, 0.03, CW - 0.08, 0.12, "Report the mode; continue with stage k + 1\n"
         "(no further improvement found —\nnot a certificate of accuracy)", INK_MUTED,
         face="#f4f4f1", weight="bold", size=7.4)
    # 세로 흐름(통과 = yes)
    for y0, y1, lab in ((0.88, 0.81, ""), (0.70, 0.63, "yes"), (0.52, 0.45, "yes"), (0.34, 0.155, "yes")):
        _arrow(ax, mid, y0, mid, y1)
        if lab:
            ax.text(mid + 0.012, (y0 + y1) / 2, lab, fontsize=7.4, color=INK_MUTED, va="center")
    # 처방(오른쪽)
    _box(ax, RX, 0.70, RW, 0.11, "Redraw every iteration, node-free\nsampling; re-solve the stage;\n"
         "no stable solution: undetermined,\nas are the stages solved from it", LAYER_COLORS["instrument"], size=7.0)
    _box(ax, RX, 0.53, RW, 0.09, "Exact projection or re-orthogonalize\nat use; re-solve the stage",
         LAYER_COLORS["algebra"], size=7.4)
    _box(ax, RX, 0.345, RW, 0.11, "Keep the lowest Λ: a lower admissible\nvalue was found; flag and re-test\n"
         "(error bound conditional on\nProposition 2)", LAYER_COLORS["optimization"], size=7.0)
    rep = LAYER_COLORS["representation"]
    _box(ax, RX, 0.13, RW, 0.12, "Benchmark problems only — probe:\nfitted target deflated below the stage?\n"
         "yes: optimization path; no: undetermined", rep, face=_tint(rep), style=dec, size=7.0)
    for y, lab in ((0.755, "no"), (0.575, "no"), (0.415, "no: Λ lower\nby > t")):
        _arrow(ax, CX + CW, y, RX, y)
        ax.text((CX + CW + RX) / 2, y + 0.012, lab, fontsize=6.9 if "\n" in lab else 7.4, color=INK_MUTED,
                ha="center", va="bottom")
    _arrow(ax, RX, 0.37, CX + CW, 0.37)
    ax.text((CX + CW + RX) / 2, 0.358, "re-test", fontsize=6.9, color=INK_MUTED, ha="center", va="top")
    # 목표 형상이 있을 때만 — 점선 화살표
    ax.annotate("", xy=(RX + RW / 2, 0.25), xytext=(RX + RW / 2, 0.345),
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.0, ls=(0, (3, 2)), shrinkA=0, shrinkB=0))
    ax.text(RX + RW / 2 + 0.012, 0.30, "target shape\navailable", fontsize=6.6, color=INK_MUTED, va="center")
    return _save(fig, outdir, "p5_fig_protocol.png")


# ── 그림 7(r2): 강형식 arm(E24) ───────────────────────────────────────────────────────────────
# 형태: (a)(b) 단계 → 참조 모드 지도 — 단계 k(가로)에서 시드가 짝지어진 참조 모드(세로; MAC < 0.9는 맨 아래 줄)를 거품
# 넓이로 센다. 점선 대각선이 순서대로 찾는 경우다(변분형의 Rayleigh 몫 최소화가 정하는 순서). (c) 셀마다 감사 Rayleigh 몫과
# 학습된 λ의 차이 대 자유면 견인력의 일(질량당, Green 항등식 분해 `p5_exploit.e24_green`) — 점선 y = x 위에 놓이면 차이를
# 표면 일이 다 설명한다. 색은 검증된 팔레트 슬롯 1(파랑) = 상자 조건, 슬롯 2(주황) = L-브래킷, 마커 모양이 같은 정보를 싣는다.
E24_FIG = {"box": ("e24_pinn_box", "(i) Baseline"), "thick": ("e24_pinn_thick", "(ii) Thick block"),
           "lbracket": ("e24_pinn_lb", "(vi) L-bracket")}
E24_FIG_MODES = 6


def _accurate(r) -> bool:
    import math
    return (r["outcome"] != "invalid_quadrature" and r["matched_mode"] == r["mode"] and r["mac"] >= 0.9
            and math.isfinite(r["e_lam"]) and r["e_lam"] <= 0.05)


def stage_mode_counts(records, n_modes: int = E24_FIG_MODES) -> dict:
    """{(단계, 짝 모드): 시드 수} — 짝 모드는 MAC ≥ 0.9일 때만 세고, 그 밖은 0(알아볼 수 없음)으로 센다."""
    out = {}
    for r in records:
        k = int(r["mode"])
        if k > n_modes:
            continue
        m = int(r["matched_mode"]) if r["mac"] >= 0.9 else 0
        out[(k, m)] = out.get((k, m), 0) + 1
    return out


def fig_strong(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """그림 7(r2) — (a)(b) 강형식 단계가 간 참조 모드(시드 수), (c) 감사 Λ − 학습된 λ 대 자유면 견인력의 일.
    상자 조건은 두꺼운 상자 기록이 있으면 그것을 쓴다(규칙 P). Green 분해 산출물이 없는 구조는 (c)에서 빠진다."""
    box = "thick" if os.path.exists(os.path.join(data_dir, f"p5_neural_{E24_FIG['thick'][0]}_records.jsonl")) else "box"
    keys = (box, "lbracket")
    colors, markers = (C_DIAG, C_EXACT), ("o", "^")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.12]})
    stages = np.arange(1, E24_FIG_MODES + 1)
    for i, (ax, key) in enumerate(zip(axes[:2], keys)):
        tag, title = E24_FIG[key]
        cnt = stage_mode_counts(_jsonl(data_dir, f"p5_neural_{tag}_records.jsonl"))
        ax.plot([0.5, E24_FIG_MODES + 0.5], [0.5, E24_FIG_MODES + 0.5], ls=":", color=INK_MUTED, lw=1.0, zorder=1)
        for (k, m), c in sorted(cnt.items()):
            ax.scatter(k, m, s=22 * c, color=colors[i], alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)
            if c >= 4:
                ax.text(k, m, str(c), ha="center", va="center", fontsize=6.6, color="white", zorder=4)
        ax.set_xticks(stages); ax.set_yticks(range(0, E24_FIG_MODES + 1))
        ax.set_yticklabels(["MAC < 0.9"] + [str(m) for m in stages], fontsize=8)
        ax.set_xlim(0.4, E24_FIG_MODES + 0.6); ax.set_ylim(-0.6, E24_FIG_MODES + 0.6)
        ax.set_xlabel("Stage", fontsize=9)
        ax.set_ylabel("Matched reference mode" if i == 0 else "", fontsize=9)
        ax.set_title(f"({'ab'[i]}) {title}, strong form", fontsize=9.5, color=INK, pad=6)
    ax = axes[2]
    lim = 0.0
    for key, color, marker in zip(keys, colors, markers):
        tag, title = E24_FIG[key]
        p = os.path.join(data_dir, f"p5_exploit_green_{tag}.csv")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        gap = np.array([float(r["Lam_grid"]) - float(r["lam_param"]) for r in rows])
        sw = np.array([float(r["surface_work"]) for r in rows])
        lim = max(lim, float(np.abs(np.concatenate([gap, sw])).max()))
        ax.scatter(sw, gap, s=16, color=color, marker=marker, alpha=0.85, edgecolor="white", linewidth=0.4,
                   zorder=3, label=title.split(") ")[1])
    lim = 1.5 * (lim or 1.0)
    line = np.concatenate([-np.logspace(np.log10(lim), -3, 200), np.logspace(-3, np.log10(lim), 200)])
    ax.plot(line, line, ls=":", color=INK_MUTED, lw=1.0, zorder=1)
    # 차이는 상자에서 ±3, L-브래킷에서 ±50까지 퍼진다 — 0 근처는 선형(±0.1), 그 밖은 로그인 대칭 로그 축.
    for setter in (ax.set_xscale, ax.set_yscale):
        setter("symlog", linthresh=0.1, linscale=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Free-surface traction work per unit mass", fontsize=9)
    ax.set_ylabel("Audit quotient − trainable eigenvalue", fontsize=9)
    ax.set_title("(c) Green identity, strong form", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    for a in axes:
        a.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
        a.grid(color="#e6e6e3", lw=0.8, zorder=0); a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_strong.png")


# ─────────────────────────────────────────────────────────────────────────────
# 그림 5(r6.3) — 대수 층: 드리프트 법칙, 남은 질량, 학습이 최소화한 몫
# ─────────────────────────────────────────────────────────────────────────────
# 원고 표 3의 일곱 조건: (구조, B, 저장 관행 태그, 정확 사영 태그). B 4,536은 저장 관행만 돌았다.
ALGEBRA_RUNS = (("Baseline", 1134, "e15r", "e15b"), ("Baseline", 283, "e32_b283_orth_mc", "e32_b283_exact_mc"),
                ("Baseline", 4536, "e20r_b4", None), ("T-bracket", 1651, "e32_tb_orth_mc", "e26_tb_raw_mc"))
ALGEBRA_MASS_LINE = 0.01
ALGEBRA_MARKERS = {4: "o", 5: "s", 6: "^"}
# (b)(r6.4, 검토 v2 §1.3): 실행마다 색 — 기준 구조는 표본 크기 순으로 한 색상의 밝음 → 어둠, T-브래킷은 다른 색상의 마름모.
ALGEBRA_RUN_STYLE = (("e32_b283_orth_mc", "Baseline, B = 283", "#86b6ef", "o"), ("e15r", "Baseline, B = 1,134", C_DIAG, "o"),
                     ("e20r_b4", "Baseline, B = 4,536", "#104281", "o"), ("e32_tb_orth_mc", "T-bracket, B = 1,651", C_QR, "D"))
ALGEBRA_ZETA_FLOOR = 1e-6         # 이보다 작은 잔차 질량은 왼쪽 끝에 그린다
ALGEBRA_BAND_ZETA = 0.05          # 띠의 c 범위를 읽는 셀: ζ > 0.05


def algebra_band(mech) -> tuple:
    """(c_lo, c_hi) — 저장 관행 모드 4–6 중 ζ > `ALGEBRA_BAND_ZETA`인 셀의 c = (1 − 비)(1 + ζ)/ζ 범위. 학습 몫이 정확히
    뺀 출력과 잔차의 몫의 질량 가중 평균이면 비 = 1 − cζ/(1 + ζ)이고 c = 1 − (잔차의 몫)/(정확히 뺀 출력의 몫)이다."""
    c = [(1.0 - float(r["ratio_mean"])) * (1.0 + float(r["leak_mean"])) / float(r["leak_mean"]) for r in mech
         if int(r["stage"]) >= 4 and float(r["leak_mean"]) > ALGEBRA_BAND_ZETA]
    return min(c), max(c)


def _acc(r) -> bool:
    """게이트 없는 정확(`p5_stats.accurate`와 같은 규칙)."""
    e = r["e_lam"]
    return (r["outcome"] != "invalid_quadrature" and r["matched_mode"] == r["mode"] and r["mac"] >= 0.9
            and e == e and e <= 0.05)


def fig_algebra(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """대수 층 — (a) 저장 관행에서 저장 함수의 훈련 표본 직교성 드리프트(시드 중앙)와 B, 기울기 −1/2 기준선; (b) 저장 관행
    단계 출력이 훈련 표본에서 내는 몫 — 대각 뺄셈(학습이 최소화한 몫)을 같은 표본의 정확 사영으로 나눈 값(새 표본 평균,
    `p5_exploit.e32_mechanism`)과 잔차 질량 ζ, 실행마다 색(r6.4: 남은 질량 대신 ζ — 표본 크기별 띠가 한 곡선족으로 모인다)과
    질량 가중 평균꼴 1 − cζ/(1 + ζ)의 띠; (c) 단계 출력의 남은 질량(셀마다, 감사 격자)과 정확 여부, 저장 관행 대 정확
    사영. 찬 표식은 정확, 빈 표식은 부정확. (a)·(c)의 색은 규칙(저장 관행 파랑, 정확 사영 주황)이고 모양이 모드다."""
    from matplotlib.ticker import NullFormatter
    recs = {t: _jsonl(data_dir, f"p5_neural_{t}_records.jsonl") for _, _, ts, te in ALGEBRA_RUNS for t in (ts, te) if t}
    mech_path = os.path.join(data_dir, "p5_exploit_e32_mechanism.csv")
    mech = _read_csv(data_dir, "p5_exploit_e32_mechanism.csv") if os.path.exists(mech_path) else None
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], width_ratios=[1, 1.15])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    ax = ax_a
    base = [(B, ts) for g, B, ts, _ in ALGEBRA_RUNS if g == "Baseline"]
    lo4 = None
    for m, mk in ALGEBRA_MARKERS.items():
        pts = sorted((B, float(np.median([r["batch_offdiag"] for r in recs[ts] if r["mode"] == m]))) for B, ts in base)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C_DIAG, lw=1.4, marker=mk, ms=6,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        ax.text(pts[-1][0] * 1.12, pts[-1][1], f"mode {m}", fontsize=7.6, color=INK_MUTED, ha="left", va="center")
        tb = float(np.median([r["batch_offdiag"] for r in recs["e32_tb_orth_mc"] if r["mode"] == m]))
        ax.scatter([1651], [tb], marker=mk, s=34, facecolor="white", edgecolor=C_DIAG, linewidth=1.1, zorder=4)
        lo4 = pts[0][1] if m == 4 else lo4
    ref_B = np.array([283, 4536], dtype=float)
    ref = lo4 * 0.62 * np.sqrt(283 / ref_B)
    ax.plot(ref_B, ref, color=INK_MUTED, lw=1.0, ls="--", zorder=2)
    ax.text(600, lo4 * 0.62 * np.sqrt(283 / 600) * 1.12, "slope −1/2", fontsize=7.6, color=INK_MUTED, ha="left",
            va="bottom")
    ax.scatter([], [], marker="o", s=30, facecolor="white", edgecolor=C_DIAG, label="T-bracket, B = 1,651")
    ax.legend(fontsize=7.6, frameon=False, loc="upper right", handletextpad=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([283, 1134, 4536]); ax.set_xticklabels(["283", "1,134", "4,536"])
    ax.set_yticks([0.01, 0.02, 0.05, 0.1]); ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1"])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlim(200, 9000); ax.set_ylim(0.008, 0.16)
    ax.set_xlabel("Training sample size B", fontsize=9)
    ax.set_ylabel("Drift max|C|, seed median", fontsize=9)
    ax.set_title("(a) Drift of the stored functions, baseline", fontsize=9.5, color=INK, pad=7)

    ax = ax_b
    if mech is not None:
        c_lo, c_hi = algebra_band(mech)
        zz = np.logspace(np.log10(ALGEBRA_ZETA_FLOOR), np.log10(200.0), 400)
        ax.fill_between(zz, 1.0 - c_hi * zz / (1.0 + zz), 1.0 - c_lo * zz / (1.0 + zz), color=INK_MUTED, alpha=0.12,
                        linewidth=0, zorder=1)
        ax.text(2e-3, 0.42, f"1 − c$_{{\\rm res}}$ζ/(1 + ζ),\nc$_{{\\rm res}}$ = {c_lo:.2f} to {c_hi:.2f}", fontsize=7.6,
                color=INK_MUTED, ha="left",
                va="center", linespacing=1.3)
        for src, label, color, mk in ALGEBRA_RUN_STYLE:
            rs = [r for r in mech if r["source"] == src and int(r["stage"]) >= 4]
            x = np.array([max(float(r["leak_mean"]), ALGEBRA_ZETA_FLOOR) for r in rs])
            y = np.array([float(r["ratio_mean"]) for r in rs])
            a = np.array([r["accurate"] == "True" for r in rs])
            s = 18 if mk == "o" else 15
            ax.scatter(x[a], y[a], s=s, marker=mk, color=color, edgecolor="white", linewidth=0.3, alpha=0.95, zorder=3)
            ax.scatter(x[~a], y[~a], s=s, marker=mk, facecolor="white", edgecolor=color, linewidth=0.9, zorder=3)
            ax.scatter([], [], s=s, marker=mk, color=color, label=label)
        ax.axhline(1.0, color="#cfcfca", lw=0.9, zorder=1)
        ax.axvline(1.0, color=INK_MUTED, lw=0.9, ls=":", zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(ALGEBRA_ZETA_FLOOR * 0.6, 200.0)
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1.0, 1e2])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=7.4, frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Residual mass ζ, relative to the exactly subtracted output", fontsize=9)
    ax.set_ylabel("Training quotient, Eq. (3) / Eq. (2)", fontsize=9)
    ax.set_title("(b) The quotient that training minimized", fontsize=9.5, color=INK, pad=7)

    ax = ax_c
    rng = np.random.default_rng(5)
    xt, xl, x0 = [], [], 0.0
    for g, B, ts, te in ALGEBRA_RUNS:
        for m in (4, 5, 6):
            for tag, color, dx in ((te, C_EXACT, -0.17), (ts, C_DIAG, 0.17)):
                if tag is None:
                    continue
                rs = [r for r in recs[tag] if r["mode"] == m]
                y = np.array([max(float(r["projected_mass_fraction"]), 1e-5) for r in rs])
                a = np.array([_acc(r) for r in rs])
                xx = x0 + dx + rng.uniform(-0.09, 0.09, size=len(rs))
                ax.scatter(xx[a], y[a], s=13, marker=ALGEBRA_MARKERS[m], color=color, edgecolor="white", linewidth=0.3,
                           alpha=0.9, zorder=3)
                ax.scatter(xx[~a], y[~a], s=13, marker=ALGEBRA_MARKERS[m], facecolor="white", edgecolor=color,
                           linewidth=0.8, zorder=3)
            xt.append(x0); xl.append(str(m))
            x0 += 1.0
        ax.text(x0 - 2.0, 2.2, f"{g}, B = {B:,}", fontsize=7.8, color=INK, ha="center", va="bottom")
        x0 += 0.6
    ax.axhline(ALGEBRA_MASS_LINE, color=INK_MUTED, lw=0.9, ls=":", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4 * 0.8, 6.0)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
    ax.set_xlim(-0.6, x0 - 0.9)
    ax.set_xlabel("Mode of the stage", fontsize=9)
    ax.set_ylabel("Remaining mass", fontsize=9)
    ax.set_title("(c) Remaining mass of the stage output, cell by cell", fontsize=9.5, color=INK, pad=7)
    handles = [plt.Line2D([], [], ls="none", marker="o", color=C_DIAG, ms=5, label="Storage practice, Eq. (3)"),
               plt.Line2D([], [], ls="none", marker="o", color=C_EXACT, ms=5, label="Exact projection, Eq. (2)"),
               plt.Line2D([], [], ls="none", marker="o", markerfacecolor="white", markeredgecolor=INK_MUTED, ms=5,
                          label="Hollow: inaccurate")]
    ax.legend(handles=handles, fontsize=7.6, frameon=False, loc="lower center", ncol=3, handletextpad=0.2,
              bbox_to_anchor=(0.5, -0.36))
    for a in (ax_a, ax_b, ax_c):
        a.tick_params(labelsize=8.2, colors=INK_MUTED, length=3)
        a.grid(color="#e6e6e3", lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color("#cfcfca")
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_algebra.png")


# ─────────────────────────────────────────────────────────────────────────────
# 그림 6(r6.3) — 순서 검사가 읽는 하강: 검출과 인증의 문턱, 그리고 사각지대
# ─────────────────────────────────────────────────────────────────────────────
ORDER_T, ORDER_T_CERT = 0.05, 0.0528     # 등록 문턱; δ² ≤ 10⁻³에서 감사 고유값의 e_λ = 0.05를 인증하는 t_req(원고 식 (9), r6.5)
ORDER_ZOOM = (-0.10, 0.08)
ORDER_FULL = (-0.45, 0.75)


def _order_rows(data_dir):
    """(행 이름, 하강 배열, 색, 마커, 속 채움) — 첫 통과 여덟 구조(960셀)와 보류 L-브래킷 단계 6."""
    rows = _read_csv(data_dir, "p5_posthoc_order_drops.csv")
    fp = [r for r in rows if r["group"] == "first_pass" and r["drop"] != ""]
    lb = [r for r in rows if r["group"] == "staged_allocation" and r["ladder"] == "lb04 round 1"
          and r["stage"] == "6" and r["drop"] != ""]
    pick = (lambda rs: np.array([float(r["drop"]) for r in rs]))
    acc = [r for r in fp if r["class"] == "accurate"]
    return [("Accurate, distinct eigenvalue", pick([r for r in acc if r["same_cluster"] != "True"]), C_DIAG, "o", True),
            ("Accurate, exactly degenerate pair", pick([r for r in acc if r["same_cluster"] == "True"]), C_DIAG, "o",
             False),
            ("Settled at a higher mode", pick([r for r in fp if r["class"] == "higher"]), C_EXACT, "^", True),
            ("Lower mode or mixture", pick([r for r in fp if r["class"] in ("lower", "other")]), INK_MUTED, "s", True),
            ("Held-out L-bracket, stage 6", pick(lb), C_FIXED, "D", True)]


def fig_order_check(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """순서 검사가 문턱과 견주는 양 — 단계마다 뒤의 사다리 안정 단계 가운데 가장 깊은 하강 1 − Λ_later/Λ_stage
    (`p5_posthoc.order_drops`). (a) 전 구간: 더 높은 모드에 선 단계만 오른쪽에 있다. (b) 0 근처 확대: 정확한 단계는
    정확히 퇴화한 쌍 안에서만 0을 넘고(0.8 % 이하), 보류 L-브래킷의 단계 6은 등록 문턱 아래에 남는다. 행과 마커가 부류를
    싣고 색은 보조다(흑백 인쇄). 축 밖으로 나간 점은 행 위에 개수로 적는다."""
    groups = _order_rows(data_dir)
    rng = np.random.default_rng(7)
    n = len(groups)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}" if abs(v) > 1e-12 else "0")
    for ax, (lo, hi) in zip(axes, (ORDER_FULL, ORDER_ZOOM)):
        if ax is axes[1]:
            ax.axvspan(0.0, ORDER_T, color=BAND, alpha=0.55, lw=0, zorder=0)
        for i, (name, x, color, marker, filled) in enumerate(groups):
            y = n - 1 - i
            inside = (x >= lo) & (x <= hi)
            yy = y + rng.uniform(-0.22, 0.22, size=int(inside.sum()))
            ax.scatter(x[inside], yy, s=16, marker=marker, color=color if filled else "white",
                       edgecolor=color if not filled else "white", linewidth=0.9 if not filled else 0.4,
                       alpha=0.85, zorder=3)
            n_left, n_right = int((x < lo).sum()), int((x > hi).sum())
            if n_left:
                ax.text(lo + 0.004 * (hi - lo), y + 0.38, f"◂ {n_left} further left", fontsize=6.8, color=INK_MUTED,
                        ha="left", va="center")
            if n_right:
                ax.text(hi - 0.004 * (hi - lo), y, f"{n_right} ▸", fontsize=6.8, color=INK_MUTED, ha="right",
                        va="center")
        ax.axvline(0.0, color="#cfcfca", lw=0.9, zorder=1)
        ax.axvline(ORDER_T, color=INK, lw=1.2, zorder=2)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(pct)
        ax.set_xlabel("Deepest drop 1 − Λ$_{later}$ / Λ$_{stage}$", fontsize=9)
    ax = axes[1]
    ax.axvline(0.01, color=INK_MUTED, lw=1.0, ls=":", zorder=2)
    ax.axvline(ORDER_T_CERT, color=INK, lw=1.0, ls="--", zorder=2)
    top = n - 0.02
    ax.text(0.01 - 0.0012, n - 1.4, "0.01", fontsize=7.2, color=INK_MUTED, ha="right", va="center")
    ax.text(ORDER_T - 0.0015, top, "t = 0.05", fontsize=7.2, color=INK, ha="right", va="top")
    ax.text(ORDER_T_CERT + 0.0015, top, "certifies\ne$_λ$ > 0.05", fontsize=7.2, color=INK, ha="left", va="top")
    ax.text(0.03, -0.72, "below t:\nnot flagged", fontsize=7.0, color=INK_MUTED, ha="center", va="center")
    axes[0].text(ORDER_T + 0.012, top, "t = 0.05", fontsize=7.2, color=INK, ha="left", va="top")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"{g[0]} ({len(g[1])})" for g in groups][::-1], fontsize=8.2)
    axes[0].set_ylim(-1.0, n + 0.05)
    axes[0].set_title("(a) Every reported stage of the first passes", fontsize=9.5, color=INK, pad=7)
    axes[1].set_title("(b) Near zero: detection and certification", fontsize=9.5, color=INK, pad=7)
    for a in axes:
        a.tick_params(labelsize=8.2, colors=INK_MUTED, length=3)
        a.grid(axis="x", color="#e6e6e3", lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color("#cfcfca")
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_order_check.png")


# ─────────────────────────────────────────────────────────────────────────────
# 그림 7(r6.3) — 학습 없는 판독: 계산된 그대로 → 정렬 → Rayleigh–Ritz, 실행마다의 비용
# ─────────────────────────────────────────────────────────────────────────────
# 원고 표 5(r6.2)를 옮긴 것이다. 각 수의 출처는 보충자료 출처 목록에 있고, 첫 통과와 원 설정의 합계는 시험이
# `p5_posthoc.json`(order_check)과 맞춘다. (그룹, 실행, 셀 수, 계산된 그대로, 정렬, RR, 비용 문자열)
READOUTS = (
    ("Eight structures", "Original setting", 960, 539, 542, None, "—"),
    ("Eight structures", "First pass", 960, 859, 932, 934, "1"),
    ("(iii) Clamped–clamped block", "First pass", 120, 79, 116, 118, "1"),
    ("(iii) Clamped–clamped block", "End-to-end protocol, four paths", 120, 120, 120, 120, "7.45"),
    ("(iii) Clamped–clamped block", "Staged allocation, first round", 120, 118, 120, 120, "2.35"),
    ("(iii) Clamped–clamped block", "Staged allocation, final chain", 120, 120, 120, 120, "4.89 (2.47)"),
    ("(iii) Clamped–clamped block", "Uniform held-rate schedule", 120, 114, 120, 120, "2.57"),
    ("(v) Near-degenerate section", "First pass", 120, 101, 119, 119, "1"),
    ("(v) Near-degenerate section", "First pass and one added stage", 120, 101, 119, 119, "1.27"),
    ("(v) Near-degenerate section", "Staged allocation, first round", 120, 112, 120, 120, "2.38"),
    ("(v) Near-degenerate section", "Staged allocation, final chain", 120, 120, 120, 120, "4.63 (2.76)"),
    ("(vi) L-bracket", "First pass", 120, 100, 100, 100, "1"),
    ("(vi) L-bracket", "First pass and one added stage", 120, 100, 118, 118, "1.27"),
    ("Held-out block", "First pass", 120, 112, 118, 120, "1"),
    ("Held-out block", "Staged allocation", 120, 120, 120, 120, "2.36"),
    ("Held-out L-bracket", "First pass", 120, 98, 98, 98, "1"),
    ("Held-out L-bracket", "Staged allocation", 120, 100, 110, 109, "2.37"),
    ("Held-out L-bracket", "Stage 6 re-solved, expensive paths", 120, 100, 100, 100, "3.9"),
)


def fig_readouts(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """원고 표 5의 그림 — 실행마다 계산된 그대로(빈 원), 감사 Λ 순 정렬(찬 원), 저장 스팬의 Rayleigh–Ritz(마름모, 행
    아래)의 정확 셀 비율과 첫 통과에 대한 비용. 학습 없는 판독이 비용 1에서 얻는 몫과 학습하는 수리가 더하는 몫을 한 장에서
    견준다. 모양이 부류를 싣고 색은 보조다."""
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    y, ticks, labels, heads = 0.0, [], [], []
    group = None
    for g, run, n, comp, reo, rr, cost in READOUTS:
        if g != group:
            y -= 0.55 if group is not None else 0.0
            heads.append((y, g))
            y -= 0.85
            group = g
        a, b = comp / n, reo / n
        ax.plot([a, b], [y, y], color="#b9b8b2", lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter([a], [y], s=34, marker="o", facecolor="white", edgecolor=INK_MUTED, linewidth=1.1, zorder=3)
        ax.scatter([b], [y], s=30, marker="o", color=C_DIAG, edgecolor="white", linewidth=0.5, zorder=4)
        if rr is not None:
            ax.scatter([rr / n], [y - 0.28], s=20, marker="D", color=C_QR, edgecolor="white", linewidth=0.4, zorder=4)
        ax.text(1.012, y, cost, fontsize=7.8, color=INK, va="center", ha="left", transform=ax.get_yaxis_transform())
        ticks.append(y)
        labels.append(run if n == 120 else f"{run} (of 960)")
        y -= 1.0
    for yh, g in heads:
        ax.text(-0.01, yh, g, fontsize=8.4, weight="bold", color=INK, va="center", ha="right",
                transform=ax.get_yaxis_transform())
    ax.text(1.012, 0.55, "Cost", fontsize=8.0, weight="bold", color=INK, va="center", ha="left",
            transform=ax.get_yaxis_transform())
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.9)
    ax.set_ylim(y + 0.3, 1.0)
    ax.set_xlim(0.54, 1.005)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Accurate cells", fontsize=9)
    handles = [plt.Line2D([], [], ls="none", marker="o", markerfacecolor="white", markeredgecolor=INK_MUTED, ms=6,
                          label="As computed, stage order"),
               plt.Line2D([], [], ls="none", marker="o", color=C_DIAG, ms=6, label="Re-ordered by audit eigenvalue"),
               plt.Line2D([], [], ls="none", marker="D", color=C_QR, ms=5, label="Rayleigh–Ritz in the stored span")]
    ax.legend(handles=handles, fontsize=7.8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              handletextpad=0.3, columnspacing=1.2)
    ax.tick_params(labelsize=8.2, colors=INK_MUTED, length=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfca")
    fig.tight_layout()
    return _save(fig, outdir, "p5_fig_readouts.png")


ALL = {"p5_fig_wall": fig_wall, "p5_fig_ladder": fig_ladder,
       "p5_fig_modes3d": fig_modes3d, "p5_fig_modes3d_plain": fig_modes3d_plain,
       "p5_fig_transition": fig_transition,
       "p5_fig_stage6": fig_stage6, "p5_fig_pipeline": fig_pipeline, "p5_fig_protocol": fig_protocol,
       "p5_fig_strong": fig_strong, "p5_fig_algebra": fig_algebra, "p5_fig_order_check": fig_order_check,
       "p5_fig_readouts": fig_readouts}
