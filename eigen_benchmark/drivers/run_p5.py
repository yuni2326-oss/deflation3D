"""P5 참조와 인증 — 계획 §2 전체를 한 드라이버로 재생성한다.

산출: p5_reference.csv(모드별 Λ*·수렴차수·잔차), p5_quadrature_study.csv,
p5_shape_certification.csv, p5_spectrum_gaps.csv, manifest_p5.json.

**실행 순서가 규약의 일부다.** 구적 격자(§2.4)를 먼저 정하지 않으면 모드형상 인증
(§2.2)을 어느 점에서 재야 할지 정해지지 않는다 — 채점이 읽는 점에서 재야 한다.
"""
from __future__ import annotations

import os

import numpy as np

from ..problems import p5_block as p5
from . import manifest

# 계획 §2.1의 세밀화 사다리. 첫 사다리 (6,2,1)–(24,8,4)는 최조밀 단계가 아직
# 점근영역이 아니어서 잔차가 3.1e-3이었다. 한 단 올려 9.6e-4로 좁혔다.
LADDER = [(12, 4, 2), (24, 8, 4), (48, 16, 8)]
N_MODES_REF = 8          # 1–6 사용, 7–8은 §2.3 간격 판정용
N_MODES_USED = 6
GAP_RULE = 0.01          # §2.3 — 상대 간격이 이보다 작으면 쌍으로 채점
QUAD_CANDIDATES = [(8, 6, 4), (12, 6, 4), (16, 8, 6), (20, 10, 6),
                   (24, 12, 8), (32, 16, 10)]
QUAD_TOL = 1e-6
ADDED = "2026-09-03, P5 확장(사전규정 초안 기준)"


# 스모크용 축소 설정. 최상단 (48,16,8)은 희소 풀이만 87초라 스위트에 넣을 수 없다.
QUICK_LADDER = [(3, 2, 1), (6, 4, 2), (12, 8, 4)]
QUICK_QUAD = [(8, 6, 4), (12, 6, 4), (16, 8, 6), (20, 10, 6)]


def main(outdir=None, quick: bool = False) -> dict:
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    ladder = QUICK_LADDER if quick else LADDER
    cands = QUICK_QUAD if quick else QUAD_CANDIDATES

    # ── §2.4 구적 격자 (먼저 정한다) ────────────────────────────────────────
    quad = p5.quadrature_convergence(cands, tol=QUAD_TOL)
    grid = quad["chosen"]
    if grid is None:
        raise RuntimeError("후보 격자 중 어느 것도 §2.4 문턱을 만족하지 않았다")
    print(f"[p5 §2.4] 채택 격자 {grid} ({np.prod(grid)}점), "
          f"기준 자기수렴 {quad['reference_self_consistency']:.2e}")
    manifest.write_csv(os.path.join(d, "p5_quadrature_study.csv"), [
        {"grid": "x".join(map(str, r["grid"])), "n_points": r["n_points"],
         "rel_err_a": r["err_a"], "rel_err_b": r["err_b"],
         "meets_tol": bool(max(r["err_a"], r["err_b"]) <= QUAD_TOL),
         "chosen": bool(r["grid"] == grid), "tol": QUAD_TOL,
         "reference_grid": "x".join(map(str, quad["reference_grid"])),
         "reference_self_consistency": quad["reference_self_consistency"],
         # **ASCII만 쓴다.** 이 열은 렌더된 표에 그대로 인쇄되므로 비-ASCII가
         # 들어가면 폰트 누락으로 두부글자가 된다(기존 회귀가 잡는다).
         "field": "smooth manufactured, u = x*(trig), wavenumber "
                  f"{quad['wavenumber']:.4f}"}
        for r in quad["rows"]])

    # ── §2.1 참조 사다리와 Richardson ──────────────────────────────────────
    lams, dofs, solvers = [], [], []
    for lv in ladder:
        r = p5.solve(*lv, n_modes=N_MODES_REF)
        lams.append(r["lam"]); dofs.append(r["n_dof"]); solvers.append(r["solver"])
        print(f"[p5 §2.1] {lv} 자유도 {r['n_dof']:6d} ({r['solver']}) "
              f"Λ₁ = {r['lam'][0]:.6e}")
    lams = np.asarray(lams)
    rc = p5.richardson([lv[0] for lv in ladder], lams)
    eb = p5.beam_theory_estimate()

    ref_rows = []
    for m in range(N_MODES_REF):
        ref_rows.append({
            "mode": m + 1,
            "lam_star": rc["lam_star"][m], "rate": rc["rate"][m],
            "rel_residual": rc["rel_residual"][m],
            "monotone": bool(rc["monotone"][m]),
            "used_in_grid": bool(m < N_MODES_USED),
            **{f"lam_level{i + 1}": lams[i, m] for i in range(len(ladder))},
            "beam_theory_lam": eb["lam"][m] if m < len(eb["lam"]) else float("nan"),
            "beam_theory_family": eb["family"][m] if m < len(eb["family"]) else "",
            "sanity_ratio": (rc["lam_star"][m] / eb["lam"][m]
                             if m < len(eb["lam"]) else float("nan"))})
    manifest.write_csv(os.path.join(d, "p5_reference.csv"), ref_rows)
    worst_res = float(np.nanmax(rc["rel_residual"][:N_MODES_USED]))
    print(f"[p5 §2.1] modes 1–{N_MODES_USED} 최대 외삽 잔차 {worst_res:.2e}")

    # ── §2.3 근접 고유값 규칙 (참조값으로 판정) ────────────────────────────
    lam = rc["lam_star"]
    gap_rows = []
    for m in range(N_MODES_USED):
        nxt = lam[m + 1] if m + 1 < len(lam) else float("nan")
        rel = abs(nxt - lam[m]) / abs(lam[m]) if np.isfinite(nxt) else float("nan")
        gap_rows.append({"mode": m + 1, "lam_star": lam[m],
                         "lam_next": nxt, "relative_gap": rel,
                         "rule": GAP_RULE,
                         "scored_as_pair": bool(np.isfinite(rel) and rel < GAP_RULE)})
    manifest.write_csv(os.path.join(d, "p5_spectrum_gaps.csv"), gap_rows)
    n_pair = sum(r["scored_as_pair"] for r in gap_rows)
    print(f"[p5 §2.3] 쌍으로 채점할 모드 {n_pair}개 "
          f"(최소 상대 간격 {min(r['relative_gap'] for r in gap_rows if np.isfinite(r['relative_gap'])):.3f})")

    # ── §2.2 모드형상 인증 (§2.4가 정한 점에서) ────────────────────────────
    cert = p5.certify_mode_shapes(ladder, n_modes=N_MODES_USED, grid=grid)
    manifest.write_csv(os.path.join(d, "p5_shape_certification.csv"), [
        {"mode": r["mode"], "self_mac": r["self_mac"],
         "per_pair": " ".join(f"{v:.7f}" for v in r["per_pair"]),
         "n_points": r["n_points"],
         "grid": "x".join(map(str, grid)), "threshold": cert["threshold"],
         "certified": bool(r["self_mac"] >= cert["threshold"])}
        for r in cert["rows"]])
    print(f"[p5 §2.2] 최저 self-MAC {cert['worst_self_mac']:.7f} "
          f"→ {'인증' if cert['certified'] else '미인증'}")

    manifest.write_json(os.path.join(d, "manifest_p5.json"), manifest.build({
        "driver": "run_p5.main",
        "geometry": dict(p5.P5_GEOMETRY),
        "ladder": [list(lv) for lv in ladder], "quick": bool(quick),
        "dof_per_level": dofs, "solver_per_level": solvers,
        "n_modes_reference": N_MODES_REF, "n_modes_used": N_MODES_USED,
        "quadrature_grid": list(grid), "quadrature_tol": QUAD_TOL,
        "quadrature_reference_grid": list(quad["reference_grid"]),
        "worst_extrapolation_residual": worst_res,
        "worst_self_mac": cert["worst_self_mac"],
        "shape_certified": cert["certified"],
        "gap_rule": GAP_RULE, "n_modes_scored_as_pair": int(n_pair),
        "reference_note": "Λ* is the Richardson extrapolant of the three-level "
                          "ladder; the quoted uncertainty is the relative "
                          "extrapolation step |Λ_finest − Λ*|/|Λ*| and is not a "
                          "statistical interval",
        "quadrature_note": "the grid is certified against a smooth manufactured "
                           "field, not against a FEM mode: a global tensor grid "
                           "cannot integrate a C0 FEM strain field to 1e-6 "
                           "because the strain is discontinuous across element "
                           "faces (the 3D analogue of the P1 break-point "
                           "alignment requirement)",
        "not_run": ["neural arms", "Stage 0 branch decision", "main grid"],
        "added": ADDED}))
    return {"outdir": d, "grid": grid, "worst_residual": worst_res,
            "worst_self_mac": cert["worst_self_mac"], "n_pair": int(n_pair)}



# ─────────────────────────────────────────────────────────────────────────────
# 모드 도감(원고 v4 그림 1) — 세 구조의 참조 모드형상과 성격.
#
# 렌더러는 계산하지 않는다(render/ 규약). 형상과 성격을 여기서 산출물로 쓰고 그림은
# 그것을 읽기만 한다. **형상은 참조 사다리 최상단 (48, 16, 8)에서** 풀고, 한 단 아래
# (24, 8, 4)와의 self-MAC을 같이 남겨 그림이 참조의 형상임을 인증한다.
#
# **지배 성분 비율은 질량가중**이다: f_c = ∫ρ u_c² / ∫ρ |u|², 24 × 12 × 10 Gauss.
# 원고 표 1이 이 정의다. E13 참조 JSON(2026-09-24, 커밋 49460de)은 같은 점에서
# 가중 없이 합해 비율이 최대 0.039 다르다 — 라벨은 같으므로 그 예측은 무영향이다.
# ─────────────────────────────────────────────────────────────────────────────

ATLAS_STRUCTURES = (
    ("baseline", {}, "x0", "p5_reference.csv"),
    ("e13a_thick", {"h": 0.10}, "x0", "p5_reference_e13a_thick.csv"),
    ("e13b_clamped2", {}, "two_face", "p5_reference_e13b_clamped2.csv"),
)
ATLAS_LEVEL = LADDER[-1]
ATLAS_CHECK_LEVEL = LADDER[-2]
ATLAS_N_MODES = 6                  # 성격 표는 여섯, 형상 산출물은 넷
ATLAS_N_SHAPES = 4
ATLAS_GRID = (49, 9, 3)            # 렌더용 구조점: 길이·폭·두께 방향 점 수
FRACTION_GRID = (24, 12, 10)       # 원고 표 1
SYMMETRY_LINES = (0.15, 0.85)      # 폭 방향 두 줄(폭에 대한 비), 두께 중앙
SYMMETRY_SPAN = (0.05, 0.95, 25)   # 길이 방향 표본(길이에 대한 비), 점 수
CHARACTER_THRESHOLD = 0.5


SIGN_TIE_RTOL = 1e-6


def normalize_sign(U: np.ndarray) -> np.ndarray:
    """절댓값이 가장 큰 성분이 양수가 되도록 부호를 고정한다(고유벡터 부호는 임의).

    **동률은 잡음에 맡기지 않는다.** 비틀림 모드는 최대 |u_z|가 대칭인 두 모서리에서 같고
    부호가 반대라, 어느 쪽이 근소하게 큰지를 풀이 잡음이 정한다. 최대의 1 − 10⁻⁶ 안에 드는
    항목 가운데 평탄 색인이 가장 작은 것을 기준으로 삼는다."""
    U = np.asarray(U, float)
    a = np.abs(U).ravel()
    i = int(np.flatnonzero(a >= a.max() * (1.0 - SIGN_TIE_RTOL))[0])
    return U if U.ravel()[i] >= 0 else -U


def component_fractions(res: dict, geometry=None, grid=FRACTION_GRID) -> np.ndarray:
    """모드별 질량가중 성분 비율 (n_modes, 3)."""
    g = p5.P5_GEOMETRY if geometry is None else geometry
    pts, W = p5._tensor_grid(grid, g)
    out = []
    for m in range(res["vec_full"].shape[1]):
        U = p5.eval_mode_at(res["mesh"], res["vec_full"][:, m], pts)
        e = g["rho"] * np.einsum("q,qc->c", W, U ** 2)
        out.append(e / e.sum())
    return np.array(out)


def _symmetry_points(g):
    xs = np.linspace(SYMMETRY_SPAN[0] * g["L"], SYMMETRY_SPAN[1] * g["L"],
                     SYMMETRY_SPAN[2])
    ys = np.array(SYMMETRY_LINES) * g["b"]
    return np.array([[x, y, g["h"] / 2] for x in xs for y in ys]), len(xs)


DOMINANT_TIE = 0.05


def _dominant(fractions) -> str:
    """가장 큰 성분. 1·2위 차가 `DOMINANT_TIE`보다 작으면 둘을 함께 쓴다("yz").

    정사각 단면의 비틀림과 축퇴쌍은 u_y와 u_z의 몫이 대칭상 같아 argmax가 반올림 잡음이
    된다. 커밋된 세 구조의 1·2위 차는 최소 0.60이라 이 규칙은 그 라벨을 바꾸지 않는다."""
    f = np.asarray(fractions, float)
    i, j = np.argsort(-f, kind="stable")[:2]
    if f[i] - f[j] < DOMINANT_TIE:
        return "".join(sorted("xyz"[i] + "xyz"[j]))
    return "xyz"[i]


def mode_character(U_lines: np.ndarray, fractions) -> dict:
    """U_lines: (길이 방향 점, 폭 방향 두 줄, 3), 군집이면 앞에 구성 모드 축이 붙는다.
    지배 성분과, u_z가 지배에 들면 u_z의 폭 방향 대칭 상관으로 굽힘(> +0.5)·비틀림(< −0.5)을
    가른다.

    **군집(축퇴쌍)은 구성 모드를 합쳐 잰다.** 대칭 상관의 분자·분모를 구성 모드에 걸쳐 더하면
    Σ_m (L₁u_m)·(L₂u_m) = tr(L₁UUᵀL₂ᵀ)라 쌍 안의 직교 회전에 불변이다. `fractions`는 군집 전체로
    모은 값을 준다. 모드 하나이면 이전 식과 같다."""
    dom = _dominant(fractions)
    if "z" not in dom:
        # y·x-지배 모드의 u_z는 두께 중앙면에서 대칭상 0이다 — 남는 값은 풀이 잡음이다.
        return {"dominant": dom, "width_symmetry": None,
                "character": {"y": "lateral", "x": "axial"}.get(dom, "mixed")}
    U = np.asarray(U_lines, float)
    uz = (U if U.ndim == 4 else U[None])[..., 2]
    a, b = uz[..., 0], uz[..., 1]
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    c = float((a * b).sum()) / den if den > 1e-30 else float("nan")
    kind = ("bending" if c > CHARACTER_THRESHOLD else
            "torsion" if c < -CHARACTER_THRESHOLD else "mixed")
    return {"dominant": dom, "width_symmetry": c, "character": kind}


def _render_points(g):
    axes = [np.linspace(0.0, g[k], n) for k, n in zip(("L", "b", "h"), ATLAS_GRID)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def mode_atlas(outdir=None) -> dict:
    """세 구조 × 모드 1–6의 성격(JSON)과 모드 1–4의 형상(CSV)을 쓴다."""
    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    shape_rows, structures = [], {}
    for tag, geo_delta, faces, ref_csv in ATLAS_STRUCTURES:
        g = {**p5.P5_GEOMETRY, **geo_delta}
        # 형상을 커밋하므로 시작벡터를 고정한다 — 기본값이면 실행마다 마지막 자리가 달라진다.
        fine = p5.solve(*ATLAS_LEVEL, n_modes=ATLAS_N_MODES, geometry=g, faces=faces,
                        start_vector="ones")
        coarse = p5.solve(*ATLAS_CHECK_LEVEL, n_modes=ATLAS_N_MODES, geometry=g,
                          faces=faces, start_vector="ones")
        frac = component_fractions(fine, geometry=g)
        sym_pts, n_x = _symmetry_points(g)
        rpts = _render_points(g)
        # Λ*는 커밋된 참조에서 읽는다 — 출력 폴더가 어디든 같은 참조를 가리켜야 한다.
        lam_star = [float(r["lam_star"]) for r in manifest.read_csv(
            os.path.join(manifest.P5_OUTDIR, ref_csv))]
        modes = []
        for m in range(ATLAS_N_MODES):
            U_sym = p5.eval_mode_at(fine["mesh"], fine["vec_full"][:, m], sym_pts)
            ch = mode_character(U_sym.reshape(n_x, len(SYMMETRY_LINES), 3), frac[m])
            Uf = p5.eval_mode_at(fine["mesh"], fine["vec_full"][:, m], rpts)
            Uc = p5.eval_mode_at(coarse["mesh"], coarse["vec_full"][:, m], rpts)
            modes.append({"mode": m + 1, "lam_fem": float(fine["lam"][m]),
                          "lam_star": lam_star[m],
                          "fx": float(frac[m, 0]), "fy": float(frac[m, 1]),
                          "fz": float(frac[m, 2]),
                          "self_mac_levels": p5.point_mac(Uf, Uc), **ch})
            if m < ATLAS_N_SHAPES:
                Us = normalize_sign(Uf)
                for (x, y, z), (ux, uy, uz) in zip(rpts, Us):
                    shape_rows.append({"structure": tag, "mode": m + 1,
                                       "x": float(f"{x:.6g}"), "y": float(f"{y:.6g}"),
                                       "z": float(f"{z:.6g}"),
                                       "ux": float(f"{ux:.6g}"), "uy": float(f"{uy:.6g}"),
                                       "uz": float(f"{uz:.6g}")})
        structures[tag] = {"geometry": g, "faces": faces, "modes": modes,
                           "dof": int(fine["n_dof"]), "solver": fine["solver"]}
    manifest.write_csv(os.path.join(d, "p5_mode_shapes.csv"), shape_rows)
    manifest.write_json(os.path.join(d, "p5_mode_character.json"),
                        {"level": list(ATLAS_LEVEL), "check_level": list(ATLAS_CHECK_LEVEL),
                         "render_grid": list(ATLAS_GRID),
                         "fraction_grid": list(FRACTION_GRID),
                         "fraction_definition": "mass-weighted: int rho u_c^2 / int rho |u|^2",
                         "character_rule": "z-dominant modes split by width-wise symmetry of "
                                           "u_z (> +0.5 bending, < -0.5 torsion); undefined "
                                           "(null) for y- and x-dominant modes, whose u_z "
                                           "vanishes on the mid-thickness plane by symmetry",
                         "start_vector": "ones",
                         "structures": structures})
    manifest.write_json(os.path.join(d, "manifest_p5_mode_atlas.json"), manifest.build({
        "driver": "run_p5.mode_atlas", "level": list(ATLAS_LEVEL),
        "check_level": list(ATLAS_CHECK_LEVEL), "render_grid": list(ATLAS_GRID),
        "structures": [t for t, *_ in ATLAS_STRUCTURES],
        "sign": "each mode normalised so its largest-magnitude component is positive"}))
    return structures


# ─────────────────────────────────────────────────────────────────────────────
# 구조별 참조(E16 정사각 단면의 전제).
#
# E13의 두 참조는 커밋된 함수 없이 만들어졌다(2026-09-24, 49460de) — 이번에는 사다리·군집·
# 성격·형상 인증을 한 함수로 남긴다. **축퇴쌍은 부분공간으로 인증한다**: 쌍 안의 회전은
# 단계마다 달라 모드별 self-MAC이 원리적으로 실패한다(`p5.point_subspace_mac`). 시작벡터를
# 고정해 쌍 안의 회전까지 실행마다 같게 한다.
# ─────────────────────────────────────────────────────────────────────────────

# 정사각 단면(b = h)용 대칭 사다리. y·z 요소 수가 같아야 이산 문제도 y ↔ z 대칭이고 굽힘
# 쌍이 반올림 수준까지 축퇴한다 — 비대칭 메시는 쌍을 인위적으로 벌린다.
SQUARE_LADDER = [(12, 3, 3), (24, 6, 6), (48, 12, 12)]
REFERENCE_CERT_GRID = (16, 8, 8)


def structure_reference(tag: str, geo_delta: dict, faces: str = "x0", ladder=None,
                        n_modes: int = N_MODES_REF, n_used: int = N_MODES_USED,
                        cert_grid=REFERENCE_CERT_GRID, outdir=None) -> dict:
    """구조 하나의 참조: 3단 사다리 Richardson, 간격 군집, 군집 단위 성격·형상 인증.

    산출 p5_reference_<tag>.csv(신경 드라이버의 `_reference`가 읽는 열을 포함),
    p5_reference_<tag>.json, manifest_p5_reference_<tag>.json.

    - **군집**은 `GAP_RULE`로 Λ*에서 묶는다(외삽이 깨진 모드는 최상단 값). 단계마다 따로 묶은
      결과도 남겨, 세밀화 중 순서가 뒤바뀐 모드를 드러낸다.
    - **성분 비율·성격은 군집으로 모은다.** 쌍 구성 모드의 개별 비율은 쌍 안의 회전에 달린
      값이라 모드 행에는 기록만 하고, 지배 성분과 성격은 군집 행에서 읽는다.
    - **형상 인증은 인접 단계 사이 부분공간의 최소 cos²**다. 단일 모드에서는 self-MAC과 같다.
      채점이 읽는 점(`cert_grid`)에서 잰다."""
    from .run_p5_neural import clusters_from_gaps

    d = manifest.ensure_outdir(outdir, default=manifest.P5_OUTDIR)
    ladder = [tuple(int(v) for v in lv) for lv in (LADDER if ladder is None else ladder)]
    cert_grid = tuple(int(v) for v in cert_grid)
    g = {**p5.P5_GEOMETRY, **geo_delta}
    sols = []
    for lv in ladder:
        s = p5.solve(*lv, n_modes=n_modes, geometry=g, faces=faces, start_vector="ones")
        sols.append(s)
        print(f"[p5 ref {tag}] {lv} 자유도 {s['n_dof']:6d} ({s['solver']}) "
              f"Λ₁ = {s['lam'][0]:.6e}")
    lams = np.array([s["lam"] for s in sols])
    rc = p5.richardson([lv[0] for lv in ladder], lams)
    star = rc["lam_star"]
    basis = np.where(np.isfinite(star), star, lams[-1])
    clusters = clusters_from_gaps([float(v) for v in basis])
    level_clusters = [clusters_from_gaps([float(v) for v in lv]) for lv in lams]
    used = [c for c in clusters if min(c) <= n_used]
    # 마지막으로 푼 모드를 포함한 군집은 그 위의 모드와도 묶일 수 있어 잘렸을 수 있다.
    complete = max(max(c) for c in used) < n_modes

    pts, _ = p5._tensor_grid(cert_grid, g)
    fields = [[p5.eval_mode_at(s["mesh"], s["vec_full"][:, m], pts) for m in range(n_modes)]
              for s in sols]
    fine = sols[-1]
    f_pts, f_w = p5._tensor_grid(FRACTION_GRID, g)
    energy = np.array([g["rho"] * np.einsum(
        "q,qc->c", f_w, p5.eval_mode_at(fine["mesh"], fine["vec_full"][:, m], f_pts) ** 2)
        for m in range(n_modes)])
    sym_pts, n_x = _symmetry_points(g)
    lines = [p5.eval_mode_at(fine["mesh"], fine["vec_full"][:, m], sym_pts)
             .reshape(n_x, len(SYMMETRY_LINES), 3) for m in range(n_modes)]

    cluster_rows = []
    for c in clusters:
        idx = [m - 1 for m in c]
        e = energy[idx].sum(axis=0)
        f = e / e.sum()
        ch = mode_character(np.stack([lines[i] for i in idx]) if len(c) > 1
                            else lines[idx[0]], f)
        per_pair = [p5.point_subspace_mac([fields[i][k] for k in idx],
                                          [fields[i + 1][k] for k in idx])
                    for i in range(len(ladder) - 1)]
        prefix = "" if len(c) == 1 else "pair_" if len(c) == 2 else f"cluster{len(c)}_"
        cluster_rows.append({
            "cluster": list(c), "size": len(c),
            "fx": float(f[0]), "fy": float(f[1]), "fz": float(f[2]),
            "dominant": ch["dominant"], "width_symmetry": ch["width_symmetry"],
            "character": prefix + ch["character"],
            "shape_cert": float(min(per_pair)), "per_pair": [float(v) for v in per_pair],
            "relative_spread": float((basis[idx[-1]] - basis[idx[0]]) / abs(basis[idx[0]]))})
    by_mode = {m: row for row in cluster_rows for m in row["cluster"]}

    modes = []
    for m in range(1, n_modes + 1):
        cr, fm = by_mode[m], energy[m - 1] / energy[m - 1].sum()
        modes.append({
            "mode": m, "lam_star": float(star[m - 1]), "order": float(rc["rate"][m - 1]),
            "increment": float(rc["rel_residual"][m - 1]),
            "monotone": bool(rc["monotone"][m - 1]), "used_in_grid": bool(m <= n_used),
            "lam_levels": [float(v) for v in lams[:, m - 1]],
            "fx": float(fm[0]), "fy": float(fm[1]), "fz": float(fm[2]),
            "cluster": list(cr["cluster"]), "dominant": cr["dominant"],
            "width_symmetry": cr["width_symmetry"], "character": cr["character"],
            "shape_cert": cr["shape_cert"]})
    gaps = [float((basis[i + 1] - basis[i]) / abs(basis[i])) for i in range(n_modes - 1)]
    worst_res = float(np.nanmax(rc["rel_residual"][:n_used]))
    worst_cert = float(min(r["shape_cert"] for r in cluster_rows if min(r["cluster"]) <= n_used))
    consistent = all(lc == clusters for lc in level_clusters)

    manifest.write_csv(os.path.join(d, f"p5_reference_{tag}.csv"), [
        {"mode": r["mode"], "lam_star": r["lam_star"], "rel_residual": r["increment"],
         "rate": r["order"], "monotone": r["monotone"], "used_in_grid": r["used_in_grid"],
         **{f"lam_level{i + 1}": v for i, v in enumerate(r["lam_levels"])},
         "cluster": "-".join(map(str, r["cluster"]))} for r in modes])
    out = {"tag": tag, "geometry": g, "faces": faces, "ladder": [list(lv) for lv in ladder],
           "cert_grid": list(cert_grid), "fraction_grid": list(FRACTION_GRID),
           "start_vector": "ones", "gap_rule": GAP_RULE,
           "clusters": [list(c) for c in clusters],
           "level_clusters": [[list(c) for c in lc] for lc in level_clusters],
           "clusters_consistent_across_levels": bool(consistent),
           "clusters_complete": bool(complete), "gaps": gaps,
           "worst_extrapolation_residual": worst_res, "worst_shape_cert": worst_cert,
           "shape_certified": bool(worst_cert >= p5.SELF_MAC_THRESHOLD),
           "dof_per_level": [int(s["n_dof"]) for s in sols],
           "solver_per_level": [s["solver"] for s in sols],
           "modes": modes, "cluster_rows": cluster_rows,
           "fraction_note": "per-mode fractions of cluster members depend on the arbitrary "
                            "rotation inside the degenerate eigenspace; read dominant and "
                            "character from the cluster rows",
           "certification_note": "minimum cos^2 of the principal angles between adjacent "
                                 "ladder levels, unweighted point samples at cert_grid; "
                                 "equals the per-mode self-MAC for a singleton"}
    manifest.write_json(os.path.join(d, f"p5_reference_{tag}.json"), out)
    manifest.write_json(os.path.join(d, f"manifest_p5_reference_{tag}.json"), manifest.build({
        "driver": "run_p5.structure_reference", "tag": tag, "geometry": g, "faces": faces,
        "ladder": [list(lv) for lv in ladder], "cert_grid": list(cert_grid),
        "n_modes": n_modes, "n_used": n_used, "clusters": [list(c) for c in clusters],
        "worst_extrapolation_residual": worst_res, "worst_shape_cert": worst_cert}))
    print(f"[p5 ref {tag}] 군집 {clusters} · 최대 외삽 잔차 {worst_res:.2e} · "
          f"최저 형상 인증 {worst_cert:.7f}")
    return {**out, "clusters": [tuple(c) for c in clusters]}


if __name__ == "__main__":
    print(main())
