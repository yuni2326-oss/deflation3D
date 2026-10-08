"""P5 — 무차원 3D 고정-자유 세장 블록의 선형탄성 고유치 문제.

    Ω = (0, L) × (0, b) × (0, h),  x = 0 면 고정, 나머지 자유
    a(u,v) = ∫ ε(u):C:ε(v),  b(u,v) = ∫ ρ u·v,  Λ = ω²ρL²/E

**P5가 추가하는 난점 하나.** 벡터장 3성분과 **근접한 모드 패밀리**(굽힘·측방·비틀림)다.
복잡 형상·순환대칭·재진입 모서리는 의도적으로 배제한다 — 함께 넣으면 실패를 어느
난점에 귀속할지 알 수 없다(P1–P4의 "문제 하나에 난점 하나" 규약).

**요소는 27절점 hex를 쓴다.** P4의 9절점 Lagrange Q2를 그대로 텐서곱으로 올린 것이라
형상함수가 1D 2차의 곱으로 닫힌 식이고, serendipity hex20처럼 절점 배치가 특수하지
않아 검산이 쉽다. 영역이 볼록 직육면체이므로 등급 메시도, 거리함수 구성도 필요 없다.
"""
from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp

from ..quadrature import gauss_legendre

# **무차원 규칙**(논문 경계, 2026-08-05): 절대 치수·재료물성·Hz를 쓰지 않는다.
# L = E = ρ = 1로 두면 Λ = ω²ρL²/E가 곧 ω²이고, 형상은 b/L, h/L과 ν만으로 정해진다.
P5_GEOMETRY = {"L": 1.0, "b": 0.20, "h": 0.04, "E": 1.0, "rho": 1.0, "nu": 0.29}

_S1D = (-1.0, 0.0, 1.0)


def _shape_1d(s):
    """1D 2차 Lagrange(절점 −1, 0, 1)의 값과 도함수. P4와 같은 식이다."""
    return (np.array([s * (s - 1) / 2, 1 - s ** 2, s * (s + 1) / 2]),
            np.array([s - 0.5, -2 * s, s + 0.5]))


def hex27_shape(xi, eta, zeta):
    """27절점 hex의 N(27,)과 국소도함수 dN(27,3). 절점 순서 9k + 3j + i."""
    Lx, dLx = _shape_1d(xi)
    Ly, dLy = _shape_1d(eta)
    Lz, dLz = _shape_1d(zeta)
    N = np.empty(27)
    dN = np.empty((27, 3))
    for k in range(3):
        for j in range(3):
            for i in range(3):
                a = 9 * k + 3 * j + i
                N[a] = Lx[i] * Ly[j] * Lz[k]
                dN[a, 0] = dLx[i] * Ly[j] * Lz[k]
                dN[a, 1] = Lx[i] * dLy[j] * Lz[k]
                dN[a, 2] = Lx[i] * Ly[j] * dLz[k]
    return N, dN


def _axis_nodes(n: int, length: float) -> np.ndarray:
    """요소 n개를 균등 분할한 축의 절점 좌표(중간절점 포함), (2n+1,).

    등급 배치를 쓰지 않는다 — P4에서 과도 등급이 오히려 해로웠고(β=5 < β=3),
    직육면체에는 특이점이 없어 균등 세분이 최선이다."""
    v = np.linspace(0.0, length, n + 1)
    out = np.empty(2 * n + 1)
    out[0::2] = v
    out[1::2] = 0.5 * (v[:-1] + v[1:])
    return out


def _is_l(g) -> bool:
    return g is not None and g.get("kind") == "L"


def _is_bilayer(g) -> bool:
    return g is not None and g.get("kind") == "bilayer"


def young_at(pts, g):
    """점마다의 영률. 두 층 구조(E22)는 z < h/2에서 E, 그 위에서 E_top이다. 그 밖의 구조는
    커밋된 산술이 그대로이도록 float 하나를 돌려준다."""
    if not _is_bilayer(g):
        return float(g["E"])
    z = np.asarray(pts, float)[:, 2]
    return np.where(z < 0.5 * float(g["h"]), float(g["E"]), float(g["E_top"]))


def l_blocks(nx: int, ny: int, nz: int, g) -> list:
    """L-브래킷을 상자 셋으로: 팔1 [0, L−b]×[0, b], 모서리 [L−b, L]×[0, b], 팔2 [L−b, L]×[b, b+a],
    두께 [0, h]. 반환 [(lo, hi, (nx, ny, nz))]. 길이가 0인 블록은 뺀다.

    (nx, ny, nz)는 팔1 길이 방향·폭·두께의 요소 수다. **평면 안에서 Δx = Δy를 강제한다** — 모서리
    블록이 정사각 요소가 되고, 두 팔이 같은 평면 해상도를 갖는다. 그래서 nx = ny·L/b여야 한다."""
    L, b, h, a = (float(g[k]) for k in ("L", "b", "h", "a"))
    d = b / ny
    if abs(nx * d - L) > 1e-12 * L:
        raise ValueError(f"L-브래킷은 Δx = Δy여야 합니다: nx = ny·L/b = {L / d:g}")
    counts = {"arm1": (L - b) / d, "arm2": a / d}
    for k, v in counts.items():
        if abs(v - round(v)) > 1e-9:
            raise ValueError(f"{k} 길이가 요소 크기의 정수배가 아니다: {v:g}")
    n1, n2 = int(round(counts["arm1"])), int(round(counts["arm2"]))
    out = []
    if n1 > 0:
        out.append(((0.0, 0.0, 0.0), (L - b, b, h), (n1, ny, nz)))
    out.append(((L - b, 0.0, 0.0), (L, b, h), (ny, ny, nz)))
    if n2 > 0:
        out.append(((L - b, b, 0.0), (L, b + a, h), (ny, n2, nz)))
    return out


def build_mesh_blocks(blocks) -> dict:
    """상자 블록들의 hex27 메시를 합치고 경계면 절점을 좌표로 병합한다.

    블록마다 `build_mesh`와 같은 절점·요소 순서를 쓰고, 공유면의 절점 좌표는 두 블록에서 같은
    산술로 나오므로 정확히 일치한다(병합은 12자리 반올림으로 안전하게)."""
    nodes, elems, meta, n0, e0 = [], [], [], 0, 0
    for lo, hi, (nx, ny, nz) in blocks:
        ax = [lo[i] + _axis_nodes(n, hi[i] - lo[i]) for i, n in enumerate((nx, ny, nz))]
        Nx, Ny, Nz = len(ax[0]), len(ax[1]), len(ax[2])
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        nodes.append(np.column_stack([X.ravel(order="F"), Y.ravel(order="F"),
                                      Z.ravel(order="F")]))
        idx = np.arange(Nx * Ny * Nz).reshape((Nx, Ny, Nz), order="F") + n0
        el = np.empty((nx * ny * nz, 27), dtype=int)
        e = 0
        for ez in range(nz):
            for ey in range(ny):
                for ex in range(nx):
                    for k in range(3):
                        for j in range(3):
                            for i in range(3):
                                el[e, 9 * k + 3 * j + i] = idx[2 * ex + i, 2 * ey + j, 2 * ez + k]
                    e += 1
        elems.append(el)
        meta.append({"lo": np.array(lo, float), "hi": np.array(hi, float),
                     "shape": (nx, ny, nz), "elem_offset": e0})
        n0 += Nx * Ny * Nz
        e0 += nx * ny * nz
    allnodes = np.vstack(nodes)
    _, first, inverse = np.unique(np.round(allnodes, 12), axis=0, return_index=True,
                                  return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    order = np.argsort(first)                       # 처음 나온 순서를 유지한다
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    remap = rank[inverse]
    merged = allnodes[first[order]]
    return {"nodes": merged, "elems": remap[np.vstack(elems)], "shape": None, "blocks": meta}


def build_mesh(nx: int, ny: int, nz: int, geometry=None) -> dict:
    """구조 hex27 메시. 반환 {nodes(n_node,3), elems(n_elem,27), shape}.

    `geometry["kind"] == "L"`이면 L-브래킷의 다중 블록 메시다(`l_blocks`)."""
    if min(nx, ny, nz) < 1:
        raise ValueError("요소 수는 축마다 1 이상이어야 합니다")
    if _is_l(geometry):
        return build_mesh_blocks(l_blocks(nx, ny, nz, geometry))
    if _is_bilayer(geometry) and nz % 2:
        raise ValueError(f"두 층 구조는 계면 z = h/2가 요소면이어야 한다(n_z 짝수): n_z = {nz}")
    g = P5_GEOMETRY if geometry is None else geometry
    ax = _axis_nodes(nx, g["L"])
    ay = _axis_nodes(ny, g["b"])
    az = _axis_nodes(nz, g["h"])
    Nx, Ny, Nz = len(ax), len(ay), len(az)

    # 절점 인덱스 = ix + Nx*(iy + Ny*iz) — 축 방향으로 붙어 있어 캐시에 유리하다
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    nodes = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"),
                             Z.ravel(order="F")])
    # meshgrid(indexing="ij") + ravel(order="F")는 ix가 가장 빨리 도는 순서다
    idx = np.arange(Nx * Ny * Nz).reshape((Nx, Ny, Nz), order="F")

    elems = np.empty((nx * ny * nz, 27), dtype=int)
    e = 0
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                for k in range(3):
                    for j in range(3):
                        for i in range(3):
                            elems[e, 9 * k + 3 * j + i] = idx[2 * ex + i,
                                                              2 * ey + j,
                                                              2 * ez + k]
                e += 1
    return {"nodes": nodes, "elems": elems, "shape": (nx, ny, nz)}


def isotropic_D(E: float, nu: float) -> np.ndarray:
    """3D 등방 탄성행렬 (6,6), Voigt 순서 (xx, yy, zz, yz, xz, xy).

    **평면응력이 아니다.** P4는 얇은 판이라 평면응력을 썼지만 P5는 3차원 블록이므로
    완전 3D 구성식을 쓴다 — 여기서 잘못 줄이면 비틀림 강성이 틀어진다."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def assemble(mesh: dict, geometry=None, n_q: int = 3):
    """BC 미부과 전체 (K, M) — CSR. 절점당 3자유도, dof = 3*node + c."""
    g = P5_GEOMETRY if geometry is None else geometry
    D = isotropic_D(g["E"], g["nu"])
    rho = g["rho"]
    # E22 두 층: 요소 중심이 계면 위면 윗층 재료(계면은 요소면이다 — build_mesh가 보장).
    bilayer = _is_bilayer(g)
    if bilayer:
        D_top, z_mid = isotropic_D(g["E_top"], g["nu"]), 0.5 * float(g["h"])
    xyz, elems = mesh["nodes"], mesh["elems"]
    n_dof = 3 * xyz.shape[0]
    gp, gw = gauss_legendre(n_q, -1.0, 1.0)

    # 요소별 (N, dN)은 격자에만 의존하므로 한 번 만들어 재사용한다
    quad = []
    for a, xi in enumerate(gp):
        for b_, et in enumerate(gp):
            for c_, ze in enumerate(gp):
                N, dN = hex27_shape(xi, et, ze)
                quad.append((N, dN, gw[a] * gw[b_] * gw[c_]))

    rows, cols, kv, mv = [], [], [], []
    for el in elems:
        ce = xyz[el]                                     # (27,3)
        De = D_top if bilayer and ce[:, 2].mean() > z_mid else D
        Ke = np.zeros((81, 81))
        Me = np.zeros((81, 81))
        for N, dN, w0 in quad:
            J = dN.T @ ce                                # (3,3)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise RuntimeError("야코비안이 비양수 — 메시가 뒤집혔다")
            dNx = np.linalg.solve(J, dN.T).T             # (27,3) 물리 도함수
            B = np.zeros((6, 81))
            B[0, 0::3] = dNx[:, 0]
            B[1, 1::3] = dNx[:, 1]
            B[2, 2::3] = dNx[:, 2]
            B[3, 1::3] = dNx[:, 2]
            B[3, 2::3] = dNx[:, 1]
            B[4, 0::3] = dNx[:, 2]
            B[4, 2::3] = dNx[:, 0]
            B[5, 0::3] = dNx[:, 1]
            B[5, 1::3] = dNx[:, 0]
            wq = w0 * detJ
            Ke += wq * (B.T @ De @ B)
            NtN = np.outer(N, N) * (wq * rho)
            for c in range(3):
                Me[c::3, c::3] += NtN
        dofs = (3 * el[:, None] + np.arange(3)).ravel()
        R, C = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(R.ravel()); cols.append(C.ravel())
        kv.append(Ke.ravel()); mv.append(Me.ravel())

    rows = np.concatenate(rows); cols = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kv), (rows, cols)),
                      shape=(n_dof, n_dof)).tocsr()
    M = sp.coo_matrix((np.concatenate(mv), (rows, cols)),
                      shape=(n_dof, n_dof)).tocsr()
    return K, M


def apply_clamped(K, M, mesh: dict, tol: float = 1e-12, faces: str = "x0",
                  geometry=None):
    """고정면의 세 성분을 제거. 반환 (K_c, M_c, free(bool, n_dof)).

    `faces`는 `"x0"`(커밋된 한 면, 기본값)와 `"two_face"`(x = 0과 x = L, E13-b)다.
    기본값을 바꾸지 않으므로 커밋된 참조는 그대로 재현된다."""
    g = P5_GEOMETRY if geometry is None else geometry
    xyz = mesh["nodes"]
    fixed_node = np.abs(xyz[:, 0]) < tol
    if faces == "two_face":
        fixed_node |= np.abs(xyz[:, 0] - float(g["L"])) < tol
    elif faces != "x0":
        raise ValueError(f"faces는 x0|two_face여야 합니다: {faces!r}")
    free = np.repeat(~fixed_node, 3)
    return K[free][:, free].tocsr(), M[free][:, free].tocsr(), free


# 이 크기를 넘으면 밀집 eigh가 메모리에 들어가지 않는다(자유도² × 8 B × 사본 몇 개).
# 참조 격자의 최상단이 2만 자유도를 넘으므로 희소 경로가 필요하다.
DENSE_MAX_DOF = 6000


START_VECTORS = (None, "ones")


def solve(nx: int, ny: int, nz: int, n_modes: int = 6, geometry=None,
          n_q: int = 3, solver: str | None = None, faces: str = "x0",
          start_vector: str | None = None) -> dict:
    """가장 낮은 n_modes개. 반환 lam, vec_full(n_dof, n_modes), mesh, free, solver.

    `solver`는 "dense"|"sparse"|None(크기로 자동). 어느 경로로 풀었는지 결과에
    남긴다 — 참조값의 출처이므로 manifest에 실려야 한다.

    `start_vector`는 희소 경로의 ARPACK 시작벡터다. None(기본)이면 ARPACK이 난수로
    고르므로 고유벡터가 실행마다 마지막 자리에서 달라진다 — 커밋된 모든 MAC이 이 기본값
    위에 있으므로 바꾸지 않는다. "ones"는 모드 도감처럼 형상 자체를 산출물로 커밋할 때
    쓴다."""
    if start_vector not in START_VECTORS:
        raise ValueError(f"start_vector는 {START_VECTORS} 중 하나여야 합니다: {start_vector!r}")
    mesh = build_mesh(nx, ny, nz, geometry)
    K, M = assemble(mesh, geometry, n_q)
    Kc, Mc, free = apply_clamped(K, M, mesh, faces=faces, geometry=geometry)
    n_free = int(free.sum())
    how = solver or ("dense" if n_free <= DENSE_MAX_DOF else "sparse")

    if how == "dense":
        from scipy.linalg import eigh
        w_all, V_all = eigh(Kc.toarray(), Mc.toarray())
        order = np.argsort(w_all)[:n_modes]
        w, V = w_all[order], V_all[:, order]
    elif how == "sparse":
        from scipy.sparse.linalg import eigsh
        # **shift-invert가 필요하다.** which="SM"을 그대로 쓰면 0 근처 수렴이 느리고
        # 불안정하다. σ = 0으로 이동하면 가장 작은 쪽이 가장 큰 쪽으로 뒤집힌다.
        v0 = None if start_vector is None else np.ones(Kc.shape[0])
        w, V = eigsh(Kc.tocsc(), k=n_modes, M=Mc.tocsc(), sigma=0.0,
                     which="LM", v0=v0)
        order = np.argsort(w)
        w, V = w[order], V[:, order]
    else:
        raise ValueError(f"solver는 dense|sparse여야 합니다: {solver!r}")

    vec = np.zeros((K.shape[0], len(w)))
    vec[free, :] = V
    return {"lam": w, "vec_full": vec, "mesh": mesh, "free": free,
            "n_dof": n_free, "n_q": n_q, "levels": (nx, ny, nz),
            "solver": how}


def richardson(levels, values) -> dict:
    """세 단계 h, h/2, h/4의 값에서 외삽값·관측 수렴차수·상대 잔차.

    P4와 같은 모델 λ_h = λ_* + C h^p. 단조성이 깨지면(d1/d2 ≤ 1) NaN·∞를 돌려
    "수렴하지 않았다"를 숨기지 않는다."""
    v = np.asarray(values, float)
    if v.shape[0] != 3:
        raise ValueError("세 단계가 필요합니다")
    d1, d2 = v[0] - v[1], v[1] - v[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = d1 / d2
        p = np.log2(np.where(ratio > 1.0, ratio, np.nan))
        star = v[2] - d2 / (2 ** p - 1)
        rel = np.abs(v[2] - star) / np.abs(star)
    bad = ~(ratio > 1.0)
    star = np.where(bad, np.nan, star)
    rel = np.where(bad, np.inf, rel)
    return {"levels": list(levels), "lam_star": star, "rate": p,
            "rel_residual": rel, "monotone": ~bad}


def eval_mode_at(mesh: dict, vec_full: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """모드 벡터를 임의 점에서 평가. 반환 (n_pts, 3).

    구조격자이므로 점이 든 요소를 좌표로 바로 찾는다 — 탐색이 필요 없다."""
    if mesh.get("blocks") is not None:
        e, loc = _locate(mesh, pts)
        out = np.empty((len(e), 3))
        for q in range(len(e)):
            N, _ = hex27_shape(*loc[q])
            dofs = (3 * mesh["elems"][e[q]][:, None] + np.arange(3))
            out[q] = (N[:, None] * vec_full[dofs]).sum(axis=0)
        return out
    g_nodes = mesh["nodes"]
    nx, ny, nz = mesh["shape"]
    hi = g_nodes.max(axis=0)
    pts = np.atleast_2d(np.asarray(pts, float))
    out = np.empty((pts.shape[0], 3))
    ne = np.array([nx, ny, nz])
    for q, p in enumerate(pts):
        t = np.clip(p / hi, 0.0, 1.0)
        cell = np.minimum((t * ne).astype(int), ne - 1)
        # 셀 내부 국소좌표 [-1, 1]
        lo = cell / ne * hi
        loc = 2.0 * (p - lo) / (hi / ne) - 1.0
        loc = np.clip(loc, -1.0, 1.0)
        e = cell[0] + nx * (cell[1] + ny * cell[2])
        N, _ = hex27_shape(*loc)
        el = mesh["elems"][e]
        dofs = (3 * el[:, None] + np.arange(3))
        out[q] = (N[:, None] * vec_full[dofs]).sum(axis=0)
    return out


def _locate(mesh: dict, pts: np.ndarray):
    """구조격자이므로 점이 든 요소와 국소좌표를 좌표로 바로 계산한다 — 탐색 없음.
    다중 블록 메시는 점이 든 블록을 먼저 고른다(공유면 위의 점은 어느 쪽이든 같은 값이다)."""
    if mesh.get("blocks") is not None:
        pts = np.atleast_2d(np.asarray(pts, float))
        e = np.full(len(pts), -1, dtype=int)
        loc = np.zeros((len(pts), 3))
        for blk in mesh["blocks"]:
            lo, hi, ne = blk["lo"], blk["hi"], np.array(blk["shape"])
            span = hi - lo
            tol = 1e-12 * span.max()
            inside = (e < 0) & np.all((pts >= lo - tol) & (pts <= hi + tol), axis=1)
            if not inside.any():
                continue
            t = np.clip((pts[inside] - lo) / span, 0.0, 1.0)
            cell = np.minimum((t * ne).astype(int), ne - 1)
            c_lo = lo + cell / ne * span
            loc[inside] = np.clip(2.0 * (pts[inside] - c_lo) / (span / ne) - 1.0, -1.0, 1.0)
            e[inside] = blk["elem_offset"] + cell[:, 0] + ne[0] * (cell[:, 1] + ne[1] * cell[:, 2])
        if (e < 0).any():
            raise ValueError(f"영역 밖의 점 {int((e < 0).sum())}개")
        return e, loc
    hi = mesh["nodes"].max(axis=0)
    nx, ny, nz = mesh["shape"]
    ne = np.array([nx, ny, nz])
    pts = np.atleast_2d(np.asarray(pts, float))
    t = np.clip(pts / hi, 0.0, 1.0)
    cell = np.minimum((t * ne).astype(int), ne - 1)
    lo = cell / ne * hi
    loc = np.clip(2.0 * (pts - lo) / (hi / ne) - 1.0, -1.0, 1.0)
    e = cell[:, 0] + nx * (cell[:, 1] + ny * cell[:, 2])
    return e, loc


def eval_grad_at(mesh: dict, vec_full: np.ndarray, pts) -> np.ndarray:
    """변위 기울기 ∂u_i/∂x_j를 임의 점에서. 반환 (n_pts, 3, 3).

    a(u,u) = ∫ε:C:ε를 요소 조립이 아니라 **텐서 Gauss로** 재려면 필요하다
    (신경 arm이 쓰는 것과 같은 구적이므로, 그 구적의 충분성을 이걸로 시험한다)."""
    e, loc = _locate(mesh, pts)
    xyz, elems = mesh["nodes"], mesh["elems"]
    out = np.empty((len(e), 3, 3))
    for q in range(len(e)):
        el = elems[e[q]]
        _, dN = hex27_shape(*loc[q])
        J = dN.T @ xyz[el]
        dNx = np.linalg.solve(J, dN.T).T                  # (27,3)
        ue = vec_full[(3 * el[:, None] + np.arange(3))]    # (27,3)
        out[q] = ue.T @ dNx                               # ∂u_i/∂x_j
    return out


def tensor_gauss_forms(mesh: dict, vec_full: np.ndarray, grid, geometry=None):
    """(a(u,u), b(u,u))를 영역 전체의 텐서 Gauss로 적분. grid = (nx, ny, nz) 점수.

    **신경 arm이 쓰는 구적을 그대로 쓴다.** 영역이 볼록 직육면체이므로 메시 기반
    구적이 필요 없고, 여기서 재는 값이 신경 Rayleigh 몫의 분자·분모와 같은 근사다."""
    g = P5_GEOMETRY if geometry is None else geometry
    D = isotropic_D(g["E"], g["nu"])
    hi = np.array([g["L"], g["b"], g["h"]])
    axes, wts = [], []
    for n, length in zip(grid, hi):
        p, w = gauss_legendre(int(n), 0.0, float(length))
        axes.append(p); wts.append(w)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    W = np.einsum("i,j,k->ijk", *wts).ravel()
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    G = eval_grad_at(mesh, vec_full, pts)                 # (nq,3,3)
    eps = np.empty((len(pts), 6))
    eps[:, 0] = G[:, 0, 0]
    eps[:, 1] = G[:, 1, 1]
    eps[:, 2] = G[:, 2, 2]
    eps[:, 3] = G[:, 1, 2] + G[:, 2, 1]                   # 공학 전단 γ_yz
    eps[:, 4] = G[:, 0, 2] + G[:, 2, 0]
    eps[:, 5] = G[:, 0, 1] + G[:, 1, 0]
    a = float(np.einsum("q,qi,ij,qj->", W, eps, D, eps))
    u = eval_mode_at(mesh, vec_full, pts)                 # (nq,3)
    b = float(g["rho"] * np.einsum("q,qi,qi->", W, u, u))
    return a, b


def mode_at(res: dict, mode: int, pts) -> np.ndarray:
    """1-기반 모드 번호로 평가."""
    return eval_mode_at(res["mesh"], res["vec_full"][:, mode - 1], pts)


def point_mac(u, v) -> float:
    """구적점에서 표본된 벡터장 두 개의 MAC. 성분을 이어붙여 계산한다.

    가중을 넣지 않는다 — 채점이 읽는 것은 같은 점집합에서의 형상 일치이고,
    구적 가중을 넣으면 점 배치에 따라 값이 달라져 단계 간 비교가 흐려진다."""
    a = np.asarray(u, float).ravel()
    b = np.asarray(v, float).ravel()
    num = float(a @ b) ** 2
    den = float(a @ a) * float(b @ b)
    if den <= 0.0 or not np.isfinite(num) or not np.isfinite(den):
        return 0.0
    return float(min(max(num / den, 0.0), 1.0))


def point_subspace_mac(Us, Vs) -> float:
    """구적점 표본 벡터장 묶음 두 개가 펼치는 부분공간의 일치 — **최소** 주각의 cos².

    축퇴쌍은 세밀화 단계마다 쌍 안에서 다르게 회전해 나오므로(고유공간 안의 어떤 회전도
    고유벡터다) 모드별 `point_mac`은 원리적으로 실패한다. 부분공간 비교는 그 회전에 불변이다.
    평균이 아니라 최소를 쓰는 것은 한 방향만 어긋나도 인증이 실패해야 하기 때문이다.
    묶음이 하나씩이면 `point_mac`과 같다(가중을 넣지 않는 것도 같다)."""
    A = np.column_stack([np.asarray(u, float).ravel() for u in Us])
    B = np.column_stack([np.asarray(v, float).ravel() for v in Vs])
    if A.shape[1] != B.shape[1]:
        raise ValueError("두 묶음의 모드 수가 같아야 합니다")
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(min(max(float(s.min()) ** 2, 0.0), 1.0))


SELF_MAC_THRESHOLD = 0.999


def certify_mode_shapes(levels, n_modes: int = 6, grid=(16, 8, 6),
                        geometry=None, threshold: float = SELF_MAC_THRESHOLD,
                        n_q: int = 3) -> dict:
    """§2.2 — 인접 세밀화 단계 간 self-MAC을 구적점에서 잰다.

    **채점이 읽는 점에서 재야 한다.** MAC은 표본 위치에 의존하므로, 절점에서 재고
    구적점에서 채점하면 인증한 양과 쓰는 양이 다르다. P1의 교훈(참조도 솔버와 같은
    인증이 필요하다)의 3D 적용이다."""
    if len(levels) < 2:
        raise ValueError("인접 비교에는 두 단계 이상이 필요합니다")
    pts, _ = _tensor_grid(grid, geometry)
    fields = []
    for lv in levels:
        r = solve(*lv, n_modes=n_modes, geometry=geometry, n_q=n_q)
        fields.append([eval_mode_at(r["mesh"], r["vec_full"][:, m], pts)
                       for m in range(n_modes)])
    rows = []
    for m in range(n_modes):
        macs = [point_mac(fields[i][m], fields[i + 1][m])
                for i in range(len(levels) - 1)]
        rows.append({"mode": m + 1, "self_mac": float(min(macs)),
                     "per_pair": [float(v) for v in macs],
                     "n_points": int(len(pts))})
    worst = min(r["self_mac"] for r in rows)
    return {"rows": rows, "levels": [tuple(lv) for lv in levels],
            "grid": tuple(int(v) for v in grid), "threshold": threshold,
            "worst_self_mac": worst, "certified": bool(worst >= threshold)}


# §2.4 인증에 쓰는 파수. modes 1–6의 최고 굽힘(β₃)보다 한 단 위인 β₄를 써서
# 목표 대역을 여유 있게 덮는다.
_CERT_WAVENUMBER = 10.9955407348


def manufactured_field(pts, geometry=None, k: float = _CERT_WAVENUMBER,
                       my: int = 1, mz: int = 1):
    """구적 격자 인증용 **매끄러운** 해석 장과 그 기울기. 반환 (u(n,3), G(n,3,3)).

    신경 시행함수와 같은 구조 u = x·(매끄러운 함수)를 쓴다 — 고정 BC를 정확히
    만족하고 전역에서 해석적이므로, 텐서 Gauss가 지수적으로 수렴한다. FEM 모드를
    쓰면 C⁰ 꺾임 때문에 격자의 해상도와 장의 불연속이 섞인다.

    `my`·`mz`는 **횡방향 조화 차수**다. v1은 축방향 파수만 올리고 횡방향은 반파 하나로
    고정했는데, 3차원에서 격자가 못 보는 성분은 단면 안에도 있다(비틀림·전단). 기본값
    (1, 1)은 v1과 같은 장을 준다 — 커밋된 인증이 조용히 바뀌지 않는다."""
    g = P5_GEOMETRY if geometry is None else geometry
    b, h = g["b"], g["h"]
    p = np.atleast_2d(np.asarray(pts, float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    qy, qz = int(my) * np.pi / b, int(mz) * np.pi / h
    s, c = np.sin(k * x), np.cos(k * x)
    cy, sy = np.cos(qy * y), np.sin(qy * y)
    cz, sz = np.cos(qz * z), np.sin(qz * z)

    u = np.column_stack([x * s * cy, x * c * sz, x * s * (1.0 + y / b)])
    G = np.empty((len(p), 3, 3))
    # ∂u_x
    G[:, 0, 0] = (s + x * k * c) * cy
    G[:, 0, 1] = x * s * (-qy * sy)
    G[:, 0, 2] = 0.0
    # ∂u_y
    G[:, 1, 0] = (c - x * k * s) * sz
    G[:, 1, 1] = 0.0
    G[:, 1, 2] = x * c * (qz * cz)
    # ∂u_z
    G[:, 2, 0] = (s + x * k * c) * (1.0 + y / b)
    G[:, 2, 1] = x * s / b
    G[:, 2, 2] = 0.0
    return u, G


def _forms_from_field(u, G, W, geometry=None, pts=None):
    """(a, b). 두 층 구조는 점마다의 영률로 a를 잰다 — `pts`가 필요하다."""
    g = P5_GEOMETRY if geometry is None else geometry
    D = isotropic_D(g["E"], g["nu"])
    if _is_bilayer(g):
        if pts is None:
            raise ValueError("두 층 구조의 에너지에는 점마다의 영률이 필요하다 — pts를 넘긴다")
        W_a, D = W * young_at(pts, g), isotropic_D(1.0, g["nu"])
    else:
        W_a = W
    eps = np.empty((len(u), 6))
    eps[:, 0] = G[:, 0, 0]
    eps[:, 1] = G[:, 1, 1]
    eps[:, 2] = G[:, 2, 2]
    eps[:, 3] = G[:, 1, 2] + G[:, 2, 1]
    eps[:, 4] = G[:, 0, 2] + G[:, 2, 0]
    eps[:, 5] = G[:, 0, 1] + G[:, 1, 0]
    a = float(np.einsum("q,qi,ij,qj->", W_a, eps, D, eps))
    b = float(g["rho"] * np.einsum("q,qi,qi->", W, u, u))
    return a, b


def l_grid_blocks(grid, g) -> list:
    """L-브래킷의 구적 블록. grid = (nL, nW, nT): 팔의 길이 방향은 단위 길이당 nL점(올림), 폭은 nW점,
    두께는 nT점. 모서리는 nW × nW. 반환 [(lo, hi, (n_x, n_y, n_z))]."""
    nL, nW, nT = (int(v) for v in grid)
    L, b, h, a = (float(g[k]) for k in ("L", "b", "h", "a"))
    up = lambda x: max(1, int(np.ceil(x - 1e-9)))
    out = []
    if L - b > 0:
        out.append(((0.0, 0.0, 0.0), (L - b, b, h), (up(nL * (L - b)), nW, nT)))
    out.append(((L - b, 0.0, 0.0), (L, b, h), (nW, nW, nT)))
    if a > 0:
        out.append(((L - b, b, 0.0), (L, b + a, h), (nW, up(nL * a), nT)))
    return out


def _tensor_grid(grid, geometry=None):
    g = P5_GEOMETRY if geometry is None else geometry
    if _is_l(g):
        P, Wt = [], []
        for lo, hi_, n in l_grid_blocks(grid, g):
            axes, wts = [], []
            for k in range(3):
                p, w = gauss_legendre(int(n[k]), float(lo[k]), float(hi_[k]))
                axes.append(p); wts.append(w)
            X, Y, Z = np.meshgrid(*axes, indexing="ij")
            P.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
            Wt.append(np.einsum("i,j,k->ijk", *wts).ravel())
        return np.vstack(P), np.concatenate(Wt)
    if _is_bilayer(g):
        # 두께 방향은 층마다 grid[2]점의 Gauss — 계면의 불연속을 점이 가로지르지 않는다.
        h2 = 0.5 * float(g["h"])
        axes, wts = [], []
        for n, length in zip(grid[:2], (g["L"], g["b"])):
            p, w = gauss_legendre(int(n), 0.0, float(length))
            axes.append(p); wts.append(w)
        z0, w0 = gauss_legendre(int(grid[2]), 0.0, h2)
        z1, w1 = gauss_legendre(int(grid[2]), h2, float(g["h"]))
        axes.append(np.concatenate([z0, z1])); wts.append(np.concatenate([w0, w1]))
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        W = np.einsum("i,j,k->ijk", *wts).ravel()
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), W
    hi = (g["L"], g["b"], g["h"])
    axes, wts = [], []
    for n, length in zip(grid, hi):
        p, w = gauss_legendre(int(n), 0.0, float(length))
        axes.append(p); wts.append(w)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    W = np.einsum("i,j,k->ijk", *wts).ravel()
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), W


# v2 §2.2-2.3 — 감사 격자. 훈련 격자와 분리해, 게이트·채점·보고되는 모든 양을 여기서 잰다.
AUDIT_FIELDS = ((_CERT_WAVENUMBER, 1, 1),
                (2.0 * _CERT_WAVENUMBER, 1, 1),
                (_CERT_WAVENUMBER, 3, 2),
                (2.0 * _CERT_WAVENUMBER, 3, 2))
AUDIT_AXIS_MARGIN = 1.5


def _field_errors(grid, ref_grid, geometry, fields):
    """후보 격자와 기준 격자에서 각 장의 a·b 상대오차."""
    p_r, w_r = _tensor_grid(ref_grid, geometry)
    p_c, w_c = _tensor_grid(grid, geometry)
    out = []
    for k, my, mz in fields:
        a_r, b_r = _forms_from_field(*manufactured_field(p_r, geometry, k, my, mz),
                                     w_r, geometry, pts=p_r)
        a_c, b_c = _forms_from_field(*manufactured_field(p_c, geometry, k, my, mz),
                                     w_c, geometry, pts=p_c)
        out.append({"k": float(k), "my": int(my), "mz": int(mz),
                    "err_a": abs(a_c / a_r - 1), "err_b": abs(b_c / b_r - 1)})
    return out


def certify_audit_grid(train_family_max, tol: float = 1e-6, geometry=None,
                       fields=AUDIT_FIELDS, reference=(96, 40, 24),
                       margin: float = AUDIT_AXIS_MARGIN, steps: int = 6,
                       fem_ladder=None, fem_modes=(1,)) -> dict:
    """v2 §2.2-2.3 — 훈련 격자족 위에서 인증을 통과하는 **최소** 감사 격자.

    세 조건을 모두 건다. (a) 축마다 훈련 격자족 최대치의 `margin`배 이상 — 훈련이
    본 적 없는 해상도라야 감사가 된다. (b) `fields`의 모든 파장·횡조화에서 a·b가
    `tol` 안 — 하나의 파장만 보면 인증이 약하다. (c) 한 단계 위 격자와 `tol` 안에서
    일치 — 자기 수렴 확인 없이는 (b)가 순환이다.

    `fem_ladder`를 주면 FEM 참조 모드로 잰 오차도 **측정만** 해서 남긴다. v2 §2.3.1은
    이를 1e-6 문턱으로 요구하지만, C⁰ 변형률은 요소면에서 불연속이라 전역 텐서격자로
    도달할 수 없다(P1의 조각경계 정렬 요구의 3차원 판본). 문턱으로 쓰지 않는 대신
    실측 바닥을 남겨, 제작해로 대체한 근거가 검증 가능하게 한다."""
    floor = tuple(int(math.ceil(margin * v)) for v in train_family_max)
    cand = [tuple(floor[i] + j * max(1, floor[i] // 10) for i in range(3))
            for j in range(int(steps))]
    chosen, chosen_fields, cross = None, None, float("nan")
    rows = []
    for grid in cand:
        fe = _field_errors(grid, reference, geometry, fields)
        worst = max(max(f["err_a"], f["err_b"]) for f in fe)
        rows.append({"grid": grid, "n_points": int(len(_tensor_grid(grid, geometry)[1])),
                     "worst": worst})
        if worst > tol or chosen is not None:
            continue
        up = tuple(int(math.ceil(v * 1.25)) for v in grid)
        fu = _field_errors(up, reference, geometry, fields)
        # 두 격자가 기준에 대해 각각 tol 안이면 서로도 tol 안이지만, 계획이 요구한
        # 것은 **서로의 차이**이므로 그대로 잰다.
        cross = max(abs(a["err_a"] - b_["err_a"]) for a, b_ in zip(fe, fu))
        cross = max(cross, max(abs(a["err_b"] - b_["err_b"]) for a, b_ in zip(fe, fu)))
        if cross <= tol:
            chosen, chosen_fields = grid, fe
    out = {"chosen": chosen, "rows": rows, "fields": chosen_fields or [],
           "meets_tol": chosen is not None, "cross_grid_rel_diff": cross,
           "tol": tol, "axis_floor": floor, "margin": margin,
           "train_family_max": tuple(int(v) for v in train_family_max),
           "reference_grid": tuple(reference),
           "fem_gate_applied": False}
    if fem_ladder is not None and chosen is not None:
        sol = solve(*fem_ladder, n_modes=max(fem_modes) + 2, geometry=geometry)
        mesh = sol["mesh"]
        worst = 0.0
        for m in fem_modes:
            vec = sol["vec_full"][:, m - 1]
            a_c, b_c = tensor_gauss_forms(mesh, vec, chosen, geometry)
            a_r, b_r = tensor_gauss_forms(mesh, vec, reference, geometry)
            worst = max(worst, abs(a_c / a_r - 1), abs(b_c / b_r - 1))
        out["fem_mode_floor"] = float(worst)
        out["fem_modes"] = tuple(int(m) for m in fem_modes)
        out["fem_note"] = ("measured, not gated: a C0 FEM strain field is discontinuous "
                           "across element faces, so a global tensor Gauss grid cannot "
                           "integrate it to 1e-6 at any order")
    return out


def quadrature_convergence(grids, tol: float = 1e-6, geometry=None,
                           reference=(96, 40, 24), check=(80, 34, 20)) -> dict:
    """§2.4 — 후보 텐서 Gauss 격자 중 a·b를 tol 안에서 적분하는 **최소** 격자.

    기준값은 후보보다 훨씬 촘촘한 격자에서 얻고, 그 기준 자체가 수렴했는지를 두 번째
    촘촘한 격자와 비교해 확인한다(자기 수렴 확인 없이 기준을 쓰면 인증이 순환이다)."""
    pts, W = _tensor_grid(reference, geometry)
    a_ref, b_ref = _forms_from_field(*manufactured_field(pts, geometry), W, geometry, pts=pts)
    pts2, W2 = _tensor_grid(check, geometry)
    a_chk, b_chk = _forms_from_field(*manufactured_field(pts2, geometry), W2, geometry,
                                     pts=pts2)
    ref_self = max(abs(a_chk / a_ref - 1), abs(b_chk / b_ref - 1))

    rows = []
    for grid in grids:
        p_, w_ = _tensor_grid(grid, geometry)
        a, b = _forms_from_field(*manufactured_field(p_, geometry), w_, geometry, pts=p_)
        rows.append({"grid": tuple(int(v) for v in grid),
                     "n_points": int(len(w_)),
                     "err_a": abs(a / a_ref - 1), "err_b": abs(b / b_ref - 1)})
    ok = [r for r in rows if max(r["err_a"], r["err_b"]) <= tol]
    chosen = min(ok, key=lambda r: r["n_points"])["grid"] if ok else None
    return {"rows": rows, "chosen": chosen, "tol": tol,
            "reference_grid": tuple(reference), "check_grid": tuple(check),
            "reference_self_consistency": ref_self,
            "wavenumber": _CERT_WAVENUMBER}


def beam_theory_estimate(geometry=None) -> dict:
    """§2.5 sanity check — 외팔보 이론의 Λ 추정. **참조가 아니다.**

    굽힘(약축 h)·측방(강축 b)·비틀림 패밀리를 섞어 정렬한다. 이 값의 용도는
    단면비가 modes 1–6에 세 패밀리를 넣으면서 §2.3의 근접 규칙(간격 < 1 %)을
    발동시키지 않는지 실행 전에 확인하는 것이다."""
    g = P5_GEOMETRY if geometry is None else geometry
    L, b, h = g["L"], g["b"], g["h"]
    E, rho, nu = g["E"], g["rho"], g["nu"]
    G = E / (2 * (1 + nu))
    beta = np.array([1.8751040687, 4.6940911330, 7.8547574382, 10.9955407348])
    def bend(t):
        return beta ** 2 * np.sqrt(E * t ** 2 / (12 * rho)) / L ** 2
    Jt = b * h ** 3 / 3                       # 얇은 직사각의 비틀림 상수
    Ip = rho * b * h * (b ** 2 + h ** 2) / 12
    wt = (np.pi / (2 * L)) * np.sqrt(G * Jt / Ip) * np.array([1, 3, 5])
    fam = ([("bending", w) for w in bend(h)] + [("lateral", w) for w in bend(b)]
           + [("torsion", w) for w in wt])
    fam.sort(key=lambda t: t[1])
    lam = np.array([w ** 2 * rho * L ** 2 / E for _, w in fam])
    return {"lam": lam, "family": [f for f, _ in fam], "geometry": dict(g)}
