"""P5 — 3D 탄성의 강형식 물리정보 신경 고유해석(E24, 사전규정 2026-10-17-p5-e24-prespecification.md).

변분형(`p5_neural.solve_p5_neural`)과 **같은 망·하드 BC·감사 격자·사다리·채점 출력**을 쓰고, 학습 목적함수만 강형식이다.

    L = (1/V)∫_Ω |∇·σ(u) + λρu|²  +  (1/A)∫_Γ |σ(u)·n|²  +  ((1/V)∫_Ω ρ|u|² − 1)²  +  Σ_j c_j²,
    c_j = ∫ρ u·φ_j / (‖u‖ ‖φ_j‖),   φ_j = 앞 단계가 저장한 원시 출력

- **고유값 λ는 단계마다 학습되는 스칼라**(시드별)다. λ = exp(η)로 두어 Adam의 걸음이 상대 변화가 되게 했다 — Λ의 크기가
  구조마다 세 자리 이상 다르다(기준 상자 λ₁ ≈ 1.7 × 10⁻³, L-브래킷 λ₆ ≈ 1.29). 단계 1은 초기 망의 훈련 표본 Rayleigh 몫에서,
  단계 k ≥ 2는 단계 k − 1의 최종 λ에서 시작한다(참조를 보지 않는다). 강형식 문헌에서 λ를 학습 변수로 두는 방식이다([3]·[7]).
- **deflation은 사영이 아니라 벌점이다**(순차 강형식 기준선, Jin 외 2022): 저장된 원시 출력과의 정규화 질량 내적을 **훈련
  표본에서** 0으로 민다. 저장하는 것은 원시 출력이고(변분형과 같다), 보고하는 단계 출력도 사영하지 않은 망 그 자체다.
- **자유면 견인력은 벌점**이다(고정면은 하드 BC로 정확). 변분형에서는 자연 경계조건이라 없던 항이다. 경계 표본은 자유면의
  면적에 비례해 균등하게 뽑고 내부 표본과 같은 시점에 바꾼다.
- 가중은 모두 1이다 — 조정하지 않은 강형식 기준선이다.

보고되는 Λ는 감사 격자에서 잰 **망 출력의 Rayleigh 몫**이다. 그래서 채점·하한 게이트·사다리가 변분형과 같은 양을 읽는다.
학습된 λ, 잔차·견인력의 훈련/감사 값, 저장 모드와의 직교 결함(훈련 표본·감사 격자)을 따로 남긴다.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.func import functional_call, jacfwd, vmap

from . import p5_neural as p5n
from ..problems.p5_block import P5_GEOMETRY

ARM_NAME = "p5_pinn_3d[strong form, orthogonality penalty]"
AUDIT_CHUNK = 1500            # 감사 격자 잔차(헤시안)를 점 묶음으로 잰다 — 메모리 상한
BOUNDARY_AUDIT_POINTS = 4000  # 견인력 감사 표본(고정 난수)
BOUNDARY_AUDIT_SEED = 777


# ── 자유 경계 ──────────────────────────────────────────────────────────────────

def free_faces(g, bc: str = "clamped_x0") -> list:
    """자유면 직사각형 목록 [(o, e1, e2, n)] — 점은 o + s·e1 + t·e2 (s, t ∈ [0, 1]), n은 바깥 법선.

    고정면(x = 0, 양끝 고정이면 x = L도)은 하드 BC가 정확히 맞추므로 빼고, 나머지 모든 면을 넣는다. 상자는 다섯 면,
    L-브래킷은 옆면 다섯과 윗·아랫면(상자 셋씩), T-브래킷은 옆면 일곱과 윗·아랫면(상자 넷씩)이다."""
    kind = g.get("kind")
    L, b, h = (float(g[k]) for k in ("L", "b", "h"))
    F = []

    def lat(x0, y0, x1, y1, n):           # 평면 안의 선분 (x0,y0)–(x1,y1)을 두께 방향으로 올린 옆면
        F.append(((x0, y0, 0.0), (x1 - x0, y1 - y0, 0.0), (0.0, 0.0, h), (n[0], n[1], 0.0)))

    def tops(boxes):                       # 평면 안의 직사각형들의 윗·아랫면
        for (x0, y0, x1, y1) in boxes:
            F.append(((x0, y0, h), (x1 - x0, 0.0, 0.0), (0.0, y1 - y0, 0.0), (0.0, 0.0, 1.0)))
            F.append(((x0, y0, 0.0), (x1 - x0, 0.0, 0.0), (0.0, y1 - y0, 0.0), (0.0, 0.0, -1.0)))

    if kind is None:
        if bc == "clamped_x0":
            lat(L, 0.0, L, b, (1.0, 0.0))
        elif bc != "two_face":
            raise ValueError(f"알 수 없는 경계조건: {bc!r}")
        lat(0.0, 0.0, L, 0.0, (0.0, -1.0))
        lat(0.0, b, L, b, (0.0, 1.0))
        tops([(0.0, 0.0, L, b)])
    elif kind == "L":
        a = float(g["a"])
        lat(0.0, 0.0, L, 0.0, (0.0, -1.0))
        lat(L, 0.0, L, b + a, (1.0, 0.0))
        lat(L - b, b + a, L, b + a, (0.0, 1.0))
        lat(L - b, b, L - b, b + a, (-1.0, 0.0))
        lat(0.0, b, L - b, b, (0.0, 1.0))
        tops([(0.0, 0.0, L - b, b), (L - b, 0.0, L, b), (L - b, b, L, b + a)])
    elif kind == "T":
        a = float(g["a"])
        lat(0.0, 0.0, L - b, 0.0, (0.0, -1.0))
        lat(L - b, -a, L - b, 0.0, (-1.0, 0.0))
        lat(L - b, -a, L, -a, (0.0, -1.0))
        lat(L, -a, L, b + a, (1.0, 0.0))
        lat(L - b, b + a, L, b + a, (0.0, 1.0))
        lat(L - b, b, L - b, b + a, (-1.0, 0.0))
        lat(0.0, b, L - b, b, (0.0, 1.0))
        tops([(0.0, 0.0, L - b, b), (L - b, 0.0, L, b), (L - b, -a, L, 0.0), (L - b, b, L, b + a)])
    else:
        raise NotImplementedError(f"강형식은 균질 상자와 L·T-브래킷만 다룬다(재료 계면 조건이 없다): {kind!r}")
    return [tuple(np.asarray(v, float) for v in f) for f in F]


def face_area(f) -> float:
    return float(np.linalg.norm(f[1]) * np.linalg.norm(f[2]))


def boundary_sample(n_points: int, rng, device: str = "cpu", geometry=None, bc: str = "clamped_x0"):
    """자유면 위의 균등 표본 — 면을 면적에 비례해 고르고 면 안에서 균등하게. 반환 (pts, normals, w), w = A/n."""
    torch.set_default_dtype(torch.float64)
    g = P5_GEOMETRY if geometry is None else geometry
    F = free_faces(g, bc)
    area = np.array([face_area(f) for f in F])
    pick = rng.choice(len(F), size=int(n_points), p=area / area.sum())
    st = rng.uniform(size=(int(n_points), 2))
    O = np.stack([F[i][0] for i in pick])
    E1 = np.stack([F[i][1] for i in pick])
    E2 = np.stack([F[i][2] for i in pick])
    N = np.stack([F[i][3] for i in pick])
    P = O + st[:, :1] * E1 + st[:, 1:] * E2
    w = np.full(int(n_points), area.sum() / int(n_points))
    return (torch.as_tensor(P, device=device), torch.as_tensor(N, device=device),
            torch.as_tensor(w, device=device))


# ── 강형식 연산 ────────────────────────────────────────────────────────────────

def _u_vec(p, base, x, scale=None, dist=None, geom=None):
    """한 점에서 u = d(x)·N(x̂) (3,) — `p5_neural._u_comp`의 세 성분을 한 번에."""
    n = functional_call(base, (p, {}), ((x if scale is None else x / scale).reshape(1, 3),)).reshape(3)
    d = x[0] if dist is None else p5n.BC_DISTANCE[dist](x, geom)
    return d * n


def lame(E: float, nu: float):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def div_sigma_of(f, x, nu: float, E: float = 1.0):
    """변위장 f: R³ → R³의 한 점에서 (u, ∇·σ(u)). 등방 균질: (∇·σ)_i = μ∇²u_i + (λ+μ)∂_i(∇·u).
    헤시안은 전진 모드 두 번으로 잰다(입력이 3차원이라 역방향보다 싸다)."""
    H = jacfwd(jacfwd(f))(x)                     # H[i, j, k] = ∂²u_i/∂x_j∂x_k
    lam_l, mu = lame(E, nu)
    lap = torch.einsum("ijj->i", H)
    gdiv = torch.einsum("jij->i", H)
    return f(x), mu * lap + (lam_l + mu) * gdiv


def traction_of(f, x, n, nu: float, E: float = 1.0):
    """변위장 f의 한 점에서 견인력 σ(u)·n (3,)."""
    J = jacfwd(f)(x)
    eps = 0.5 * (J + J.T)
    lam_l, mu = lame(E, nu)
    sig = lam_l * torch.trace(eps) * torch.eye(3, dtype=x.dtype, device=x.device) + 2.0 * mu * eps
    return sig @ n


def div_sigma_point(p, base, x, nu: float, E: float = 1.0, scale=None, dist=None, geom=None):
    """망 한 시드의 한 점에서 (u, ∇·σ(u))."""
    return div_sigma_of(lambda y: _u_vec(p, base, y, scale, dist, geom), x, nu, E)


def traction_point(p, base, x, n, nu: float, E: float = 1.0, scale=None, dist=None, geom=None):
    """망 한 시드의 한 점에서 견인력 σ(u)·n."""
    return traction_of(lambda y: _u_vec(p, base, y, scale, dist, geom), x, n, nu, E)


def _u_batch(ens, pts, scale, dist, geom):
    """전 시드의 원시 변위 (n_seeds, 3, n) — 직교성 벌점이 읽는 저장 모드 값. 기울기를 남기지 않는다."""
    with torch.no_grad():
        return vmap(lambda p: vmap(lambda x: _u_vec(p, ens.base, x, scale, dist, geom))(pts).T)(ens.params)


def _mass(a, b, w, rho):
    """질량 내적 ∫ρ a·b, a·b (3, n)."""
    return (w * rho * (a * b).sum(dim=0)).sum()


# ── 풀이 ──────────────────────────────────────────────────────────────────────

def solve_p5_pinn(n_modes: int = 6, n_seeds: int = 20, iters: int = 20000, seed: int = 0,
                  lr: float = 2e-3, width: int = 128, depth: int = 5, snapshot_every: int = 25,
                  geometry=None, device: str | None = None, audit_grid=None,
                  resample_every: int | None = 250, resample_seed: int = 0, family_axes=None,
                  mc_points: int | None = None, boundary_points: int = 600,
                  check_grids=None, normalize_input: bool = False, bc: str = "clamped_x0",
                  on_mode_done=None, heartbeat_every: int | None = None,
                  return_params: bool = False, n_diag_seeds=None, gram_offdiag: bool = False,
                  per_seed_span: bool = False, ladder_chunk: int | None = None,
                  traction_weight: float = 1.0) -> dict:
    """3D 구조의 최저 n_modes개를 강형식 잔차와 직교성 벌점으로 순차적으로 푼다.

    훈련 표본은 변분형과 같은 두 경로다: `mc_points`를 주면 균등 몬테카를로(`resample_every`마다 새 표본), 아니면 가우스
    격자족(`family_axes`, `resample_every`마다 다시 뽑음). 표본이 바뀔 때 저장 모드의 값과 경계 표본도 함께 바꾼다. 난수는
    내부 표본·경계 표본이 각자의 생성기를 쓴다(내부 표본의 순서는 변분형과 같다).

    반환 사전의 키는 `p5_neural.solve_p5_neural`과 같고(채점이 그대로 읽는다), 강형식 계측을 덧붙인다.

    E28: `traction_weight`는 자유면 견인력 항의 가중이다(나머지 셋은 1). 기본값 1이면 커밋된 손실과 비트 단위로 같다
    (1.0을 곱해도 값이 바뀌지 않는다). 기록하는 항별 손실(`loss_terms`)은 가중 전 값이다."""
    if not resample_every:
        raise ValueError("강형식 arm은 재표집 조건만 다룬다(resample_every)")
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = P5_GEOMETRY if geometry is None else geometry
    if g.get("kind") == "bilayer":
        raise NotImplementedError("강형식은 재료 계면 조건이 없다 — 두 층 구조는 다루지 않는다")
    nu, E, rho = float(g["nu"]), float(g["E"]), float(g["rho"])
    sc = torch.tensor(p5n.input_scale(g), device=dev) if normalize_input else None
    dist = None if bc == "clamped_x0" else bc
    pts_a, w_a = p5n.quadrature(audit_grid if audit_grid is not None else p5n.CERTIFIED_GRID, dev, g)
    V = float(w_a.sum())
    family = None if mc_points else p5n.quadrature_family(dev, g, family_axes)
    rng = np.random.default_rng(resample_seed)              # 내부 표본 — 변분형과 같은 생성기·순서
    rng_b = np.random.default_rng(resample_seed + 10_007)   # 경계 표본 — 따로
    bpa, bna, bwa = boundary_sample(BOUNDARY_AUDIT_POINTS, np.random.default_rng(BOUNDARY_AUDIT_SEED),
                                    dev, g, bc)
    A = float(bwa.sum())
    w_trac = float(traction_weight)
    grids_used = set()

    def _state_raw(ens, points):
        with torch.no_grad():
            return vmap(lambda p: p5n._state_one(p, ens.base, points, sc, dist, g))(ens.params)

    def _rq(S, w):
        return vmap(lambda s_: p5n.rq_from_state(s_, w, nu, E, rho))(S)

    def _res_audit(ens, eta):
        """감사 격자 위 잔차의 평균제곱 (1/V)∫|r|² — 점 묶음으로."""
        tot = torch.zeros(n_seeds, device=dev)
        lam_ = torch.exp(eta)
        with torch.no_grad():
            for P_, W_ in zip(torch.split(pts_a, AUDIT_CHUNK), torch.split(w_a, AUDIT_CHUNK)):
                def one(p, l_):
                    u, ds = vmap(lambda x: div_sigma_point(p, ens.base, x, nu, E, sc, dist, g))(P_)
                    r = ds + l_ * rho * u
                    return (W_ * (r ** 2).sum(dim=-1)).sum()
                tot = tot + vmap(one)(ens.params, lam_)
        return (tot / V).cpu().numpy().copy()

    def _bc_value(ens, bp, bn, bw, area):
        with torch.no_grad():
            def one(p):
                t = vmap(lambda x, n_: traction_point(p, ens.base, x, n_, nu, E, sc, dist, g))(bp, bn)
                return (bw * (t ** 2).sum(dim=-1)).sum() / area
            return vmap(one)(ens.params).cpu().numpy().copy()

    lam, lam_train, lam_param, stage_secs, hists, span_diag = [], [], [], [], [], []
    res_train, res_audit, bc_train, bc_audit, orth_train, orth_audit = [], [], [], [], [], []
    proj_mass, shapes, prevs, ensembles, loss_terms = [], [], [], [], []

    for k in range(n_modes):
        ens = p5n.EnsembleVecMLP3D(n_seeds, width, depth, seed=seed * 1000 + k, device=dev)
        if prevs:
            P = torch.stack(prevs, dim=1)
            span_diag.append(p5n.span_diagnostics_3d(P.detach(), w_a, with_offdiag=gram_offdiag,
                                                     per_seed=per_seed_span, n_diag_seeds=n_diag_seeds))
        else:
            span_diag.append(None)

        def draw():
            if mc_points:
                cp, cw = p5n.mc_quadrature(mc_points, rng, dev, g)
            else:
                cp, cw = family[int(rng.integers(len(family)))]
            grids_used.add(int(len(cw)))
            bp, bn, bw = boundary_sample(boundary_points, rng_b, dev, g, bc)
            stored = tuple(_u_batch(e_, cp, sc, dist, g) for e_ in ensembles)
            return cp, cw, bp, bn, bw, stored

        cur = draw()
        # 단계 1은 초기 망의 훈련 표본 Rayleigh 몫, 단계 k ≥ 2는 앞 단계의 최종 λ에서 시작한다.
        if k == 0:
            init = _rq(_state_raw(ens, cur[0]), cur[1]).detach().clamp(min=1e-12)
        else:
            init = torch.as_tensor(lam_param[-1], device=dev).clamp(min=1e-12)
        eta = torch.log(init).clone().requires_grad_(True)
        opt = torch.optim.Adam(list(ens.params.values()) + [eta], lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 3, 1), gamma=0.4)

        def loss_one(p, e_, cp, cw, bp, bn, bw, *stored):
            u, ds = vmap(lambda x: div_sigma_point(p, ens.base, x, nu, E, sc, dist, g))(cp)
            r = ds + torch.exp(e_) * rho * u
            l_res = (cw * (r ** 2).sum(dim=-1)).sum() / V
            t = vmap(lambda x, n_: traction_point(p, ens.base, x, n_, nu, E, sc, dist, g))(bp, bn)
            l_bc = (bw * (t ** 2).sum(dim=-1)).sum() / A
            uT = u.T
            m = _mass(uT, uT, cw, rho)
            l_norm = (m / V - 1.0) ** 2
            l_orth = torch.zeros((), dtype=u.dtype, device=u.device)
            for phi in stored:
                c = _mass(uT, phi, cw, rho) / torch.sqrt(m * _mass(phi, phi, cw, rho))
                l_orth = l_orth + c ** 2
            return l_res + w_trac * l_bc + l_norm + l_orth, torch.stack([l_res, l_bc, l_norm, l_orth])

        hist, t0 = [], time.perf_counter()
        for it in range(iters + 1):
            if it % resample_every == 0 and 0 < it < iters:
                cur = draw()
            cp, cw, bp, bn, bw, stored = cur
            if it % snapshot_every == 0 or it == iters:
                hist.append(_rq(_state_raw(ens, pts_a), w_a).detach().cpu().numpy().copy())
            if heartbeat_every and it and it % heartbeat_every == 0:
                el = time.perf_counter() - t0
                print(f"[p5 진행] {ARM_NAME} k={k + 1} it={it}/{iters} 경과 {el / 3600:.2f}h "
                      f"ms/iter {el / it * 1000:.1f}", flush=True)
            if it == iters:
                break
            tot, terms = vmap(lambda p, e_, *s_: loss_one(p, e_, cp, cw, bp, bn, bw, *s_))(
                ens.params, eta, *stored)
            opt.zero_grad(set_to_none=True)
            tot.sum().backward()
            opt.step(); sched.step()
        stage_secs.append(time.perf_counter() - t0)
        hists.append(np.asarray(hist))

        with torch.no_grad():
            cp, cw, bp, bn, bw, stored = cur
            tot, terms = vmap(lambda p, e_, *s_: loss_one(p, e_, cp, cw, bp, bn, bw, *s_))(
                ens.params, eta, *stored)
            loss_terms.append(terms.cpu().numpy().copy())          # (n_seeds, 4) 마지막 훈련 표본
            S_raw = _state_raw(ens, pts_a)
            lam.append(_rq(S_raw, w_a).cpu().numpy().copy())
            lam_train.append(_rq(_state_raw(ens, cp), cw).cpu().numpy().copy())
            lam_param.append(torch.exp(eta).detach().cpu().numpy().copy())
            res_train.append(terms[:, 0].cpu().numpy().copy())
            bc_train.append(terms[:, 1].cpu().numpy().copy())
            # 직교 결함 — 훈련 표본(벌점이 본 곳)과 감사 격자에서 저장 모드와의 정규화 내적 최대 |c|
            u_tr = _u_batch(ens, cp, sc, dist, g)
            if stored:
                c_tr = torch.stack([vmap(lambda a, b_: _mass(a, b_, cw, rho) / torch.sqrt(
                    _mass(a, a, cw, rho) * _mass(b_, b_, cw, rho)))(u_tr, phi) for phi in stored], 1)
                orth_train.append(c_tr.abs().amax(dim=1).cpu().numpy().copy())
                Pa = torch.stack(prevs, dim=1)[:, :, :3]
                ua = S_raw[:, :3]
                c_au = torch.stack([vmap(lambda a, b_: _mass(a, b_, w_a, rho) / torch.sqrt(
                    _mass(a, a, w_a, rho) * _mass(b_, b_, w_a, rho)))(ua, Pa[:, j]) for j in range(Pa.shape[1])], 1)
                orth_audit.append(c_au.abs().amax(dim=1).cpu().numpy().copy())
                S_pr = vmap(lambda s_, *q: p5n._project_exact(s_, torch.stack(q, 0), w_a))(
                    S_raw, *tuple(prevs))
                m0 = (w_a * (S_raw[:, :3] ** 2).sum(dim=1)).sum(dim=-1)
                m1 = (w_a * (S_pr[:, :3] ** 2).sum(dim=1)).sum(dim=-1)
                proj_mass.append((m1 / torch.clamp(m0, min=1e-300)).cpu().numpy().copy())
            else:
                orth_train.append(np.zeros(n_seeds)); orth_audit.append(np.zeros(n_seeds))
                proj_mass.append(np.ones(n_seeds))
            shapes.append(S_raw[:, :3].cpu().numpy().copy())     # 보고하는 출력 = 사영하지 않은 망
            prevs.append(S_raw.detach())
        res_audit.append(_res_audit(ens, eta.detach()))
        bc_audit.append(_bc_value(ens, bpa, bna, bwa, A))
        ensembles.append(ens)
        if on_mode_done is not None:
            on_mode_done({"mode": k + 1, "arm": ARM_NAME, "lam": lam[-1], "lam_train": lam_train[-1],
                          "hist": hists[-1], "shapes": shapes[-1], "proj_mass": proj_mass[-1],
                          "seconds": stage_secs[-1], "w": w_a.cpu().numpy(),
                          "span_diagnostics": span_diag[-1], "n_seeds": n_seeds, "iters": iters})

    # ── 사다리: 같은 망의 Rayleigh 몫을 더 높은 차수에서 다시 잰다(사영 없음 — 보고하는 양이 원시 망의 몫이다) ──
    rungs = [] if check_grids is None else [tuple(int(v) for v in gg) for gg in check_grids]
    lam_check = drift = lam_ladder = None
    if rungs:
        base = np.asarray(lam).T
        measured = [base]
        for gg in rungs:
            p2, w2 = p5n.quadrature(gg, dev, g)
            chk = []
            with torch.no_grad():
                for e_ in ensembles:
                    if ladder_chunk is None:
                        S2 = _state_raw(e_, p2)
                    else:
                        S2 = torch.cat([_state_raw(e_, c_) for c_ in torch.split(p2, int(ladder_chunk))], dim=-1)
                    chk.append(_rq(S2, w2).cpu().numpy().copy())
                    del S2
            measured.append(np.asarray(chk).T)
        lam_check = measured[1]
        drift = np.abs(lam_check - base) / np.abs(base)
        lam_ladder = np.stack(measured, axis=-1)
    audit_eff = tuple(int(v) for v in (audit_grid if audit_grid is not None else p5n.CERTIFIED_GRID))
    return {"lam": np.asarray(lam).T, "lam_train": np.asarray(lam_train).T,
            "audit_grid": audit_eff, "n_q_audit": int(len(w_a)), "hist_grid": "audit",
            "lam_check": lam_check, "quadrature_drift": drift, "lam_ladder": lam_ladder,
            "check_grids": None if not rungs else (audit_eff,) + tuple(rungs),
            "check_grid": rungs[0] if rungs else None,
            "params": ([{k_: v.detach().cpu().numpy() for k_, v in e_.params.items()} for e_ in ensembles]
                       if return_params else None),
            "resample_every": resample_every, "n_grids_used": len(grids_used),
            "family_axes": None if family_axes is None else [list(a) for a in family_axes],
            "projected_mass_fraction": np.asarray(proj_mass).T, "shapes": shapes,
            "stored": [p_.cpu().numpy() for p_ in prevs], "w": w_a.cpu().numpy(),
            "hist": hists, "stage_seconds": stage_secs, "span_diagnostics": span_diag,
            "arm": ARM_NAME, "projection": "penalty", "storage": "raw", "mc_points": mc_points,
            "boundary_points": int(boundary_points), "batch_offdiag": None,
            "n_seeds": n_seeds, "iters": iters, "n_stored": 0, "stored_iters": None,
            "lr_milestones": None, "fourier": None, "width": width, "depth": depth, "bc": bc,
            # 강형식 계측 (n_seeds, n_modes)
            "lam_param": np.asarray(lam_param).T,
            "res_train": np.asarray(res_train).T, "res_audit": np.asarray(res_audit).T,
            "bc_train": np.asarray(bc_train).T, "bc_audit": np.asarray(bc_audit).T,
            "orth_train": np.asarray(orth_train).T, "orth_audit": np.asarray(orth_audit).T,
            "loss_terms": np.stack(loss_terms, axis=1),          # (n_seeds, n_modes, 4)
            "loss_term_names": ["residual", "traction", "normalization", "orthogonality"],
            "weights": {"residual": 1.0, "traction": w_trac, "normalization": 1.0, "orthogonality": 1.0},
            "eigenvalue": "trainable lambda = exp(eta) per seed; stage 1 starts at the training-sample "
                          "Rayleigh quotient of the initial network, stage k at the final lambda of stage k-1",
            "disclosure": {**ensembles[-1].disclosure(),
                           "input_scaling": "per-axis [0,1]^3" if normalize_input else "raw physical coordinates",
                           "formulation": "strong form: residual + free-surface traction + normalization + "
                                          "orthogonality penalty, " + ("unit weights" if w_trac == 1.0 else
                                                                       f"traction weight {w_trac:g}, others one")},
            "dtype": str(torch.get_default_dtype()).replace("torch.", "")}
