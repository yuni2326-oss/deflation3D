"""P5 — 3D 벡터 탄성의 순차 Rayleigh 최소화와 세 deflation 규칙.

P4 2D 판본과 **같은 규약**을 따른다. 다른 것은 셋뿐이다.

1. 상태가 12성분이다: 변위 3 + 기울기 9(∂u_i/∂x_j). 사영은 질량 내적으로 계수를
   구하고 그 계수를 **12성분 전부에** 적용한다 — 선형연산이므로 도함수가 따라간다.
2. 하드 BC가 닫힌 식이다: φ = x·N. 거리함수 구성도 ADF도 필요 없고 정확하다.
3. 구적이 텐서 Gauss다. 격자는 §2.4가 인증한 (16, 8, 6)을 **그대로** 쓴다 —
   다른 격자를 쓰면 인증한 양과 계산하는 양이 달라진다.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, stack_module_state, vmap

from ..problems.p5_block import P5_GEOMETRY
from ..quadrature import gauss_legendre

# §2.4가 인증한 격자. 손으로 바꾸지 않는다 — 바꾸면 run_p5의 인증을 다시 돌려야 한다.
CERTIFIED_GRID = (16, 8, 6)


def quadrature(grid=CERTIFIED_GRID, device: str = "cpu", geometry=None):
    """영역 전체의 텐서 Gauss. 반환 (pts(n_q,3), w(n_q,)).

    영역이 볼록 직육면체이므로 메시 기반 구적이 필요 없다 — P4의 블록 구적과 달리
    한 번의 텐서곱으로 끝난다."""
    torch.set_default_dtype(torch.float64)
    g = P5_GEOMETRY if geometry is None else geometry
    axes, wts = [], []
    for n, length in zip(grid, (g["L"], g["b"], g["h"])):
        p, ww = gauss_legendre(int(n), 0.0, float(length))
        axes.append(torch.as_tensor(p, device=device))
        wts.append(torch.as_tensor(ww, device=device))
    X, Y, Z = torch.meshgrid(*axes, indexing="ij")
    W = torch.einsum("i,j,k->ijk", *wts).reshape(-1)
    pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
    return pts, W


# 재표집 구적의 격자 가족. **모두 인증 격자 이상으로 촘촘하다** — 각 반복의 목적함수가
# 그 자체로 정확해야 하고, 정확도를 낮춰 무작위성을 사는 것이 아니다.
# 고정 격자를 이기려면 함수가 이 격자 전부를 동시에 속여야 하는데, Gauss 절점은 차수마다
# 서로 다른 위치에 놓이므로 한 격자의 구멍이 다른 격자의 구멍이 아니다.
FAMILY_NX = (16, 17, 18, 19, 20)
FAMILY_NY = (8, 9, 10)
FAMILY_NZ = (6, 7, 8)


def quadrature_family(device: str = "cpu", geometry=None, axes=None):
    """(pts, w) 목록. 격자 축 조합 전부 — 기본 5×3×3 = 45개.

    `axes`는 (nx들, ny들, nz들). E3(감사 해상도 재표집)이 격자족 자체를 올린다."""
    ax, ay, az = axes if axes is not None else (FAMILY_NX, FAMILY_NY, FAMILY_NZ)
    return [quadrature((nx, ny, nz), device, geometry)
            for nx in ax for ny in ay for nz in az]


def mc_quadrature(n_points: int, rng, device: str = "cpu", geometry=None):
    """균등 몬테카를로 구적 — 점 n_points개, 가중은 모두 V/n (E15).

    Rowan 외(IJNME 2025)가 고차원 예에서 쓴 방식이다. 난수는 넘겨받은 numpy Generator에서
    뽑으므로 재표집 시드가 같으면 비트 단위로 재현된다."""
    torch.set_default_dtype(torch.float64)
    g = P5_GEOMETRY if geometry is None else geometry
    hi = np.array([g["L"], g["b"], g["h"]], dtype=float)
    pts = torch.as_tensor(rng.uniform(size=(int(n_points), 3)) * hi, device=device)
    vol = float(hi.prod())
    w = torch.full((int(n_points),), vol / int(n_points), device=device)
    return pts, w


STORAGE = ("raw", "orthogonalized")


class _VecMLP3D(nn.Module):
    """N: R³ → R³. 시행함수는 u = x·N이므로 여기서 BC를 신경쓰지 않는다."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyz):
        return self.net(xyz)


class EnsembleVecMLP3D:
    """시드별 파라미터를 배치로 쌓아 vmap 한 번에 전 시드를 학습한다(P1·P4와 같다)."""

    def __init__(self, n_seeds: int, width: int = 64, depth: int = 4,
                 seed: int = 0, device: str | None = None):
        torch.set_default_dtype(torch.float64)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        models = [_VecMLP3D(width, depth).to(dev) for _ in range(n_seeds)]
        self.params, _ = stack_module_state(models)
        self.base = _VecMLP3D(width, depth).to(dev).to("meta")
        self.n_seeds, self.width, self.depth, self.device = n_seeds, width, depth, dev

    def disclosure(self) -> dict:
        n = sum(v[0].numel() for v in self.params.values())
        return {"width": self.width, "depth": self.depth, "activation": "tanh",
                "n_params_per_seed": int(n), "n_seeds": self.n_seeds,
                "ansatz": "u = x * N(x,y,z); the clamped face x = 0 is satisfied "
                          "exactly by construction, not by a penalty"}


# 하드 BC의 거리함수. 커밋된 실행은 `clamped_x0`이며 기본값이다(E13-b가 둘째를 더한다).
BC_DISTANCE = {
    "clamped_x0": lambda x, g: x[0],                    # x = 0 한 면
    "two_face": lambda x, g: x[0] * (g["L"] - x[0]),    # x = 0과 x = L 두 면
}


def _u_comp(p, base, pt, j: int, scale=None, dist=None, geom=None):
    """한 점에서 u_j — grad의 스칼라 진입점. D(x) = x가 닫힌 식이다.

    `scale`을 주면 **망이 보는 좌표만** 축별로 나눈다(E8, 검토 §4.3). 곱해지는 x는
    물리 좌표로 남으므로 하드 BC u(0) = 0은 그대로 정확하다 — 정규화 실험이 경계조건
    변경과 뒤섞이면 안 된다."""
    x = pt.reshape(1, 3)
    n = functional_call(base, (p, {}), (x if scale is None else x / scale,)).reshape(-1)
    d = (pt.reshape(3)[0] if dist is None
         else BC_DISTANCE[dist](pt.reshape(3), geom))
    return d * n[j]


def _state_one(p, base, pts, scale=None, dist=None, geom=None):
    """한 시드의 상태 (12, n_q) = (u₀,u₁,u₂, ∂u₀/∂x_j, ∂u₁/∂x_j, ∂u₂/∂x_j).

    정확 질량-직교 사영은 장뿐 아니라 **변형률까지** 같은 계수로 빼야 하므로 열두
    성분을 한 텐서로 들고 다닌다(P4의 여섯 성분과 같은 이유)."""
    comps, grads = [], []
    for j in range(3):
        comps.append(vmap(lambda t, j=j: _u_comp(p, base, t, j, scale, dist, geom))(pts))
        grads.append(vmap(lambda t, j=j:
                          grad(_u_comp, argnums=2)(p, base, t, j, scale,
                                                   dist, geom))(pts))
    return torch.cat([torch.stack(comps, 0),
                      torch.cat([g.T for g in grads], 0)], 0)


def state_batch(ens: EnsembleVecMLP3D, pts, scale=None, bc=None, geometry=None):
    """전 시드의 상태 (n_seeds, 12, n_q)."""
    g = P5_GEOMETRY if geometry is None else geometry
    return vmap(lambda p: _state_one(p, ens.base, pts, scale, bc, g))(ens.params)


def rq_from_state(S, w, nu: float, E: float = 1.0, rho: float = 1.0):
    """상태에서 Λ = ∫ε:C:ε / ∫ρ|u|².

    등방 탄성의 이차형식을 라메 상수로 직접 쓴다:
        ε:C:ε = λ(tr ε)² + 2μ Σᵢ εᵢᵢ² + μ(γ_yz² + γ_xz² + γ_xy²)
    **평면응력으로 줄이지 않는다** — 3차원 블록이므로 그렇게 줄이면 비틀림 강성이 틀린다.
    """
    u = S[:3]
    G = S[3:].reshape(3, 3, -1)                 # G[i, j] = ∂u_i/∂x_j
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    exx, eyy, ezz = G[0, 0], G[1, 1], G[2, 2]
    gyz = G[1, 2] + G[2, 1]
    gxz = G[0, 2] + G[2, 0]
    gxy = G[0, 1] + G[1, 0]
    tr = exx + eyy + ezz
    num = (w * (lam * tr ** 2 + 2.0 * mu * (exx ** 2 + eyy ** 2 + ezz ** 2)
                + mu * (gyz ** 2 + gxz ** 2 + gxy ** 2))).sum()
    den = (w * rho * (u ** 2).sum(dim=0)).sum()
    return num / den


def energy_ip(A, B, w, nu: float, E: float = 1.0):
    """변형률 에너지 이차형식의 **쌍선형 판** a(u,v) = ∫ ε(u):C:ε(v).

    `rq_from_state`의 분자를 두 인자로 편극한 것이다(대각에서 같다). 오라클 적합이
    이것을 필요로 하는 이유: Rayleigh 몫은 변형률이 지배하므로 변위만 맞춘 적합은
    "시행공간이 이 모드를 담을 수 있는가"를 재지 못한다. 본실험에서 MAC 0.9999인
    셀의 e_λ가 0.07까지 벌어지는 것이 바로 그 간극이다."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    GA, GB = A[3:].reshape(3, 3, -1), B[3:].reshape(3, 3, -1)
    trA = GA[0, 0] + GA[1, 1] + GA[2, 2]
    trB = GB[0, 0] + GB[1, 1] + GB[2, 2]
    diag = sum(GA[i, i] * GB[i, i] for i in range(3))
    shear = sum((GA[i, j] + GA[j, i]) * (GB[i, j] + GB[j, i])
                for i, j in ((1, 2), (0, 2), (0, 1)))
    return (w * (lam * trA * trB + 2.0 * mu * diag + mu * shear)).sum()


def _mass_ip(A, B, w):
    """질량 내적 ⟨a,b⟩ = ∫ Σᵢ aᵢbᵢ — 변위 3성분만 쓴다."""
    return (w * (A[:3] * B[:3]).sum(dim=0)).sum()


def _project_exact(S, P, w):
    """arm (b) — 정확 질량-직교 사영. S:(12,n_q), P:(k,12,n_q).

    **jitter는 이 경로에만 있다**(P4와 같은 비대칭). 대각 규칙은 역행렬을 풀지 않고
    QR은 애초에 그람을 만들지 않기 때문이다."""
    G = torch.einsum("aij,j,bij->ab", P[:, :3], w, P[:, :3])
    r = torch.einsum("aij,j,ij->a", P[:, :3], w, S[:3])
    c = torch.linalg.solve(
        G + 1e-14 * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), r)
    return S - torch.einsum("a,anj->nj", c, P)


def _project_diagonal(S, P, w):
    """arm (a′) — 그람의 대각만 쓴다. 저장 모드가 서로 질량직교할 때만 정확 사영과
    같고, 아니면 **사영이 아닌 연산자**다. P5가 시험하는 것이 바로 그 전제다."""
    num = torch.einsum("aij,j,ij->a", P[:, :3], w, S[:3])
    den = torch.einsum("aij,j,aij->a", P[:, :3], w, P[:, :3])
    return S - torch.einsum("a,anj->nj", num / den, P)


def _project_qr(S, P, w, drop_rtol: float | None = None):
    """arm (b′) — 그람을 만들지 않는 가중 MGS. √u 미달 방향은 버린다."""
    eps = torch.finfo(S.dtype).eps
    tiny = torch.finfo(S.dtype).tiny
    rtol = eps ** 0.5 if drop_rtol is None else float(drop_rtol)
    Q = []
    for a in range(P.shape[0]):
        v = P[a]
        n0 = torch.sqrt(torch.clamp(_mass_ip(v, v, w), min=0.0))
        for q in Q:
            v = v - _mass_ip(q, v, w) * q
        n = torch.sqrt(torch.clamp(_mass_ip(v, v, w), min=0.0))
        ok = n > rtol * torch.clamp(n0, min=tiny)
        Q.append(v * torch.where(ok, 1.0 / torch.clamp(n, min=tiny),
                                 torch.zeros_like(n)))
    out = S
    for q in Q:
        out = out - _mass_ip(q, S, w) * q
    return out


def _project_reorth_diagonal(S, P, w):
    """arm (a″) — 저장 집합을 **사용 전에** W-재직교화한 뒤 대각근사를 적용한다(E11).

    명제 1의 예측을 시험하는 arm이다. 재직교화하면 정규화 그람이 항등이 되어 D⁻¹G = I이고,
    대각근사가 정확 사영과 대수적으로 일치한다. 따라서 이 arm이 벽을 없애면 대각 규칙의
    실패는 규칙 자체의 한계가 아니라 **원시 출력을 직교 가정 아래 쓰는 구현의 기저
    의존성**으로 귀속된다.

    **저장된 원시 출력 자체는 바꾸지 않는다** — 사용 시점에만 재직교화한다. 그래야 이 arm이
    저장 규약이 아니라 사용 전 복구를 시험하는 것이 된다(사전규정 §5).

    재직교화에 쓰는 MGS는 `_project_qr`의 것과 같은 절차이며, 다른 점은 그 뒤에 정확
    사영이 아니라 **대각 계수**를 적용한다는 것뿐이다. 정규직교 기저에서는 두 계수가
    같으므로 결과가 일치하고, 그 일치 자체가 이 arm의 설계 근거다."""
    eps, tiny = torch.finfo(S.dtype).eps, torch.finfo(S.dtype).tiny
    rtol = eps ** 0.5
    Q = []
    for a in range(P.shape[0]):
        v = P[a]
        n0 = torch.sqrt(torch.clamp(_mass_ip(v, v, w), min=0.0))
        for q in Q:
            v = v - _mass_ip(q, v, w) * q
        n = torch.sqrt(torch.clamp(_mass_ip(v, v, w), min=0.0))
        ok = n > rtol * torch.clamp(n0, min=tiny)
        Q.append(v * torch.where(ok, 1.0 / torch.clamp(n, min=tiny),
                                 torch.zeros_like(n)))
    Qs = torch.stack(Q, 0)
    # 여기서부터가 대각 규칙이다 — 분모에 자기 노름을 쓴다(정규직교이므로 1에 가깝다).
    num = torch.einsum("aij,j,ij->a", Qs[:, :3], w, S[:3])
    den = torch.einsum("aij,j,aij->a", Qs[:, :3], w, Qs[:, :3])
    den = torch.where(den > tiny, den, torch.ones_like(den))
    return S - torch.einsum("a,anj->nj", num / den, Qs)


PROJECTIONS = {"exact": _project_exact, "diagonal": _project_diagonal,
               "qr": _project_qr, "reorth_diagonal": _project_reorth_diagonal}

ARM_NAME = {"exact": "p5_neural_3d[exact projection]",
            "diagonal": "p5_neural_3d[diagonal projection]",
            "qr": "p5_neural_3d[qr projection]",
            "reorth_diagonal": "p5_neural_3d[reorth diagonal projection]"}


def project(S, P, w, kind: str = "exact"):
    """사영 전략 하나. 이름 규약은 P1·P4와 같다(exact = (b), diagonal = (a′), qr = (b′))."""
    try:
        return PROJECTIONS[kind](S, P, w)
    except KeyError:
        raise ValueError(f"알 수 없는 사영: {kind!r} — {sorted(PROJECTIONS)}") from None


def span_diagnostics_3d(P, w, probe=None, n_diag_seeds=None,
                        with_offdiag: bool = False, per_seed: bool = False):
    """P5 저장 스팬의 진단 — P1·P4와 **같은 지표**를 세 성분 질량 내적에서 잰다.

    ⟨a,b⟩ = ∫Σᵢaᵢbᵢ는 세 성분을 이어붙이고 가중을 세 번 쓴 1차원 내적과 같으므로
    `neural.diagnostics.span_diagnostics`를 그대로 재사용한다. 지표 정의가 문제마다
    갈리면 "기전이 3D로 전이되는가"를 물을 수 없다.

    시드 요약은 **전 시드가 기본**이다(P1·P4와 같은 관례)."""
    from . import diagnostics as dgn

    w3 = torch.cat([w, w, w], dim=-1)

    def _flat(A):
        return torch.cat([A[..., 0, :], A[..., 1, :], A[..., 2, :]], dim=-1)

    def one(Pk, pr=None):
        flat = _flat(Pk)
        if pr is None:
            return dgn.span_diagnostics(flat, flat, w3, with_offdiag=with_offdiag)
        f = _flat(pr)
        return dgn.span_diagnostics(flat, flat, w3, phi=f, d2=f,
                                    with_offdiag=with_offdiag)

    if P.dim() == 3:
        return one(P, None if probe is None else probe)
    ns = (int(P.shape[0]) if n_diag_seeds is None
          else min(int(n_diag_seeds), int(P.shape[0])))
    per = [one(P[i], None if probe is None else probe[i]) for i in range(ns)]
    out = {"n_seeds_summarized": ns}
    if per_seed:
        # 요약을 만드는 이 코드가 이미 시드별 값을 계산하고 버리고 있었다. 전이 주장을
        # 시드 수준에서 세우려면 그것이 필요하다(검토 2026-09-26 §3).
        out["per_seed"] = [{"seed": i, **d} for i, d in enumerate(per)]
    for key in per[0]:
        vals = [d[key] for d in per]
        if isinstance(vals[0], bool):
            out[key + "_frac"] = float(sum(vals)) / len(vals)
        elif isinstance(vals[0], str):
            out[key] = vals[0]
        elif isinstance(vals[0], int):
            out[key] = int(np.median(vals))
        else:
            out[key] = float(np.median(vals))
    return out


def solve_p5_neural(n_modes: int = 6, n_seeds: int = 20, iters: int = 20000,
                    grid=CERTIFIED_GRID, seed: int = 0, lr: float = 2e-3,
                    width: int = 64, depth: int = 4, snapshot_every: int = 25,
                    geometry=None, projection: str = "exact",
                    device: str | None = None, n_diag_seeds=None,
                    check_grid=None, resample_every: int | None = None,
                    resample_seed: int = 0, audit_grid=None,
                    family_axes=None, on_mode_done=None,
                    heartbeat_every: int | None = None, check_grids=None,
                    gram_offdiag: bool = False, per_seed_span: bool = False,
                    return_params: bool = False,
                    normalize_input: bool = False,
                    bc: str = "clamped_x0", mc_points: int | None = None,
                    storage: str = "raw", batch_diag: bool = False) -> dict:
    """3D 블록의 최저 n_modes개 Λ를 순차 Rayleigh 최소화로 푼다.

    `projection`은 P1·P4와 같은 세 규칙이다. P5가 묻는 것은 raw 저장 스팬이 질량-직교
    불변량을 3D에서도 자연히 위반하는지이고, 위반한다면 (a′)의 mode-order wall이
    같은 방식으로 나타나고 (b)·(b′)가 제거하는지다.

    `audit_grid`(v2 §2.2)를 주면 **보고되는 모든 양**(Λ, 형상, 이력, 진단, 사영질량)을
    거기서 잰다. 훈련 손실만 `grid`/재표집 격자에서 계산한다. v1의 실패는 학습·게이트·
    채점이 같은 격자를 본 것이었다 — 학습되는 기저는 자신을 재는 격자에 맞춰 변형한다.
    `None`이면 v1과 같은 단일 격자 동작이다.

    `on_mode_done`·`heartbeat_every`는 **곁가지**다(2026-09-12 정전 대응). 둘 다
    기본 꺼짐이라 커밋된 실행은 불변이고, 켜도 수치 경로에 닿지 않는다 — 콜백은
    이미 호스트로 복사된 배열만 받고, 초기화는 스테이지마다 전역 RNG를 다시 심으며
    (`torch.manual_seed`), 재표집은 전용 Generator를 쓴다. 콜백이 던지는 예외는
    막지 않는다 — 저장의 실패를 삼켜 조용히 없는 기록을 만드는 쪽이 더 나쁘므로,
    삼킬지는 콜백 자신이 정한다(`run_p5_neural.make_mode_writer`가 삼킨다).

    **E15 — 저장 규약과 몬테카를로 구적.** `storage="orthogonalized"`는 Rowan 외 식 (8)의
    규약이다: 수렴한 스테이지 k의 함수를 û_k = R_k − Σ_{j<k} c_kj û_j 로 저장하며 계수는
    **감사 격자에서** 고정한다(c_kj = ⟨R_k, û_j⟩/⟨û_j, û_j⟩, 이미 직교인 û에 대한 고전 GS).
    그러면 저장 집합이 감사 격자에서 정확히 직교하고, 감사 격자의 대각형 뺄셈은 정확 사영과
    같다. `mc_points`는 훈련 구적을 균등 몬테카를로로 바꾼다(`resample_every`마다 새 표본,
    deflation과 Rayleigh 몫이 같은 표본을 쓴다). `batch_diag`는 훈련 표본 위 저장 집합의
    정규화 그람 최대 비대각을 스냅샷마다 재어 마지막 10 %의 평균을 남기는 계측이다.
    세 인자 모두 기본값이면 커밋된 경로와 같다."""
    if storage not in STORAGE:
        raise ValueError(f"storage는 {STORAGE} 중 하나여야 합니다: {storage!r}")
    if mc_points and not resample_every:
        raise ValueError("mc_points는 resample_every와 함께 써야 한다(표본을 언제 바꿀지)")
    arm_label = (ARM_NAME[projection] if storage == "raw"
                 else ARM_NAME[projection][:-1] + ", orthogonalized storage]")
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = P5_GEOMETRY if geometry is None else geometry
    nu, E, rho = float(g["nu"]), float(g["E"]), float(g["rho"])
    # E8(검토 §4.3): 망이 보는 좌표를 축별로 [0,1]³에 맞춘다. 원좌표는 축 스케일이
    # 최대 25배 다르다(z ∈ (0, 0.04)). **솔버 변인이지 계기가 아니다** — 켜면 시행공간이
    # 달라지므로 커밋된 실행과 비트 비교를 기대하지 않는다.
    sc = (torch.tensor([g["L"], g["b"], g["h"]], device=device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))
          if normalize_input else None)
    # E13-b: 하드 BC의 거리함수. 기본값은 커밋된 한 면 고정이다.
    dist = None if bc == "clamped_x0" else bc
    pts, w = quadrature(grid, dev, g)
    # 감사 격자를 주지 않으면 훈련 격자가 곧 감사 격자다(v1 동작).
    pts_a, w_a = (quadrature(audit_grid, dev, g) if audit_grid is not None
                  else (pts, w))

    lam, lam_train, prevs, stage_secs, hists, span_diag = [], [], [], [], [], []
    ensembles = []   # 재검·재표집에 학습된 파라미터가 필요하다
    family = (quadrature_family(dev, g, family_axes)
              if resample_every and not mc_points else None)
    coefs = []       # 직교화 저장의 스테이지별 계수 (n_seeds, k) — raw 저장이면 비어 있다
    batch_off = []   # 스테이지별 시드 평균 비대각(훈련 표본) — batch_diag일 때만
    rng = np.random.default_rng(resample_seed)
    grids_used = {int(len(w))}
    # **사영 후 남은 질량 비율.** 붕괴한 시드의 Λ는 λ₁보다도 작게 나오는데, 그것이
    # "낮은 모드를 찾았다"인지 "사영된 함수가 소실됐다"인지는 귀속이 다르다.
    # 표에 이 비율이 없으면 두 경우를 구별할 수 없다.
    proj_mass = []
    shapes = []          # 스테이지가 실제로 내놓은 함수(사영 후 변위 3성분)

    def _stack_at(points, upto):
        """앞선 upto개 모드의 **원시** 상태를 주어진 점에서 다시 계산한다.

        저장된 값을 다른 격자에 그대로 쓰면 사영이 그 격자에서 직교를 만들지 못해
        deflation이 조용히 무력해진다 — 훈련·감사 어느 쪽으로 옮기든 마찬가지다."""
        with torch.no_grad():
            raw = tuple(
                vmap(lambda p, e=e_: _state_one(p, e.base, points, sc, dist, g))(e_.params).detach()
                for e_ in ensembles[:upto])
            return raw if storage == "raw" else _orthogonalized(raw)

    def _orthogonalized(raw):
        """û_i = R_i − Σ_{j<i} c_ij û_j — 계수는 감사 격자에서 고정한 값(시드별)."""
        U = []
        for i, R in enumerate(raw):
            u = R
            for j in range(i):
                u = u - coefs[i][:, j, None, None] * U[j]
            U.append(u)
        return tuple(U)

    for k in range(n_modes):
        ens = EnsembleVecMLP3D(n_seeds, width, depth, seed=seed * 1000 + k,
                               device=dev)
        # `stack_module_state`가 돌려주는 텐서를 **그대로** 최적화한다. clone()을 끼우면
        # non-leaf가 되어 Adam이 거부한다(P4와 같은 처리).
        opt = torch.optim.Adam(list(ens.params.values()), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(iters // 3, 1), gamma=0.4)
        proj = PROJECTIONS[projection]

        def loss_audit(p, *prev):
            """감사 격자에서 잰 Rayleigh 몫 — 이력·최종 Λ·채점이 모두 이 값을 쓴다."""
            S = _state_one(p, ens.base, pts_a, sc, dist, g)
            if prev:
                S = proj(S, torch.stack(prev, 0), w_a)
            return rq_from_state(S, w_a, nu, E, rho)

        # **진단은 사영 전에, 그 스테이지가 실제로 넘겨받는 원시 후보에서 잰다.**
        # 보고되는 양이므로 감사 격자에서 잰다(v2 §2.2).
        if prevs:
            P = torch.stack(prevs, dim=1)                   # (n_seeds, k, 12, n_q_audit)
            span_diag.append(span_diagnostics_3d(P.detach(), w_a, with_offdiag=gram_offdiag,
                                                 per_seed=per_seed_span,
                                                 n_diag_seeds=n_diag_seeds))
        else:
            span_diag.append(None)

        ensembles.append(ens)
        audit_stack = tuple(prevs)     # 감사 격자에서의 저장 스팬 — 스테이지 내내 불변
        hist = []
        stage_off = []
        cur_pts, cur_w = pts, w
        cur_stack = _stack_at(cur_pts, k)
        t0 = time.perf_counter()
        for it in range(iters + 1):
            if resample_every and it % resample_every == 0 and it < iters:
                # **격자를 갈면 저장 스팬도 그 격자에서 다시 평가해야 한다.**
                # 옛 격자의 값을 그대로 쓰면 사영이 현재 격자에서 직교를 만들지 못해
                # deflation이 조용히 무력해진다.
                if mc_points:
                    cur_pts, cur_w = mc_quadrature(mc_points, rng, dev, g)
                else:
                    cur_pts, cur_w = family[int(rng.integers(len(family)))]
                grids_used.add(int(len(cur_w)))
                cur_stack = _stack_at(cur_pts, k)

            def loss_cur(p, *prev):
                S = _state_one(p, ens.base, cur_pts, sc, dist, g)
                if prev:
                    S = proj(S, torch.stack(prev, 0), cur_w)
                return rq_from_state(S, cur_w, nu, E, rho)

            r = vmap(loss_cur)(ens.params, *cur_stack)
            if it % snapshot_every == 0 or it == iters:
                # 이력은 **감사 격자**에서 남긴다 — 재표집 때문에 훈련 J는 격자가
                # 바뀔 때마다 점프하므로 게이트로 쓸 수 없다(v2 §3.1).
                with torch.no_grad():
                    hist.append(vmap(loss_audit)(ens.params, *audit_stack)
                                .detach().cpu().numpy().copy())
                    if batch_diag and k >= 2:
                        Pk = torch.stack(cur_stack, dim=1)[:, :, :3]      # (s, k, 3, n)
                        Gb = torch.einsum("sain,n,sbin->sab", Pk, cur_w, Pk)
                        d_ = torch.sqrt(torch.diagonal(Gb, dim1=1, dim2=2))
                        Cb = Gb / (d_[:, :, None] * d_[:, None, :])
                        off = (Cb - torch.diag_embed(torch.diagonal(Cb, dim1=1, dim2=2))).abs()
                        stage_off.append((it, off.amax(dim=(1, 2)).cpu().numpy().copy()))
            if heartbeat_every and it and it % heartbeat_every == 0:
                el = time.perf_counter() - t0
                print(f"[p5 진행] {arm_label} k={k + 1} "
                      f"it={it}/{iters} 경과 {el / 3600:.2f}h "
                      f"ms/iter {el / it * 1000:.1f}", flush=True)
            if it == iters:
                break
            opt.zero_grad(set_to_none=True)
            r.sum().backward()
            opt.step(); sched.step()
        stage_secs.append(time.perf_counter() - t0)
        hists.append(np.asarray(hist))
        if batch_diag:
            tail = [o for it_, o in stage_off if it_ >= int(0.9 * iters)]
            batch_off.append(np.mean(tail, axis=0) if tail else np.full(n_seeds, np.nan))

        with torch.no_grad():
            lam.append(vmap(loss_audit)(ens.params, *audit_stack).cpu().numpy().copy())
            # 훈련 격자에서 잰 값도 남긴다 — 두 격자가 실제로 갈라졌는지 확인 가능해야
            # 분리가 일어났다는 주장이 검증된다(v2 §4.1).
            def _loss_train(p, *prev):
                S = _state_one(p, ens.base, cur_pts, sc, dist, g)
                if prev:
                    S = proj(S, torch.stack(prev, 0), cur_w)
                return rq_from_state(S, cur_w, nu, E, rho)
            lam_train.append(vmap(_loss_train)(ens.params, *cur_stack)
                             .cpu().numpy().copy())

            S_raw = vmap(lambda p: _state_one(p, ens.base, pts_a, sc, dist, g))(ens.params)
            m0 = (w_a * (S_raw[:, :3] ** 2).sum(dim=1)).sum(dim=-1)
            if audit_stack:
                S_pr = vmap(lambda s_, *q: proj(s_, torch.stack(q, 0), w_a))(
                    S_raw, *audit_stack)
                m1 = (w_a * (S_pr[:, :3] ** 2).sum(dim=1)).sum(dim=-1)
                proj_mass.append((m1 / torch.clamp(m0, min=1e-300))
                                 .cpu().numpy().copy())
            else:
                S_pr = S_raw
                proj_mass.append(np.ones(n_seeds))
            shapes.append(S_pr[:, :3].cpu().numpy().copy())
            # **저장하는 것은 원시 출력이다.** 사영된 것을 저장하면 다음 스테이지의
            # 스팬이 이미 직교화돼 P5가 묻는 전제조건 자체가 사라진다.
            # 예외: E15의 직교화 저장(Rowan 규약)은 그 규약 자체를 시험한다.
            if storage == "raw":
                prevs.append(S_raw.detach())
            else:
                ck = torch.zeros(n_seeds, k, device=S_raw.device, dtype=S_raw.dtype)
                u = S_raw.detach()
                for j, Uj in enumerate(prevs):
                    num = torch.einsum("sin,n,sin->s", S_raw[:, :3], w_a, Uj[:, :3])
                    den = torch.einsum("sin,n,sin->s", Uj[:, :3], w_a, Uj[:, :3])
                    ck[:, j] = num / den
                    u = u - ck[:, j, None, None] * Uj
                coefs.append(ck)
                prevs.append(u.detach())

        if on_mode_done is not None:
            # 이 모드가 끝난 직후의 상태만 넘긴다. §3.6 구적분리는 모든 모드가 끝난
            # 뒤에 한 번 도는 후처리라 여기서는 아직 없다 — 곁가지 기록에도 없다.
            on_mode_done({"mode": k + 1, "arm": arm_label,
                          "lam": lam[-1], "lam_train": lam_train[-1],
                          "hist": hists[-1], "shapes": shapes[-1],
                          "proj_mass": proj_mass[-1], "seconds": stage_secs[-1],
                          "w": w_a.cpu().numpy(),   # 채점 가중 = 감사 격자
                          "span_diagnostics": span_diag[-1],
                          "n_seeds": n_seeds, "iters": iters})

    # ── §3.6 구적분리: 같은 신경망을 두 번째 구적차수에서 다시 잰다 ──────────
    # 사전 인증은 '참조 모드'를 덮을 뿐 '최적화가 도달하는 함수'를 덮지 못한다.
    # 값이 움직이면 그 셀의 Λ는 함수의 Rayleigh 몫이 아니라 구적 인공물이다.
    # `check_grids`(복수)는 **사다리**다 — 감사 격자를 0번 칸에 두고 그 위 차수들을
    # 이어 재서 셀마다 수렴 연구를 남긴다. 한 격자 차이로는 수렴을 볼 수 없다는 것이
    # E4의 교훈이다(발산하는 셀에서도 이웃 두 값이 우연히 가까울 수 있다).
    # 계측만 한다 — 분류는 건드리지 않으므로 커밋된 채점과 비교가 유지된다.
    lam_check = drift = lam_ladder = None
    if check_grids is not None:
        rungs = [tuple(int(v) for v in gg) for gg in check_grids]
    elif check_grid is not None:
        rungs = [tuple(int(v) for v in check_grid)]
    else:
        rungs = []
    audit_eff = tuple(int(v) for v in
                      (audit_grid if audit_grid is not None else grid))
    if rungs:
        # 기준은 **감사 격자의 Λ**다. 훈련 격자와 비교하면 재표집 자체가 드리프트로 잡힌다.
        base = np.asarray(lam).T
        measured = [base]
        for gg in rungs:
            pts2, w2 = quadrature(gg, dev, g)
            with torch.no_grad():
                prev2, chk = [], []
                raw2 = [vmap(lambda p, e=e: _state_one(p, e.base, pts2, sc, dist, g))(e.params)
                        for e in ensembles]
                stored2 = raw2 if storage == "raw" else list(_orthogonalized(tuple(raw2)))
                for i2, S2 in enumerate(raw2):
                    if prev2:
                        Sp2 = vmap(lambda s_, *q: proj(s_, torch.stack(q, 0), w2))(
                            S2, *tuple(prev2))
                    else:
                        Sp2 = S2
                    chk.append(vmap(lambda s_: rq_from_state(s_, w2, nu, E, rho))(Sp2)
                               .cpu().numpy().copy())
                    prev2.append(stored2[i2].detach())
            measured.append(np.asarray(chk).T)
        lam_check = measured[1]
        drift = np.abs(lam_check - base) / np.abs(base)
        lam_ladder = np.stack(measured, axis=-1)

    return {"lam": np.asarray(lam).T,                       # (n_seeds, n_modes) 감사격자
            "lam_train": np.asarray(lam_train).T,
            "audit_grid": None if audit_grid is None
            else tuple(int(v) for v in audit_grid),
            "n_q_audit": int(len(w_a)), "hist_grid": "audit",
            "lam_check": lam_check, "quadrature_drift": drift,
            # 망 파라미터. E7이 480셀 재측정에 77 GPU-시간을 쓴 것은 이것을 남기지
            # 않았기 때문이다 — 재현이 비트 단위임은 E4·E6·E7이 보였으므로, 남겨 두면
            # 같은 일이 순전파 비용이 된다. 기본 꺼짐(용량이 arm당 약 64 MB).
            "params": ([{k: v.detach().cpu().numpy() for k, v in e.params.items()}
                        for e in ensembles] if return_params else None),
            "lam_ladder": lam_ladder,          # (n_seeds, n_modes, 1 + 사다리 칸)
            "check_grids": None if not rungs else (audit_eff,) + tuple(rungs),
            "resample_every": resample_every,
            "family_axes": None if family_axes is None
            else [list(a) for a in family_axes],
            "n_grids_used": len(grids_used),
            # 드리프트 계기는 사다리의 첫 칸이다 — 단수 인자만 주던 실행과 같은 값이다.
            "check_grid": rungs[0] if rungs else None,
            "projected_mass_fraction": np.asarray(proj_mass).T,
            "shapes": shapes,                               # 모드별 (n_seeds, 3, n_q)
            "stored": [p.cpu().numpy() for p in prevs],     # 원시 저장 출력
            "w": w_a.cpu().numpy(),     # 형상·MAC이 쓰는 가중 = 감사 격자
            "w_train": w.cpu().numpy(),
            "hist": hists, "stage_seconds": stage_secs,
            "span_diagnostics": span_diag,
            "arm": arm_label, "projection": projection,
            "storage": storage, "mc_points": mc_points,
            "batch_offdiag": (np.stack(batch_off, axis=1) if batch_diag else None),
            "n_seeds": n_seeds, "iters": iters, "grid": tuple(int(v) for v in grid),
            "n_q": int(len(w)), "width": width, "depth": depth,
            "bc": bc,
            "disclosure": {**ens.disclosure(),
                           "input_scaling": ("per-axis [0,1]^3" if normalize_input
                                             else "raw physical coordinates")},
            "dtype": str(torch.get_default_dtype()).replace("torch.", "")}
