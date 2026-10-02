"""deflation 전략 — **이 모듈이 논문의 귀속 도구다.**

같은 구적·같은 초기화에서 모드3 성공률이 사영 100 % / 페널티 0 %로 갈린다(스파이크).
따라서 "고차모드 실패"는 spectral bias가 아니라 직교성 강제 기전의 문제다.

다섯 전략:
  NoDeflation          최저모드용
  Penalty(w)           v3 §3.3의 arm (a). 모드간 Rayleigh 갭(λ⁴: 12→486→3807)을
                       가중이 못 이기면 최저모드로 되돌아간다.
  ProjectionExact      arm (b). φ̃ = φ − Φ(ΦᵀWΦ)⁻¹ΦᵀWφ. v3 [2]의 충실한 대응물.
  ProjectionDiagonal   arm (a′). 파일럿 구현의 대각근사 c_k = ⟨φ,φ_k⟩/⟨φ_k,φ_k⟩ —
                       하위모드가 서로 직교할 때만 정확하고, 아니면 누출이 남는다.
  ProjectionQR         arm (b′). (b)와 **같은 사영**을 그람 없이 계산한다 —
                       W-내적 MGS로 Q를 얻어 φ̃ = φ − Q(QᵀWφ). 정확산술에서는
                       G⁻¹ = R⁻ᵀR⁻¹이라 (b)와 같은 답이고, 부동소수점에서는
                       κ₂(Q) ≈ 1 대 κ₂(G) = κ₂(Φ)²로 갈린다.

**φ″도 함께 사영해야 한다.** 분자가 하위모드 곡률을 품으면 RQ가 무의미해진다.
"""
from __future__ import annotations

import torch


def project_exact(phi, Phi, wq):
    """c = (ΦᵀWΦ)⁻¹ΦᵀWφ — 완전한 그람 역행렬."""
    G = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    return torch.linalg.solve(G, torch.einsum("ki,i,i->k", Phi, wq, phi))


def project_diagonal(phi, Phi, wq):
    """c_k = ⟨φ,φ_k⟩/⟨φ_k,φ_k⟩ — 그람의 대각만 쓰는 파일럿 근사."""
    num = torch.einsum("ki,i,i->k", Phi, wq, phi)
    den = torch.einsum("ki,i,ki->k", Phi, wq, Phi)
    return num / den


def project_exact_cholesky(phi, Phi, wq):
    """`project_exact`와 같은 계수를 **인수분해 커널 없이** 구한다.

    PyTorch는 float16·bfloat16에 lu/cholesky/eigh 커널을 제공하지 않는다(CPU·CUDA 모두).
    그래서 그람 경로는 그 정밀도에서 라이브러리로는 아예 실행되지 않는다 — 그것은
    **방법의 성질이 아니라 라이브러리의 성질**이므로, 같은 알고리즘을 곱셈·나눗셈만으로
    직접 써서 그 구분을 분리한다.

    k는 저장 모드 수로 작고(≤ 20) 블록화가 무의미하므로, 여기 쓰는 비블록 Cholesky는
    그 크기에서 라이브러리가 쓰는 것과 사실상 같은 알고리즘이다. vmap 안에서 돌아야
    하므로 값에 따른 분기를 쓰지 않는다.
    """
    G = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    r = torch.einsum("ki,i,i->k", Phi, wq, phi)
    k = G.shape[-1]
    tiny = torch.finfo(G.dtype).tiny
    # G = L Lᵀ (비블록 Cholesky)
    cols = []
    for j in range(k):
        v = G[:, j]
        for m in range(j):
            v = v - cols[m][j] * cols[m]
        d = torch.sqrt(torch.clamp(v[j], min=tiny))
        cols.append(v / d)
    L = torch.stack(cols, dim=1)          # 열 j가 L의 j번째 열
    # 전진대입 Ly = r
    y = []
    for i in range(k):
        acc = r[i]
        for m in range(i):
            acc = acc - L[i, m] * y[m]
        y.append(acc / torch.clamp(L[i, i], min=tiny))
    # 후진대입 Lᵀc = y
    c = [None] * k
    for i in range(k - 1, -1, -1):
        acc = y[i]
        for m in range(i + 1, k):
            acc = acc - L[m, i] * c[m]
        c[i] = acc / torch.clamp(L[i, i], min=tiny)
    return torch.stack(c)


class NoDeflation:
    name = "none"

    def apply(self, phi, d2, Phi, Phi2, wq):
        return phi, d2, phi.new_zeros(())


class Penalty:
    """arm (a) — 함수는 그대로 두고 겹침의 제곱을 손실에 더한다."""

    def __init__(self, weight: float):
        self.weight = float(weight)
        self.name = f"penalty(w={weight:g})"

    def apply(self, phi, d2, Phi, Phi2, wq):
        if Phi is None:
            return phi, d2, phi.new_zeros(())
        ov = torch.einsum("ki,i,i->k", Phi, wq, phi) ** 2
        nrm = (wq * phi ** 2).sum() * torch.einsum("ki,i,ki->k", Phi, wq, Phi)
        return phi, d2, self.weight * (ov / nrm).sum()


class _Projection:
    def __init__(self, coeff_fn, name):
        self._coeff, self.name = coeff_fn, name

    def apply(self, phi, d2, Phi, Phi2, wq):
        if Phi is None:
            return phi, d2, phi.new_zeros(())
        c = self._coeff(phi, Phi, wq)
        return phi - c @ Phi, d2 - c @ Phi2, phi.new_zeros(())


def ProjectionExact(solver: str = "library"):
    """`solver="cholesky"`면 인수분해 커널이 없는 정밀도에서도 돈다(같은 계수)."""
    if solver == "library":
        return _Projection(project_exact, "projection_exact")
    if solver == "cholesky":
        return _Projection(project_exact_cholesky, "projection_exact[cholesky]")
    raise ValueError(f"알 수 없는 solver: {solver!r}")


def ProjectionDiagonal():
    return _Projection(project_diagonal, "projection_diagonal(pilot)")


def w_orthonormalize(Phi, Phi2, wq, drop_rtol: float | None = None):
    """W-내적 modified Gram–Schmidt. Φ와 Φ″에 **같은 연산**을 적용한다.

    반환 `(Q, Q2, kept)`. Q는 ⟨q_i,q_j⟩_W = δ_ij를 만족하고 Q2는 그 **같은 선형결합**을
    저장 곡률에 적용한 것이다 — 사영계수를 공간미분에 상수로 취급한다는 (b)의 규약을
    그대로 지킨다.

    `kept[i] = 0`이면 그 방향이 수치적으로 종속이라 버려진 것이고 `Q[i] = 0`이 되어
    사영에서 자동으로 빠진다. 이것이 rank-aware의 실체다: 종속 방향을 **역행렬로
    증폭하는 대신 버린다.** 임계는 dtype에서 유도한다(기본 √eps) — 직교화 뒤 남은 노름이
    원래 노름의 그 배 아래면 남은 것은 반올림 잡음이다.

    **vmap 안에서 돌아야 하므로 값에 따른 분기를 쓰지 않는다** — 마스크로 처리한다.
    """
    eps = torch.finfo(Phi.dtype).eps
    tiny = torch.finfo(Phi.dtype).tiny
    rtol = eps ** 0.5 if drop_rtol is None else float(drop_rtol)
    Q, Q2, kept = [], [], []
    for i in range(Phi.shape[0]):
        v, v2 = Phi[i], Phi2[i]
        n0 = torch.sqrt(torch.clamp((wq * v * v).sum(), min=0.0))
        for q, q2 in zip(Q, Q2):
            r = (wq * q * v).sum()
            v = v - r * q
            v2 = v2 - r * q2
        n = torch.sqrt(torch.clamp((wq * v * v).sum(), min=0.0))
        ok = n > rtol * torch.clamp(n0, min=tiny)
        inv = torch.where(ok, 1.0 / torch.clamp(n, min=tiny), torch.zeros_like(n))
        Q.append(v * inv)
        Q2.append(v2 * inv)
        kept.append(ok.to(Phi.dtype))
    return torch.stack(Q), torch.stack(Q2), torch.stack(kept)


class _QRProjection:
    """arm (b′) — 그람을 만들지 않는 사영. φ̃ = φ − Q(QᵀWφ)."""

    name = "projection_qr"

    def __init__(self, drop_rtol: float | None = None):
        self.drop_rtol = drop_rtol

    def apply(self, phi, d2, Phi, Phi2, wq):
        if Phi is None:
            return phi, d2, phi.new_zeros(())
        Q, Q2, _ = w_orthonormalize(Phi, Phi2, wq, self.drop_rtol)
        a = torch.einsum("ki,i,i->k", Q, wq, phi)
        return phi - a @ Q, d2 - a @ Q2, phi.new_zeros(())


def ProjectionQR(drop_rtol: float | None = None):
    return _QRProjection(drop_rtol)
