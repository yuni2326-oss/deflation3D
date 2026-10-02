"""저장 스팬의 진단 — deflation이 실제로 넘겨받는 것을 한 곳에서 잰다.

**왜 별도 모듈인가.** 지금까지 이 값들은 전용 진단 드라이버가 solve를 **다시 돌려서**
쟀다(`run_p1_neural.run_orthogonality_diagnostic`). 그러면 진단이 본실행과 다른 실행의
성질이 되고, 실행조건(dtype·예산·시드)이 바뀔 때마다 재실행이 필요하다. 여기서는 순차
solver가 모드를 저장할 때마다 **그 실행 자신의** 스팬을 재고, 이후 단계의 안정화 기준이
그 로그에서 공짜로 나온다.

지표는 다섯 갈래다.
  · κ₂(G)          — arm (b)가 실제로 푸는 그람의 조건수. `kappa_gram_times_eps`가
                     그 dtype에서의 예측 붕괴량이다(≳1이면 그람 경로에 유효자리가 없다).
  · ‖I − C‖₂       — 질량정규화 그람의 항등 이탈 = ‖Π_G − A_D‖_W (대각규칙과의 연산자 거리).
  · ‖I − QᵀWQ‖₂    — MGS가 만든 기저의 직교성 결손. arm (b′)이 실제로 의존하는 양.
  · 다중임계 rank   — 1e-10·1e-12·1e-16. 단일 정수로 보고하면 임계가 숨는다.
  · 두 사영의 일치도 — (b)와 (b′)이 같은 사영이라는 명제를 **실행마다** 재는 열.

전 시드에 대해 eigvalsh를 도는 것은 비싸므로 호출자가 시드 부분집합을 넘긴다 —
요약 방식은 manifest에 적는다.
"""
from __future__ import annotations

import torch

from .deflation import project_exact, w_orthonormalize

RANK_TOLS = (1e-10, 1e-12, 1e-16)

# 대칭 고유해석기·인수분해가 없는 정밀도. 이 경우 **측정만** fp64 사본에서 한다 —
# arm이 실제로 하는 계산(MGS, 그람 풀이)은 작업 정밀도 그대로 둔다.
NO_LINALG = (torch.float16, torch.bfloat16)


def _spectrum(A):
    return torch.linalg.eigvalsh(0.5 * (A + A.transpose(-1, -2)))


def _max_offdiag(C) -> float:
    """정규화 그람 C의 비대각 최대 절댓값. 대각은 1이므로 제외한다."""
    k = C.shape[-1]
    if k < 2:
        return 0.0
    eye = torch.eye(k, dtype=torch.bool, device=C.device)
    return float(C.masked_fill(eye, 0.0).abs().amax())


def span_diagnostics(Phi, Phi2, wq, phi=None, d2=None,
                     drop_rtol: float | None = None,
                     with_offdiag: bool = False) -> dict:
    """저장 스팬 Φ(k, n_q) 하나의 진단. `phi`를 주면 두 사영의 일치도까지 잰다.

    Φ는 **원시 망 출력**이어야 한다 — 사영된 모드형을 넣으면 그람이 구성상 대각이 되어
    두 arm이 같아 보인다(실제로 그렇게 잘못 잰 적이 있다).

    `with_offdiag`는 정규화 그람의 **쌍별** 비대각 |C_ij|를 `absc_ij`로 덧붙인다
    (2026-09-20 검토). 기전 진술이 "성분 공유"로 승격되면서 어느 쌍이 큰가가 검정
    대상이 됐는데 `max_offdiag`로는 볼 수 없다. **절댓값인 이유**는 저장 모드의 부호
    규약이 임의라 부호 있는 값의 시드 중앙값이 의미를 잃기 때문이다. P1·P4와 공용
    함수이므로 기본은 꺼짐이며, 켜지 않으면 반환 키가 바뀌지 않는다."""
    k = int(Phi.shape[0])
    mdt = torch.float64 if Phi.dtype in NO_LINALG else Phi.dtype
    Pm, wm = Phi.to(mdt), wq.to(mdt)
    eps0 = float(torch.finfo(Phi.dtype).eps)
    # **스팬이 발산하면 그것이 결과다.** 예외로 실행을 죽이면 몇 모드까지 갔는지가
    # 데이터에서 사라진다 — NaN으로 채우고 깃발을 세운 뒤 계속한다.
    if not bool(torch.isfinite(Pm).all()):
        out = {"k": k, "span_finite": False, "dtype": str(Phi.dtype).replace("torch.", ""),
               "measure_dtype": str(mdt).replace("torch.", ""), "eps_machine": eps0,
               "gram_solve_finite": False}
        for key in ("kappa_gram", "kappa_gram_times_eps", "dev_identity",
                    "max_offdiag", "lam_min_normalized", "qr_orth_defect",
                    "proj_agreement_rel"):
            out[key] = float("nan")
        out["n_dropped"] = k
        for t in RANK_TOLS:
            out[f"rank_{t:.0e}".replace("e-", "e")] = 0
        return out
    G = torch.einsum("ki,i,li->kl", Pm, wm, Pm)
    dg = torch.sqrt(torch.clamp(torch.diagonal(G), min=1e-300))
    C = G / (dg.unsqueeze(-1) * dg.unsqueeze(-2))
    try:
        ev_g = _spectrum(G)
        ev_c = _spectrum(C)
    except Exception:                      # 고유해석기 자체가 수렴 실패
        out = {"k": k, "span_finite": False,
               "dtype": str(Phi.dtype).replace("torch.", ""),
               "measure_dtype": str(mdt).replace("torch.", ""), "eps_machine": eps0,
               "gram_solve_finite": False, "n_dropped": k}
        for key in ("kappa_gram", "kappa_gram_times_eps", "dev_identity",
                    "max_offdiag", "lam_min_normalized", "qr_orth_defect",
                    "proj_agreement_rel"):
            out[key] = float("nan")
        for t in RANK_TOLS:
            out[f"rank_{t:.0e}".replace("e-", "e")] = 0
        return out
    eps = eps0
    kappa = float(ev_g.amax() / torch.clamp(ev_g.amin(), min=1e-300))

    # 직교화는 **작업 정밀도로** 한다(arm이 실제로 그렇게 계산하므로). 그 결과의
    # 직교성 결손만 측정 정밀도로 잰다.
    Q, _, kept = w_orthonormalize(Phi, Phi2, wq, drop_rtol)
    keep = kept.bool()
    n_keep = int(keep.sum())
    if n_keep:
        Qm = Q[keep].to(mdt)
        Gq = torch.einsum("ki,i,li->kl", Qm, wm, Qm)
        eye = torch.eye(n_keep, dtype=Gq.dtype, device=Gq.device)
        defect = float(torch.linalg.matrix_norm(Gq - eye, ord=2))
    else:
        defect = float("nan")

    out = {
        "k": k,
        "kappa_gram": kappa,
        "kappa_gram_times_eps": kappa * eps,
        "dev_identity": float((ev_c - 1.0).abs().amax()),
        # **비대각 최대값은 스펙트럼 지표와 다른 양이다.** k = 2에서는 둘이 같지만
        # 그보다 깊으면 갈린다. P1의 표와 P5의 갈래 판정 문턱이 이 양을 지목하므로
        # 대체하지 않고 그대로 잰다.
        "max_offdiag": _max_offdiag(C),
        **({f"absc_{i + 1}{j + 1}": float(C[i, j].abs())
            for i in range(C.shape[-1]) for j in range(i + 1, C.shape[-1])}
           if with_offdiag else {}),
        "lam_min_normalized": float(ev_c.amin()),
        "qr_orth_defect": defect,
        "n_dropped": k - n_keep,
        "dtype": str(Phi.dtype).replace("torch.", ""),
        "measure_dtype": str(mdt).replace("torch.", ""),
        "span_finite": True,
        "eps_machine": eps,
    }
    top = torch.clamp(ev_g.amax(), min=1e-300)
    for t in RANK_TOLS:
        out[f"rank_{t:.0e}".replace("e-", "e")] = int((ev_g > t * top).sum())

    # 그람 풀이가 이 정밀도에서 유한한 답을 내는가 — fp32 붕괴의 직접 증거
    try:
        probe = phi if phi is not None else Phi[-1]
        c = project_exact(probe, Phi, wq)
        out["gram_solve_finite"] = bool(torch.isfinite(c).all())
    except Exception:                       # 특이행렬은 예외로 오는 경우도 있다
        out["gram_solve_finite"] = False

    if phi is not None and d2 is not None:
        # 그람 경로가 그 정밀도에 없으면 두 경로를 비교할 수 없다 — 그 사실을 NaN으로
        # 남긴다(0으로 채우면 "일치했다"로 읽힌다).
        try:
            c = project_exact(phi, Phi, wq)
            pe = (phi - c @ Phi).to(mdt)
        except Exception:
            out["proj_agreement_rel"] = float("nan")
            return out
        a = torch.einsum("ki,i,i->k", Q, wq, phi)
        pq = (phi - a @ Q).to(mdt)
        phim = phi.to(mdt)
        nrm = torch.sqrt(torch.clamp((wm * pq * pq).sum(), min=1e-300))
        diff = torch.sqrt(torch.clamp((wm * (pe - pq) ** 2).sum(), min=0.0))
        out["proj_agreement_rel"] = float(diff / nrm)
    return out
