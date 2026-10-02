"""§3.5 지표 + §7 사전등록 모드분류.

주파수 일치만으로는 모드 동일성이 증명되지 않는다(Babuška–Osborn) — 그래서 MAC·잔차·
직교성·주각을 함께 보고하고, 축퇴쌍은 고유공간으로 비교한다.

**분류 임계는 사전등록값이다. 데이터를 본 뒤 바꾸지 않는다:**
  MAC ≥ 0.9, e_λ ≤ 0.05, 재발수렴 판정은 호출자가 `converged`로 전달.
"""
from __future__ import annotations

import math

import numpy as np

MAC_MIN = 0.9
ELAM_MAX = 0.05
OUTCOMES = ("correct", "lower_mode_basin", "spurious", "non_converged")


def rel_eig_error(lam_h: float, lam_ref: float) -> float:
    if lam_ref == 0.0:
        return float("inf")
    return float(abs(lam_h - lam_ref) / abs(lam_ref))


def rel_errors_padded(lam, lam_ref, n_modes: int) -> list[float]:
    """모드별 상대 고유값오차를 길이 n_modes로 맞춘다(부족분은 NaN).

    기저 자유도가 요청 모드수보다 적으면(예: 4항 기저로 10모드) 그 모드는 표현
    자체가 불가능하다. 0이나 마지막 값으로 채우면 표가 거짓말을 하므로 NaN으로 남긴다."""
    lam = np.asarray(lam, float).ravel()
    lam_ref = np.asarray(lam_ref, float).ravel()
    out = []
    for k in range(n_modes):
        if k < len(lam) and k < len(lam_ref) and np.isfinite(lam[k]) and lam_ref[k] != 0:
            out.append(float(abs(lam[k] - lam_ref[k]) / abs(lam_ref[k])))
        else:
            out.append(float("nan"))
    return out


def mac(phi_h, phi_ref, M) -> float:
    """MAC = |φ_hᵀMφ_ref|² / ((φ_hᵀMφ_h)(φ_refᵀMφ_ref)). 스케일·부호 불변, 0..1."""
    a = np.asarray(phi_h, float).ravel()
    b = np.asarray(phi_ref, float).ravel()
    M = np.asarray(M, float)
    num = float(a @ M @ b) ** 2
    den = float(a @ M @ a) * float(b @ M @ b)
    if den <= 0.0 or not np.isfinite(num) or not np.isfinite(den):
        return 0.0
    return float(min(max(num / den, 0.0), 1.0))


def normalized_residual(K, M, lam: float, phi) -> float:
    """r_h = ‖Kφ − λMφ‖ / (‖Kφ‖ + λ‖Mφ‖)."""
    K = np.asarray(K, float); M = np.asarray(M, float)
    x = np.asarray(phi, float).ravel()
    kx = K @ x; mx = M @ x
    den = np.linalg.norm(kx) + abs(lam) * np.linalg.norm(mx)
    if den == 0.0:
        return float("inf")
    return float(np.linalg.norm(kx - lam * mx) / den)


def _m_orthonormal(Phi, M):
    """열들을 M-정규직교화(얇은 QR 동등). 주각·부분공간 MAC의 전처리."""
    Phi = np.asarray(Phi, float)
    M = np.asarray(M, float)
    G = Phi.T @ M @ Phi
    G = 0.5 * (G + G.T)
    ev, V = np.linalg.eigh(G)
    keep = ev > max(ev.max(), 1e-300) * 1e-14
    return Phi @ (V[:, keep] / np.sqrt(ev[keep]))


def orthogonality_matrix(Phi, M) -> np.ndarray:
    """O_ij = ⟨φ_i,φ_j⟩_M / (‖φ_i‖_M‖φ_j‖_M) — 비대각항이 직교성 위반량."""
    Phi = np.asarray(Phi, float); M = np.asarray(M, float)
    G = Phi.T @ M @ Phi
    d = np.sqrt(np.abs(np.diag(G)))
    d[d == 0.0] = 1.0
    return G / np.outer(d, d)


def projection_coeffs(phi, Phi, M) -> np.ndarray:
    """하위모드 사영계수 c = (ΦᵀMΦ)⁻¹ΦᵀMφ — deflation 실패의 직접 증거."""
    Phi = np.asarray(Phi, float); M = np.asarray(M, float)
    x = np.asarray(phi, float).ravel()
    G = Phi.T @ M @ Phi
    return np.linalg.solve(0.5 * (G + G.T), Phi.T @ M @ x)


def principal_angles(Phi1, Phi2, M) -> np.ndarray:
    """두 부분공간의 주각[rad] (M-내적). 축퇴 고유공간 비교의 정본 지표."""
    Q1 = _m_orthonormal(Phi1, M)
    Q2 = _m_orthonormal(Phi2, M)
    s = np.linalg.svd(Q1.T @ np.asarray(M, float) @ Q2, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_mac(Phi1, Phi2, M) -> float:
    """부분공간 MAC = (1/k)Σ cos²θ_i ∈ [0,1]. 1이면 두 고유공간이 일치."""
    ang = principal_angles(Phi1, Phi2, M)
    if len(ang) == 0:
        return 0.0
    return float(np.mean(np.cos(ang) ** 2))


def classify(phi_h, lam_h, ref_modes, ref_lams, target: int, converged: bool,
             M=None, mac_min: float = MAC_MIN, elam_max: float = ELAM_MAX) -> dict:
    """§7 사전등록 4분기 판정. target·matched_mode는 1-based.

    ref_modes: (n_dof, n_ref) 기준 모드형 열벡터들. M=None이면 단위 질량행렬."""
    ref_modes = np.asarray(ref_modes, float)
    ref_lams = np.asarray(ref_lams, float)
    x = np.asarray(phi_h, float).ravel()
    Mm = np.eye(ref_modes.shape[0]) if M is None else np.asarray(M, float)

    usable = np.all(np.isfinite(x)) and np.isfinite(lam_h)
    if not usable:
        return {"outcome": "non_converged", "matched_mode": None,
                "mac": float("nan"), "e_lam": float("nan"), "certified": False}

    # **측정은 항상 한다.** outcome만 사전등록 규칙을 따르고, mac·e_lam은 진단으로 남긴다 —
    # 그래야 "해가 틀렸다"와 "수렴을 인증하지 못했다"를 구분할 수 있다. 확률적 구적(MC)이나
    # 성격이 다른 목적함수(부분공간 trace)에서는 후자가 지배적인데, 값을 NaN으로 지우면
    # 그 사실이 데이터에서 사라진다.
    macs = np.array([mac(x, ref_modes[:, j], Mm) for j in range(ref_modes.shape[1])])
    j = int(np.argmax(macs))
    best = float(macs[j])
    e_lam = rel_eig_error(float(lam_h), float(ref_lams[target - 1]))
    matched = j + 1

    if not converged:
        return {"outcome": "non_converged", "matched_mode": matched,
                "mac": best, "e_lam": e_lam, "certified": False}

    if best < mac_min:
        outcome = "spurious"
    elif matched == target:
        outcome = "correct" if e_lam <= elam_max else "spurious"
    elif matched < target:
        outcome = "lower_mode_basin"
    else:
        outcome = "spurious"           # 목표보다 높은 차수로 도망 = 오식별
    return {"outcome": outcome, "matched_mode": matched, "mac": best,
            "e_lam": e_lam, "certified": True}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """이항비율 k/n의 Wilson 점수구간(기본 95 %). 0·n 성공에서도 유계."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max((centre - half) / den, 0.0), min((centre + half) / den, 1.0))


def confusion_counts(records) -> dict:
    """분류 레코드 목록 → 결과별 개수 + n."""
    out = {k: 0 for k in OUTCOMES}
    for r in records:
        o = r["outcome"]
        if o not in out:
            raise ValueError(f"알 수 없는 outcome: {o}")
        out[o] += 1
    out["n"] = len(records)
    return out
