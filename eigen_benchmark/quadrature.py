"""조각별 Gauss–Legendre 구적 — 기저차수 p와 구적차수 n_q를 독립으로 두기 위한 기반.

논문 §3.6 구적분리 프로토콜: 구적오차가 기저오차로 위장하지 못하게 해야 한다.
따라서 (1) 물성계면·스프링 위치 x_c를 구간경계로 삼는 조각적분, (2) n_q를 p와 무관하게
지정, (3) 차수 배증 수렴시험이 전부 이 모듈 위에서 이뤄진다.
"""
from __future__ import annotations

import numpy as np


def gauss_legendre(n: int, a: float = 0.0, b: float = 1.0):
    """구간 [a,b]의 n점 Gauss–Legendre 노드·가중치. 차수 2n−1 이하 다항을 정확적분."""
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다")
    t, wt = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (b - a)
    return half * t + 0.5 * (a + b), half * wt


def piecewise_gauss(breaks, n_per_segment: int):
    """분할점 목록으로 나뉜 각 구간에 n_per_segment점 Gauss를 깔아 이어붙인다.

    breaks=[0, x_c, 1]이면 x_c 양쪽을 따로 적분 → 기울기 불연속·물성계면을 정확히 처리.
    반환 (x, w): 오름차순 노드와 대응 가중치(전체 합 = 구간 총길이)."""
    b = np.asarray(breaks, dtype=float)
    if b.ndim != 1 or len(b) < 2:
        raise ValueError("breaks는 길이 2 이상의 1D 배열이어야 합니다")
    if np.any(np.diff(b) <= 0):
        raise ValueError("breaks는 강한 증가여야 합니다")
    xs, ws = [], []
    for lo, hi in zip(b[:-1], b[1:]):
        x, w = gauss_legendre(n_per_segment, float(lo), float(hi))
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)
