"""마크다운 표 원시조립 — 계산은 하지 않는다.

논문의 모든 표는 CSV에서 나오고, 이 모듈은 그 값을 **표시**만 한다. 결측은 추정하지 않고
`—`로 남긴다: 빈 칸이 있으면 독자가 각주를 읽지만, 채워진 칸은 아무도 의심하지 않는다.
"""
from __future__ import annotations

import math

MISSING = "—"


def fmt(x, sig: int = 3, none: str = MISSING) -> str:
    """표 한 칸의 문자열. 결측은 채우지 않는다."""
    if x is None:
        return none
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)          # 모드 번호·카운트는 정수로 — "1.00"은 표를 흐린다
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return none
        try:
            x = float(s)
        except ValueError:
            return s
    x = float(x)
    if math.isnan(x):
        return none
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x == 0.0:
        return "0"
    a = abs(x)
    if 1e-3 <= a < 1e5:
        digits = max(sig - 1 - int(math.floor(math.log10(a))), 0)
        return f"{x:.{digits}f}"
    return f"{x:.{sig - 1}e}"


def table(headers, rows, align: str | None = None) -> str:
    """GFM 파이프 표. align은 열별 'l'/'c'/'r'."""
    n = len(headers)
    cells = []
    for r in rows:
        if len(r) != n:
            raise ValueError(f"{n}열 표에 {len(r)}칸 행: {r!r}")
        # **칸 안의 파이프를 이스케이프한다.** 그러지 않으면 그 칸이 쪼개져 열이 어긋난다 —
        # 부록 B의 (g) 행이 `|rFFT|` 때문에 두 칸 밀렸고, GFM 파서와 docx 변환 모두에서
        # 표가 무너졌다. 열 수 검사는 셀 리스트 기준이라 이 종류를 잡지 못한다.
        cells.append([fmt(v).replace("|", "\\|") for v in r])
    headers = [str(h).replace("|", "\\|") for h in headers]
    align = align or "l" * n
    if len(align) != n:
        raise ValueError(f"align 길이 {len(align)} != 열 수 {n}")
    widths = [max([len(headers[j])] + [len(c[j]) for c in cells]) for j in range(n)]
    sep = {"l": lambda w: ":" + "-" * max(w - 1, 3),
           "c": lambda w: ":" + "-" * max(w - 2, 1) + ":",
           "r": lambda w: "-" * max(w - 1, 3) + ":"}

    def row(vs):
        return "| " + " | ".join(v.ljust(widths[j]) for j, v in enumerate(vs)) + " |"

    out = [row(headers),
           "| " + " | ".join(sep[align[j]](widths[j]) for j in range(n)) + " |"]
    out += [row(c) for c in cells]
    return "\n".join(out) + "\n"


def caption(n: int, text: str, source, sha: str) -> str:
    """표 캡션. 출처 CSV와 git sha를 남겨 어느 실행에서 나왔는지 추적 가능하게 한다."""
    src = source if isinstance(source, str) else ", ".join(source)
    return f"**Table {n}.** {text} (source: `{src}`, git {sha[:8]})"
