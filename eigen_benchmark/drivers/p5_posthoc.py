"""P5 원고 r4의 사후 판독 — 커밋된 레코드만 읽어 `p5_posthoc.json`에 쓴다. 판정이 아니다.

r3 종합 검토(2026-10-29)가 요청한 재집계다. 새 학습 계산은 없다.

1. **감사 격자와 사다리 최상단의 거리** d_audit = |Λ_1 − Λ_n|/|Λ_n|. 사다리 꼬리 폭 s(원고 식 7)는 마지막 세 칸만 보므로
   보고되는 감사 Λ(첫 칸)를 검사하지 않는다 — (1.20, 1.00, 1.00, 1.00)은 s = 0이다. 모든 사다리에서 안정 셀의 d_audit
   분포를 내고, 고유값을 최상단 값으로 바꿔도 선택·재풀이 울림·순서 검사·채점이 같은지 본다.
2. **순서 검사의 분모.** 뒤 단계 i > j가 사다리를 통과하고(s ≤ 10⁻³) 감사 Λ가 (1 − t)Λ_j 아래면 j를 울린다. 검사할 수 있는
   정확 셀(뒤에 안정한 단계가 있는 셀), 비종단 더 높은 모드, 비종단 그 밖 부정확, 마지막 단계의 부정확을 따로 센다.
3. **정렬 판독.** 한 사다리의 출력을 감사 Λ 순으로 다시 세워 순위로 채점한다(추가 학습 없음). 첫 통과·원 설정·프로토콜
   사슬을 견주고, 정렬 뒤에 남는 실패를 부류(마지막 단계의 건너뜀, 혼합, 고유값 오차)로 나눈다.
4. **Table S16의 재정의.** "저장 모드가 맞다"를 δ² 문턱 대신 참조 형상 식별(앞 단계 j < k가 모두 모드 j와 MAC ≥ 0.9)로
   정하고 δ²를 다시 요약한다(`p5_exploit_span_error.csv`).
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from . import p5_stats as st

DATA_DIR = st.DATA_DIR
OUT = "p5_posthoc.json"
ORDER_T = 0.05

# 사다리 — (태그, arm, 조건). 기준 구조 원 설정(v2)은 사다리를 남기지 않았으므로 같은 설정으로 다시 잰 E7(재현 비트 단위)을 쓴다.
ORIGINAL = {"(i)": ("e7_ladder_a", "exact", "resampled"), "(ii)": ("e13a_thick", "exact", "resampled"),
            "(iii)": ("e13b_clamped2", "exact", "resampled"), "(iv)": ("e16sq_exact", None, None),
            "(v)": ("e18_exact", None, None), "(vi)": ("e21lb_exact", None, None), "(vii)": ("e22bi_exact", None, None),
            "(viii)": ("e25tb_exact", None, None)}
ORIGINAL_V2 = ("v2", "exact", "resampled")
FIRST = {"(i)": "e15b", "(ii)": "e26_thick_mc", "(iii)": "e26_cc_mc", "(iv)": "e27_sq_mc", "(v)": "e26_near_mc",
         "(vi)": "e21lb_exact_mc1", "(vii)": "e26_bi_mc", "(viii) raw": "e26_tb_raw_mc", "(viii) normalized": "e25tb_protocol"}
S16_RUNS = {"(i)": "e15b", "(ii)": "e26_thick_mc", "(iii)": "e26_cc_mc", "(v)": "e26_near_mc", "(vi)": "e21lb_exact_mc1",
            "(vii)": "e26_bi_mc", "(viii)": "e26_tb_raw_mc"}


def _load(d, spec):
    t, arm, cond = (tuple(spec) + (None, None))[:3] if not isinstance(spec, str) else (spec, None, None)
    return st.load(d, t, arm=st.arm_key(arm) if arm else None, condition=cond)


def by_seed(recs) -> dict:
    out = {}
    for r in recs:
        out.setdefault(int(r["seed"]), {})[int(r["mode"])] = r
    return out


def d_audit(rec) -> float:
    lad = rec["lam_ladder"]
    return abs(lad[0] - lad[-1]) / abs(lad[-1])


def _val(rec, key: str) -> float:
    return rec["Lam"] if key == "audit" else rec["lam_ladder"][-1]


# ---------------------------------------------------------------- 1. 감사–최상단 거리

def audit_distance(d: str = DATA_DIR) -> dict:
    """사다리가 있는 모든 기록(부분 파일 제외)에서 안정 셀의 d_audit 분포. 강형식(pinn)은 따로 센다. 채점을 최상단 값으로
    바꿨을 때 정확이 뒤집히는 안정 셀 수도 낸다."""
    groups = {"variational": [], "strong_form": []}
    for name in sorted(os.listdir(d)):
        if not (name.startswith("p5_neural_") and name.endswith("_records.jsonl")) or name.endswith("_partial_records.jsonl"):
            continue
        tag = name[len("p5_neural_"):-len("_records.jsonl")]
        if not tag:                       # 태그 없는 v1 탐색 실행(사다리 없음)
            continue
        for r in st.load(d, tag):
            if r.get("lam_ladder"):
                groups["strong_form" if "pinn" in tag else "variational"].append(r)
    out = {}
    for g, recs in groups.items():
        stab = [r for r in recs if st._stable(r)]
        da = np.array([d_audit(r) for r in stab])
        acc = np.array([d_audit(r) for r in stab if st.accurate(r)])
        flips = 0
        for r in stab:
            if r["outcome"] == "invalid_quadrature":
                continue
            e_top = abs(r["lam_ladder"][-1] - r["Lam_ref"]) / r["Lam_ref"]
            flips += st.accurate(r) != (r["matched_mode"] == r["mode"] and r["mac"] >= 0.9 and e_top <= 0.05)
        out[g] = {"cells": len(recs), "stable": len(stab), **_dist(da, "stable"), **_dist(acc, "accurate_stable"),
                  "stable_above": {f"{v:g}": int((da > v).sum()) for v in (1e-3, 1e-2, 0.05)},
                  "accuracy_flips_at_top": int(flips)}
    return out


def _dist(x: np.ndarray, name: str) -> dict:
    if not len(x):
        return {f"{name}_n": 0}
    return {f"{name}_n": int(len(x)), f"{name}_median": float(np.median(x)), f"{name}_p99": float(np.percentile(x, 99)),
            f"{name}_max": float(x.max()), f"{name}_above_1e-3": int((x > 1e-3).sum())}


def ladder_group_distance(d: str = DATA_DIR) -> dict:
    """원 설정·첫 통과·끝까지 돈 프로토콜의 후보에서 안정하고 정확한 셀의 d_audit."""
    from .run_p5_neural import e28_candidates
    groups = {"original": [r for s in ORIGINAL.values() for r in _load(d, s)],
              "first_pass": [r for t in FIRST.values() for r in _load(d, t)],
              "end_to_end_candidates": [r for k in range(1, 7) for t in e28_candidates(k).values()
                                        for r in _load(d, t) if int(r["mode"]) == k]}
    out = {}
    for g, recs in groups.items():
        stab = [r for r in recs if st._stable(r)]
        out[g] = {"cells": len(recs), "stable": len(stab),
                  **_dist(np.array([d_audit(r) for r in stab]), "stable"),
                  **_dist(np.array([d_audit(r) for r in stab if st.accurate(r)]), "accurate_stable")}
    return out


def _stable_first(cands: dict, s_: int, key: str) -> str:
    order = list(cands)
    return min(order, key=lambda p_: (not st._stable(cands[p_][s_]), _val(cands[p_][s_], key), order.index(p_)))


def selection_equivalence(d: str = DATA_DIR) -> dict:
    """참조를 읽지 않는 선택과 울림을 감사 Λ 대신 최상단 Λ로 했을 때 달라지는 시드 수(달라진 시드의 정확 여부와 함께)."""
    from .run_p5_neural import e28_candidates, e28_near_candidates
    out = {}
    sel = json.load(open(os.path.join(d, st.E28_SELECTION), encoding="utf-8"))
    seln = json.load(open(os.path.join(d, st.E28_NEAR_SELECTION), encoding="utf-8"))
    for name, cand_fn, selection, stages in (("end_to_end", e28_candidates, sel, range(1, 7)),
                                             ("near_degenerate", e28_near_candidates, seln, (5, 6))):
        res = {}
        for k in stages:
            cands = {p_: {int(r["seed"]): r for r in _load(d, t) if r["mode"] == k} for p_, t in cand_fn(k).items()}
            diff, flags = [], {"audit": 0, "top": 0}
            for s_ in range(20):
                pa, pt = _stable_first(cands, s_, "audit"), _stable_first(cands, s_, "top")
                if pa != selection["stages"][str(k)]["choice"][str(s_)]:
                    raise ValueError(f"{name} 단계 {k} 시드 {s_}: 다시 고른 후보가 기록된 선택과 다르다")
                if pa != pt:
                    diff.append({"seed": s_, "audit": pa, "top": pt, "accurate_audit": bool(st.accurate(cands[pa][s_])),
                                 "accurate_top": bool(st.accurate(cands[pt][s_]))})
                for key, p_ in (("audit", pa), ("top", pt)):
                    f0 = _val(cands["first"][s_], key)
                    flags[key] += (f0 - _val(cands[p_][s_], key)) / f0 > st.E26_RESOLVE_TOL
            res[k] = {"picks_differ": diff, "flags": flags}
        out[name] = res
    picks = {"lbracket_raw": (("e21lb_exact_mc1", "e21lb_s6_raw_x4", "e21lb_s6_raw_ff"), 6),
             "lbracket_normalized": (("e21lb_exact_mc1_norm", "e21lb_s6_norm_x4", "e21lb_s6_norm_x4lo", "e21lb_s6_norm_ff"), 6),
             "lbracket_restarts": (tuple(f"e28_lb_s6_reinit{r}" for r in range(1, 5)), 6),
             **{f"cc_restarts_stage{k}": ((f"e28_cc_s{k}_first",) + tuple(f"e28_cc_s{k}_reinit{r}" for r in range(1, 5)), k)
                for k in (2, 3)}}
    for name, (tags, k) in picks.items():
        by = {t: {int(r["seed"]): r for r in _load(d, t) if r["mode"] == k} for t in tags}
        seeds = sorted(next(iter(by.values())))
        pa = {s_: min(by, key=lambda t: by[t][s_]["Lam"]) for s_ in seeds}
        pt = {s_: min(by, key=lambda t: by[t][s_]["lam_ladder"][-1]) for s_ in seeds}
        out[name] = {"picks_differ": [{"seed": s_, "audit": pa[s_], "top": pt[s_],
                                       "accurate_audit": bool(st.accurate(by[pa[s_]][s_])),
                                       "accurate_top": bool(st.accurate(by[pt[s_]][s_]))} for s_ in seeds if pa[s_] != pt[s_]]}
    resolves = {"e26_cc_mc_s2_ff": ("e26_cc_mc", 2), "e26_cc_mc_s3_ff": ("e26_cc_mc", 3), "e26_near_mc_s5_ff": ("e26_near_mc", 5),
                "e21lb_s6_raw_x4": ("e21lb_exact_mc1", 6), "e21lb_s6_raw_ff": ("e21lb_exact_mc1", 6),
                "e21lb_s6_norm_x4": ("e21lb_exact_mc1_norm", 6), "e21lb_s6_norm_x4lo": ("e21lb_exact_mc1_norm", 6),
                "e21lb_s6_norm_ff": ("e21lb_exact_mc1_norm", 6), "e21lb_s6_norm_w256": ("e21lb_exact_mc1_norm", 6),
                "e15b_s6_raw_ff": ("e15b", 6), **{f"e28_tb_s{k}_ff": ("e26_tb_raw_mc", k) for k in range(1, 7)}}
    rs = {}
    for t, (base, k) in resolves.items():
        f = {int(r["seed"]): r for r in _load(d, base) if r["mode"] == k}
        g = {int(r["seed"]): r for r in _load(d, t) if r["mode"] == k}
        fl = {key: sorted(s_ for s_ in f if (_val(f[s_], key) - _val(g[s_], key)) / _val(f[s_], key) > st.E26_RESOLVE_TOL)
              for key in ("audit", "top")}
        rs[t] = {"flagged_audit": len(fl["audit"]), "flagged_top": len(fl["top"]), "same": fl["audit"] == fl["top"]}
    out["resolve_flags"] = rs
    return out


# ---------------------------------------------------------------- 2–3. 순서 검사와 정렬 판독

def order_check(seeds: dict, t: float = ORDER_T, key: str = "audit", lamref: dict | None = None,
                override: dict | None = None) -> dict:
    """seeds: {시드: {단계: 기록}}. 뒤 단계가 사다리를 통과할 때만 울린다. 채점(정확·더 높은 모드·낮은 모드)은 기록으로만 교차한다.
    `override`는 {(시드, 단계): (짝 모드, MAC)} — 기록 밖에서 다시 채점한 단계(근축퇴 단면 단계 5의 Fourier 재풀이)."""
    c = {"cells": 0, "accurate": 0, "accurate_testable": 0, "accurate_flagged": 0, "nonterminal_higher": 0,
         "nonterminal_higher_flagged": 0, "nonterminal_lower": 0, "nonterminal_lower_flagged": 0,
         "nonterminal_other": 0, "nonterminal_other_flagged": 0, "terminal_inaccurate": 0, "terminal_lower": 0,
         "flagged": 0, "reordered_accurate": 0, "accurate_deepest_drop": 0.0, "flagged_min_drop": None,
         "higher_flagged_min_drop": None,
         "reorder_fail": {"beyond_modes": 0, "mixture": 0, "misaligned": 0, "eigenvalue": 0}}
    flags = []
    for s_, lad in sorted(seeds.items()):
        ks = sorted(lad)
        last = ks[-1]
        mm_mac = {k: (override or {}).get((s_, k), (lad[k]["matched_mode"], lad[k]["mac"])) for k in ks}
        for j in ks:
            r = lad[j]
            a = st.accurate(r)
            later = [i for i in ks if i > j and st._stable(lad[i])]
            f = any(_val(lad[i], key) < (1 - t) * _val(r, key) for i in later)
            drop = max((1 - _val(lad[i], key) / _val(r, key) for i in later), default=None)   # 가장 깊은 하강
            mm, mac = mm_mac[j]
            ok = r["outcome"] != "invalid_quadrature" and mac >= 0.9
            c["cells"] += 1
            c["flagged"] += f
            if f:
                flags.append((s_, j))
                c["flagged_min_drop"] = drop if c["flagged_min_drop"] is None else min(c["flagged_min_drop"], drop)
            if a and drop is not None:
                c["accurate_deepest_drop"] = max(c["accurate_deepest_drop"], drop)
            if a:
                c["accurate"] += 1
                c["accurate_testable"] += bool(later)
                c["accurate_flagged"] += f
            elif j == last:
                c["terminal_inaccurate"] += 1
                c["terminal_lower"] += ok and mm < j
            elif ok and mm > j:
                c["nonterminal_higher"] += 1
                c["nonterminal_higher_flagged"] += f
                if f:
                    h = c["higher_flagged_min_drop"]
                    c["higher_flagged_min_drop"] = drop if h is None else min(h, drop)
            elif ok and mm < j:
                c["nonterminal_lower"] += 1
                c["nonterminal_lower_flagged"] += f
            else:
                c["nonterminal_other"] += 1
                c["nonterminal_other_flagged"] += f
        lr = (lamref or {}).get(s_) or {m: r["Lam_ref"] for m, r in lad.items()}
        for rank, k in zip(ks, sorted(ks, key=lambda k_: lad[k_]["Lam"])):
            r = lad[k]
            mm, mac = mm_mac[k]
            members = (mm,) if (override and (s_, k) in override) else tuple(r.get("cluster") or (mm,))
            shape = r["outcome"] != "invalid_quadrature" and (mm == rank or rank in members)
            e = abs(r["Lam"] - lr[rank]) / lr[rank]
            if shape and mac >= 0.9 and e <= 0.05:
                c["reordered_accurate"] += 1
            elif shape and mac >= 0.9:
                c["reorder_fail"]["eigenvalue"] += 1          # 형상은 맞고 고유값이 허용오차 밖
            elif mac >= 0.9:
                c["reorder_fail"]["misaligned"] += 1          # 다른 모드에 깨끗이 섰다(순위가 어긋남)
            elif mac >= 0.5:
                c["reorder_fail"]["mixture"] += 1             # 모드 1–6 어느 것과도 MAC 0.9에 못 미치는 섞인 해
            else:
                c["reorder_fail"]["beyond_modes"] += 1        # 모드 1–6과 닮지 않았다 — 더 높은 모드
    c["flags"] = flags
    return c


def _sum(cs: list) -> dict:
    out = {}
    for c in cs:
        for k, v in c.items():
            if k == "flags":
                continue
            if k == "accurate_deepest_drop":
                out[k] = max(out.get(k, 0.0), v)
                continue
            if k in ("flagged_min_drop", "higher_flagged_min_drop"):
                if v is not None:
                    out[k] = v if out.get(k) is None else min(out[k], v)
                else:
                    out.setdefault(k, None)
                continue
            if isinstance(v, dict):
                out.setdefault(k, {})
                for k2, v2 in v.items():
                    out[k].setdefault(k2, 0)
                    out[k][k2] += v2
            else:
                out[k] = out.get(k, 0) + v
    return out


def _chains(d: str) -> dict:
    """참조를 읽지 않고 고른 사슬 — 끝까지 돈 프로토콜(양단 고정), 단계 5 선택을 전파한 근축퇴 단면, 모드 6 재풀이 검사를 더한
    L-브래킷 첫 통과(원좌표). 둘째는 단계 5의 Fourier 재풀이를 FEM 모드 1–8로 다시 채점한 값으로 덮는다."""
    from .run_p5_neural import e28_candidates, e28_near_candidates
    sel = json.load(open(os.path.join(d, st.E28_SELECTION), encoding="utf-8"))
    cc = {}
    for k in range(1, 7):
        cands = {p_: by_seed(_load(d, t)) for p_, t in e28_candidates(k).items()}
        for s_, p_ in sel["stages"][str(k)]["choice"].items():
            cc.setdefault(int(s_), {})[k] = cands[p_][int(s_)][k]
    first_cc = by_seed(_load(d, FIRST["(iii)"]))
    seln = json.load(open(os.path.join(d, st.E28_NEAR_SELECTION), encoding="utf-8"))
    near = by_seed(_load(d, FIRST["(v)"]))
    nr = {s_: {k: near[s_][k] for k in range(1, 5)} for s_ in near}
    for k in (5, 6):
        cands = {p_: by_seed(_load(d, t)) for p_, t in e28_near_candidates(k).items()}
        for s_, p_ in seln["stages"][str(k)]["choice"].items():
            nr[int(s_)][k] = cands[p_][int(s_)][k]
    ov = {}
    with open(os.path.join(d, "p5_exploit_e26e_e26_near_mc_s5.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if seln["stages"]["5"]["choice"][r["seed"]] == "ff":
                ov[(int(r["seed"]), 5)] = (int(r["resolve_mode"]), float(r["resolve_mac"]))
    lb = by_seed(_load(d, FIRST["(vi)"]))
    pick = st.lowest_lambda_pick({t: _load(d, t) for t in st.E26_RAW_PATH})["picked"]
    re6 = {t: by_seed(_load(d, t)) for t in st.E26_RAW_PATH}
    lbc = {s_: {**{k: lb[s_][k] for k in range(1, 6)}, 6: re6[pick[s_]][s_][6]} for s_ in lb}
    ref = lambda seeds: {s_: {m: r["Lam_ref"] for m, r in seeds[s_].items()} for s_ in seeds}
    return {"(iii) end-to-end protocol": (cc, ref(first_cc), None),
            "(v) stage-5 selection propagated": (nr, ref(near), ov),
            "(vi) re-solve test at mode 6": (lbc, ref(lb), None)}


def order_and_reorder(d: str = DATA_DIR, t: float = ORDER_T) -> dict:
    v2 = {(r["seed"], r["mode"]): r for r in _load(d, ORIGINAL_V2)}
    e7 = {(r["seed"], r["mode"]): r for r in _load(d, ORIGINAL["(i)"])}
    if sorted(v2) != sorted(e7) or any(v2[k]["Lam"] != e7[k]["Lam"] or v2[k]["outcome"] != e7[k]["outcome"]
                                       or v2[k]["matched_mode"] != e7[k]["matched_mode"] for k in v2):
        raise ValueError("E7이 v2의 정확 arm을 재현하지 않는다 — 기준 구조 원 설정의 사다리로 쓸 수 없다")
    out = {"t": t, "baseline_original_ladder": "E7 re-measurement (e7_ladder_a), which reproduces v2 bitwise",
           "e7_reproduces_v2": True,
           "e7_max_mac_difference": float(max(abs(v2[k]["mac"] - e7[k]["mac"]) for k in v2))}
    for group, lads in (("original", ORIGINAL), ("first_pass", FIRST)):
        res = {}
        for name, spec in lads.items():
            seeds = by_seed(_load(d, spec))
            c = order_check(seeds, t)
            ct = order_check(seeds, t, key="top")
            c["flags_same_with_top"] = c["flags"] == ct["flags"]
            c["flags_top"] = len(ct["flags"])
            res[name] = c
        tot = _sum(list(res.values()))
        tot["flags_same_with_top"] = int(sum(c["flags_same_with_top"] for c in res.values()))
        for c in res.values():
            c["flags"] = len(c["flags"])
        out[group] = {"ladders": res, "total": tot}
    ch = {}
    for name, (seeds, lr, ov) in _chains(d).items():
        c = order_check(seeds, t, lamref=lr, override=ov)
        c["flags"] = len(c["flags"])
        ch[name] = c
    out["chains"] = ch
    return out


# ---------------------------------------------------------------- 4. Table S16 재정의

def holds_modes(lad: dict, k: int, mac_min: float = 0.9) -> bool:
    """단계 1..k−1의 짝 모드가 순서와 관계없이 {1, …, k−1}이고 모두 MAC ≥ `mac_min`이며 하한 위반이 아닌가."""
    return (sorted(int(lad[j]["matched_mode"]) for j in range(1, k)) == list(range(1, k))
            and all(lad[j]["outcome"] != "invalid_quadrature" and lad[j]["mac"] >= mac_min for j in range(1, k)))


def span_error_by_identification(d: str = DATA_DIR) -> dict:
    """`p5_exploit_span_error.csv`의 δ²를 "앞 단계들이 참조 모드 1..k−1을 담았다"로 나눠 요약한다 — 단계 1..k−1의 짝 모드가
    순서와 관계없이 {1, …, k−1}이고 모두 MAC ≥ 0.9이며 하한 위반이 아니다. δ²는 스팬의 양이므로 순서를 묻지 않는다(단계 3·4가
    모드 4·3을 담은 시드의 단계 5는 맞다). 그 기준과 δ² > 10⁻²가 같은 시드-단계를 가르는지도 센다."""
    rows = list(csv.DictReader(open(os.path.join(d, "p5_exploit_span_error.csv"), encoding="utf-8")))
    out = {}
    for name, tag in S16_RUNS.items():
        recs = by_seed(st.load(d, tag))
        right, wrong = [], []
        cross = {"right_above_1e-2": 0, "wrong_below_1e-2": 0}
        for r in rows:
            if r["tag"] != tag:
                continue
            s_, k, eta2 = int(r["seed"]), int(r["stage"]), float(r["eta2"])
            ok = holds_modes(recs[s_], k)
            (right if ok else wrong).append((k, eta2, float(r["bound"]), float(r["c_eta2"]), s_))
            cross["right_above_1e-2"] += ok and eta2 > 1e-2
            cross["wrong_below_1e-2"] += (not ok) and eta2 <= 1e-2
        er = np.array([x[1] for x in right])
        ew = np.array([x[1] for x in wrong])
        worst = max(right, key=lambda x: x[1])
        out[name] = {"tag": tag, "n": len(right) + len(wrong), "right_n": len(right),
                     "right_median": float(np.median(er)), "right_min": float(er.min()), "right_max": float(er.max()),
                     "right_max_at": {"seed": worst[4], "stage": worst[0]},
                     "right_c_eta2_max": float(max(x[3] for x in right)),
                     "wrong_n": len(wrong), "wrong_by_stage": {str(k): sum(1 for x in wrong if x[0] == k) for k in range(2, 7)},
                     "wrong_eta2_range": [float(ew.min()), float(ew.max())] if len(ew) else None,
                     **cross, "bound_median": float(np.median([float(r["bound"]) for r in rows if r["tag"] == tag])),
                     "bound_violated": int(sum(float(r["eta2"]) > float(r["bound"]) * (1 + 1e-12) for r in rows
                                               if r["tag"] == tag))}
    return out


# ---------------------------------------------------------------- 5. 순서 검사 문턱의 민감도(r5.3 검토, 2026-11-08)

THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05)
OUT_THRESHOLD = "p5_posthoc_threshold.json"


def threshold_counts(seeds: dict, t: float, last_reported: int = 6) -> dict:
    """seeds: {시드: {단계: 기록}}. 보고하는 단계 1..`last_reported`만 센다. 단계 j는 뒤의 단계 i > j(덧붙인 단계 포함) 가운데
    사다리를 통과한 것의 감사 Λ가 (1 − t)Λ_j 아래면 울린다(`order_check`와 같은 규칙). 사슬은 고정한 채 t만 바꾸므로 검출만
    본다 — t가 달랐다면 2회차와 사슬도 달랐다."""
    c = {"inaccurate": 0, "inaccurate_flagged": 0, "accurate": 0, "accurate_testable": 0, "accurate_flagged": 0,
         "accurate_deepest_drop": 0.0}
    for s_, lad in sorted(seeds.items()):
        ks = sorted(lad)
        for j in (k for k in ks if k <= last_reported):
            r = lad[j]
            later = [i for i in ks if i > j and st._stable(lad[i])]
            f = any(lad[i]["Lam"] < (1 - t) * r["Lam"] for i in later)
            if st.accurate(r):
                c["accurate"] += 1
                c["accurate_testable"] += bool(later)
                c["accurate_flagged"] += f
                if later:
                    c["accurate_deepest_drop"] = max(c["accurate_deepest_drop"],
                                                     max(1 - lad[i]["Lam"] / r["Lam"] for i in later))
            else:
                c["inaccurate"] += 1
                c["inaccurate_flagged"] += f
    return c


def _staged_chains(d: str) -> dict:
    """단계적 배분의 사슬(단계 1–7) — E30(양단 고정·근축퇴 단면)의 1회차와 최종, E31(보류 두 구조)의 1회차(= 최종)와 첫 통과,
    그리고 같은 예산의 고른 일정(B3, 단계 1–6)."""
    from .run_p5_neural import B3_TAG, E31_FP_TAGS
    out = {}
    for s in ("cc", "nr", "cc07", "lb04"):
        c = st._e30_chains(d, s)
        out[f"{s} round 1"] = c["round1"]
        if s in ("cc", "nr"):
            out[f"{s} final"] = c["final"]
    for s in ("cc07", "lb04"):
        out[f"{s} first pass"] = by_seed(st.load(d, E31_FP_TAGS[s]))
    out["b3 uniform schedule"] = by_seed(st.load(d, B3_TAG))
    return out


def threshold_sensitivity(d: str = DATA_DIR, ts=THRESHOLDS) -> dict:
    """원 설정·첫 통과의 사다리, 프로토콜 사슬(끝까지 돈 프로토콜, 단계 5 선택, 모드 6 재풀이 검사), 단계적 배분의 사슬에 문턱
    t만 바꿔 순서 검사를 다시 적용한다. 칸마다 울린 부정확 단계 / 부정확 단계, 울린 정확 단계 / 검사할 수 있는 정확 단계."""
    groups = {"original": {n: by_seed(_load(d, spec)) for n, spec in ORIGINAL.items()},
              "first_pass": {n: by_seed(_load(d, spec)) for n, spec in FIRST.items()},
              "protocol_chains": {n: seeds for n, (seeds, _, _) in _chains(d).items()},
              "staged_allocation": _staged_chains(d)}
    out = {}
    for g, lads in groups.items():
        res = {n: {f"{t:.2f}": threshold_counts(seeds, t) for t in ts} for n, seeds in lads.items()}
        tot = {}
        for t in ts:
            cs = [res[n][f"{t:.2f}"] for n in res]
            tot[f"{t:.2f}"] = {k: (max(c[k] for c in cs) if k == "accurate_deepest_drop" else sum(c[k] for c in cs))
                               for k in cs[0]}
        out[g] = {"ladders": res, "total": tot}
    return out


def threshold_main(d: str = DATA_DIR) -> dict:
    from . import manifest
    out = {"role": "post hoc threshold sensitivity of the order check (r5.3 review of 2026-11-08); not judgments",
           "thresholds": list(THRESHOLDS),
           "rule": "stage j <= 6 is flagged when a later ladder-stable stage of the same seed has an audit eigenvalue "
                   "below (1 - t) times its own; the chains are held fixed, so only detection is read",
           "groups": threshold_sensitivity(d)}
    with open(os.path.join(d, OUT_THRESHOLD), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True,
                  default=lambda v: v.item() if hasattr(v, "item") else str(v))
        f.write("\n")
    manifest.write_json(os.path.join(d, f"manifest_{OUT_THRESHOLD}"), manifest.build({
        "driver": "p5_posthoc.threshold_main", "reads": "p5_neural_*_records.jsonl and the E28/E30/E31 selections",
        "thresholds": list(THRESHOLDS), "output": OUT_THRESHOLD, "launch": manifest.launch_provenance()}))
    return out


# ---------------------------------------------------------------- 6. 단계마다 가장 깊은 하강(r6.2 검토, 2026-11-10)
# 순서 검사가 읽는 양 1 − Λ_i/Λ_j를 단계마다 남긴다(문턱 민감도의 표가 세는 바로 그 값). 원고 그림의 자료다. 판정이 아니다.
OUT_DROPS = "p5_posthoc_order_drops.csv"


def drop_rows(group: str, ladder: str, seeds: dict, last_reported: int = 6) -> list:
    """seeds: {시드: {단계: 기록}}. 보고하는 단계 j = 1..`last_reported`마다, 같은 시드의 뒤 단계 i > j(덧붙인 단계 포함) 가운데
    사다리를 통과한 것의 가장 깊은 하강 1 − Λ_i/Λ_j(감사 Λ)와 그 i. 뒤에 안정 단계가 없으면 하강은 빈칸이다. 부류는
    `order_check`의 것(정확·더 높은 모드·더 낮은 모드·그 밖)이고, `same_cluster`는 j와 i가 같은 퇴화 군집에 짝지어졌는지다."""
    rows = []
    for s_, lad in sorted(seeds.items()):
        ks = sorted(lad)
        for j in (k for k in ks if k <= last_reported):
            r = lad[j]
            later = [i for i in ks if i > j and st._stable(lad[i])]
            by = max(later, key=lambda i: 1 - lad[i]["Lam"] / r["Lam"]) if later else None
            ok = r["outcome"] != "invalid_quadrature" and r["mac"] >= 0.9
            mm = r["matched_mode"]
            cls = ("accurate" if st.accurate(r) else "higher" if ok and mm > j else "lower" if ok and mm < j
                   else "other")
            cl_j, cl_i = tuple(r.get("cluster") or ()), tuple(lad[by].get("cluster") or ()) if by else ()
            rows.append({"group": group, "ladder": ladder, "seed": int(s_), "stage": int(j), "class": cls,
                         "matched_mode": mm, "mac": float(r["mac"]), "testable": bool(later),
                         "drop": "" if by is None else float(1 - lad[by]["Lam"] / r["Lam"]),
                         "by_stage": "" if by is None else int(by),
                         "same_cluster": bool(by is not None and len(cl_j) > 1 and cl_j == cl_i)})
    return rows


def order_drops(d: str = DATA_DIR) -> list:
    """원 설정·첫 통과(정규화 T-브래킷 대조를 뺀 여덟 구조)·단계적 배분의 사슬마다 `drop_rows`."""
    rows = []
    for name, spec in ORIGINAL.items():
        rows += drop_rows("original", name, by_seed(_load(d, spec)))
    for name, spec in FIRST.items():
        if name != "(viii) normalized":
            rows += drop_rows("first_pass", name, by_seed(_load(d, spec)))
    for name, seeds in _staged_chains(d).items():
        rows += drop_rows("staged_allocation", name, seeds)
    return rows


def order_drops_main(d: str = DATA_DIR) -> dict:
    from . import manifest
    rows = order_drops(d)
    manifest.write_csv(os.path.join(d, OUT_DROPS), rows)
    manifest.write_json(os.path.join(d, f"manifest_{OUT_DROPS.replace('.csv', '.json')}"), manifest.build({
        "driver": "p5_posthoc.order_drops_main", "reads": "p5_neural_*_records.jsonl and the E30/E31 selections",
        "output": OUT_DROPS, "rows": len(rows), "launch": manifest.launch_provenance(),
        "role": "post hoc (r6.2 review, 2026-11-10): for every reported stage j <= 6 of the original settings, the eight "
                "first passes, and the chains of the staged allocation, the deepest drop 1 - Lambda_i/Lambda_j over the "
                "later ladder-stable stages i of the same seed, the quantity that the order check compares with t; "
                "not judgments"}))
    return {"rows": len(rows), "output": OUT_DROPS}


def main(d: str = DATA_DIR) -> dict:
    out = {"role": "post hoc readings for the r4 manuscript (r3 review of 2026-10-29); not judgments",
           "audit_distance": audit_distance(d), "audit_distance_by_group": ladder_group_distance(d),
           "selection_equivalence": selection_equivalence(d), "order_check": order_and_reorder(d),
           "span_error_by_identification": span_error_by_identification(d)}
    with open(os.path.join(d, OUT), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True,
                  default=lambda v: v.item() if hasattr(v, "item") else str(v))
        f.write("\n")
    return out


if __name__ == "__main__":
    main()
