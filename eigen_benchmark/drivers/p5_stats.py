"""P5 원고가 인용하는 통계 — 커밋된 레코드만 읽고 계산해 `p5_stats.json`에 쓴다.

**시드 군집 부트스트랩.** 한 시드 안에서 모드 5는 모드 1–4의 저장 결과에 의존하므로
셀을 독립 관측으로 합치면 불확실도가 과소평가된다(원고 §4.1). 시드를 복원추출하고,
**같은 재표본 색인을 대비의 모든 칸에 쓴다** — 짝지은 설계다. 색인은 대비마다 새로
`default_rng(SEED)`에서 뽑으므로 모든 대비가 같은 재표본 열을 공유한다.

**로지스틱.** 뉴턴–랩슨. 부트스트랩 재표본은 행 가중(그 시드가 뽑힌 횟수)으로 넣고,
재표본 전체를 한 번에 푼다. 가중 적합은 행 복제 적합과 같다(시험).

**사다리 결측은 해상이 아니다.** 사다리가 도입되기 전에 돈 실행(E3 등)은 값이 없다.
`ladder_counts`는 미해상·해상·결측을 따로 센다.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

DATA_DIR = os.path.join("docs", "_generated", "data", "paper4")
WALL_MODES = (4, 5, 6)
LADDER_RESOLVED = 1e-3        # run_p5_neural.LADDER_RESOLVED와 같아야 한다(시험)
B_DEFAULT = 20_000
SEED = 12345

# E14 — 사전규정 2026-09-27 §4. 구조 순서가 고정효과 더미의 기준을 정한다(첫째가 기준).
E14_STRUCTURES = (("e14base", "baseline"), ("e14a", "e13a_thick"), ("e14b", "e13b_clamped2"))
E14_STAGES = (2, 3)
E14_CURVE_RANGE = (-3.3, 0.1, 139)   # log10 max|C| 격자(그림)

# E12 — 사전규정 2026-09-26 §2. (해상도, 격자 규약) → 실행 태그. 네 칸 모두 arm (b).
E12_CELLS = {("low", "resampled"): "e5_train_res",
             ("low", "fixed"): "e12lo_fixed_low",
             ("high", "resampled"): "e3_audit_resolution",
             ("high", "fixed"): "e12hi_fixed_high"}


# ---------------------------------------------------------------- 읽기

def arm_key(name: str) -> str:
    """arm 이름을 레코드의 대괄호 표기로. "diagonal"은 "reorth diagonal"의 부분문자열이므로
    부분일치로 고르면 두 arm이 섞인다 — 대괄호까지 포함해 고른다."""
    return f"[{name} projection]"


def load(data_dir: str, tag: str, arm: str | None = None,
         condition: str | None = None) -> list[dict]:
    """`p5_neural_{tag}_records.jsonl`. arm은 arm 문자열의 부분일치 — 모듈 안에서는 항상
    `arm_key`로 감싸 넘긴다."""
    p = os.path.join(data_dir, f"p5_neural_{tag}_records.jsonl")
    with open(p, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    if arm is not None:
        recs = [r for r in recs if arm in r["arm"]]
    if condition is not None:
        recs = [r for r in recs if r.get("condition") == condition]
    return recs


def seed_fraction(recs, modes=WALL_MODES) -> np.ndarray:
    """시드별 correct 비율, 시드 순. 시드가 모드 하나라도 빠지면 거부한다."""
    by_seed: dict[int, list] = {}
    for r in recs:
        if r["mode"] in modes:
            by_seed.setdefault(int(r["seed"]), []).append(r)
    out = []
    for s in sorted(by_seed):
        cells = by_seed[s]
        if sorted(c["mode"] for c in cells) != sorted(modes):
            raise ValueError(f"시드 {s}의 모드가 {sorted(modes)}와 다르다: "
                             f"{sorted(c['mode'] for c in cells)}")
        out.append(sum(c["outcome"] == "correct" for c in cells) / len(modes))
    return np.array(out)


def ladder_counts(recs, thr: float = LADDER_RESOLVED) -> dict:
    """사다리 꼬리 폭으로 셀을 미해상·해상·결측으로 가른다. 결측을 해상으로 세지 않는다."""
    c = {"unresolved": 0, "resolved": 0, "missing": 0}
    for r in recs:
        s = r.get("ladder_tail_spread")
        if s is None:
            c["missing"] += 1
        elif s > thr:
            c["unresolved"] += 1
        else:
            c["resolved"] += 1
    return c


# ---------------------------------------------------------------- 부트스트랩

def resample_index(n: int, B: int = B_DEFAULT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def _ci(point: float, draws: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cluster_mean(a, B: int = B_DEFAULT, seed: int = SEED):
    a = np.asarray(a, float)
    idx = resample_index(len(a), B, seed)
    return _ci(a.mean(), a[idx].mean(1))


def paired_bootstrap(a, b, B: int = B_DEFAULT, seed: int = SEED):
    """짝지은 차이 a − b의 (점추정, 2.5 %, 97.5 %). 시드가 군집이다."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("짝지은 비교는 같은 시드 집합이어야 한다")
    idx = resample_index(len(a), B, seed)
    return _ci(a.mean() - b.mean(), a[idx].mean(1) - b[idx].mean(1))


def interaction_bootstrap(a1, b1, a2, b2, B: int = B_DEFAULT, seed: int = SEED):
    """(a1 − b1) − (a2 − b2). 네 칸이 같은 재표본 색인을 쓴다."""
    arr = [np.asarray(v, float) for v in (a1, b1, a2, b2)]
    idx = resample_index(len(arr[0]), B, seed)
    m = [v[idx].mean(1) for v in arr]
    pt = (arr[0].mean() - arr[1].mean()) - (arr[2].mean() - arr[3].mean())
    return _ci(pt, (m[0] - m[1]) - (m[2] - m[3]))


# ---------------------------------------------------------------- 로지스틱

def _newton(X, y, W, iters: int = 100, tol: float = 1e-10):
    """가중 로지스틱 MLE를 W의 행마다 푼다. X (n,p), y (n,), W (B,n) → β (B,p), 수렴 (B,)."""
    B, p = W.shape[0], X.shape[1]
    beta = np.zeros((B, p))
    done = np.zeros(B, bool)
    for _ in range(iters):
        z = np.clip(beta @ X.T, -500, 500)
        mu = 1.0 / (1.0 + np.exp(-z))
        g = (W * (y - mu)) @ X
        H = np.einsum("bn,ni,nj->bij", W * mu * (1.0 - mu), X, X)
        try:
            step = np.linalg.solve(H, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        step[done] = 0.0
        beta += step
        done |= np.abs(step).max(1) < tol
        if done.all():
            break
    ok = done & np.isfinite(beta).all(1)
    return beta, ok


def logistic_fit(X, y, w=None):
    """단일 적합. 반환 (β, 수렴 여부)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    W = np.ones((1, len(y))) if w is None else np.asarray(w, float)[None, :]
    beta, ok = _newton(X, y, W)
    return beta[0], bool(ok[0])


def cluster_logistic(X, y, seed_of_row, B: int = B_DEFAULT, seed: int = SEED,
                     coef: int = 1):
    """시드 군집 부트스트랩으로 계수 하나의 구간. 반환 dict.

    **분리된 재표본을 버리지 않는다.** 한 구조의 correct 시드가 하나도 안 뽑히면 그 구조의
    더미가 −∞로 발산해(준완전 분리) 뉴턴이 수렴 판정을 못 내리지만, 다른 계수는 나머지
    관측이 식별하므로 유한하게 정착한다. 구간은 유한한 재표본 전체로 내고(`lo`, `hi`),
    수렴한 재표본만의 구간(`lo_converged`, `hi_converged`)과 미수렴 수를 함께 남긴다."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    sd = np.asarray(seed_of_row, int)
    n_seed = int(sd.max()) + 1
    beta, ok = logistic_fit(X, y)
    idx = resample_index(n_seed, B, seed)
    counts = np.stack([np.bincount(r, minlength=n_seed) for r in idx]).astype(float)
    bb, okb = _newton(X, y, counts[:, sd])
    finite = np.isfinite(bb[:, coef])
    pt, lo, hi = _ci(beta[coef], bb[finite, coef])
    _, lo_c, hi_c = _ci(beta[coef], bb[okb, coef])
    return {"beta": beta.tolist(), "converged": ok, "b": pt, "lo": lo, "hi": hi,
            "lo_converged": lo_c, "hi_converged": hi_c, "B": B,
            "n_failed": int((~okb).sum()), "n_nonfinite": int((~finite).sum())}


# ---------------------------------------------------------------- E14

def e14_observations(data_dir: str = DATA_DIR) -> dict:
    """구조 × 시드 × k ∈ {2, 3}. x = log10(단계 k 저장 스팬 최대 |C|), y = 모드 k+1 correct."""
    g, s, k, x, y = [], [], [], [], []
    for gi, (tag, _) in enumerate(E14_STRUCTURES):
        out = {(r["mode"], r["seed"]): r["outcome"] for r in load(data_dir, tag)}
        with open(os.path.join(data_dir, f"p5_span_per_seed_{tag}.csv"),
                  encoding="utf-8") as f:
            for r in csv.DictReader(f):
                kk = int(r["k"])
                if kk not in E14_STAGES:
                    continue
                tgt = int(r["stage_mode"])
                if tgt != kk + 1:
                    raise ValueError(f"{tag}: stage_mode {tgt} ≠ k + 1 = {kk + 1}")
                g.append(gi); s.append(int(r["seed"])); k.append(kk)
                x.append(np.log10(float(r["max_offdiag"])))
                y.append(out[(tgt, int(r["seed"]))] == "correct")
    return {"structure": np.array(g), "seed": np.array(s), "k": np.array(k),
            "x": np.array(x), "y": np.array(y, float)}


def e14_registered(obs: dict, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    X = np.column_stack([np.ones_like(obs["x"]), obs["x"]])
    r = cluster_logistic(X, obs["y"], obs["seed"], B, seed, coef=1)
    return {"b0": r["beta"][0], "b1": r["b"], "lo": r["lo"], "hi": r["hi"],
            "B": B, "n_failed": r["n_failed"], "n": int(len(obs["y"])),
            "n_correct": int(obs["y"].sum())}


def e14_fixed_effects(obs: dict, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """보조(판독 아님) — 구조 고정효과를 넣어도 β₁이 유지되는가."""
    d = [(obs["structure"] == j).astype(float) for j in range(1, len(E14_STRUCTURES))]
    X = np.column_stack([np.ones_like(obs["x"]), obs["x"], *d])
    r = cluster_logistic(X, obs["y"], obs["seed"], B, seed, coef=1)
    return {"b1": r["b"], "lo": r["lo"], "hi": r["hi"], "beta": r["beta"],
            "lo_converged": r["lo_converged"], "hi_converged": r["hi_converged"],
            "B": B, "n_failed": r["n_failed"], "n_nonfinite": r["n_nonfinite"]}


def _subset(obs: dict, mask) -> dict:
    return {k: v[mask] for k, v in obs.items()}


def e14_sensitivity(obs: dict, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """사후 민감도(판독 아님). 벽이 이미 저장 집합을 오염시킨 단계를 뺀다.

    사전규정은 기준 구조의 벽(모드 4)을 근거로 k ≥ 4만 뺐다. E13-b는 벽이 모드 3에서
    시작하므로 k = 3 저장 스팬에 이미 붕괴한 모드 3 출력이 들어 있고, 그 비대각은 전제조건이
    아니라 벽 자체를 일부 기록한다. `k2_only`는 세 구조 모두 벽 이전인 k = 2만, `pre_wall`은
    구조별 예측 벽 이전 단계만 쓴다."""
    walls = {0: 4, 1: 4, 2: 3}                   # E14_STRUCTURES 순서의 예측 벽
    pre = np.array([k + 1 <= walls[int(g)] for g, k in zip(obs["structure"], obs["k"])])
    return {"k2_only": e14_registered(_subset(obs, obs["k"] == 2), B, seed),
            "pre_wall": e14_registered(_subset(obs, pre), B, seed)}


def _e14_draws(obs: dict, B: int = B_DEFAULT, seed: int = SEED):
    """등록 모형(절편 + 기울기)의 점추정과 **수렴한** 시드 군집 부트스트랩 계수 (R, 2)."""
    X = np.column_stack([np.ones_like(obs["x"]), obs["x"]])
    beta, _ = logistic_fit(X, obs["y"])
    sd = obs["seed"]; n_seed = int(sd.max()) + 1
    idx = resample_index(n_seed, B, seed)
    counts = np.stack([np.bincount(r, minlength=n_seed) for r in idx]).astype(float)
    bb, ok = _newton(X, obs["y"], counts[:, sd])
    return beta, bb[ok]


def e14_logistic_curve(obs: dict, grid, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """그림용 — 적합 곡선과 시드 군집 부트스트랩 점별 95 % 띠."""
    beta, draws = _e14_draws(obs, B, seed)
    grid = np.asarray(grid, float)
    G = np.column_stack([np.ones_like(grid), grid])
    curves = 1.0 / (1.0 + np.exp(-(draws @ G.T)))
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    fit = 1.0 / (1.0 + np.exp(-(G @ beta)))
    return {"x": grid, "fit": fit, "lo": lo, "hi": hi}


# ---------------------------------------------------------------- E11 · E12 · 본실험

def e11(data_dir: str = DATA_DIR) -> dict:
    from scipy.stats import wilcoxon
    reorth = seed_fraction(load(data_dir, "e11_reorth_diag"))
    diag = seed_fraction(load(data_dir, "v2", arm=arm_key("diagonal"), condition="resampled"))
    exact = seed_fraction(load(data_dir, "v2", arm=arm_key("exact"), condition="resampled"))
    w = wilcoxon(reorth, diag, alternative="greater")
    return {"reorth_minus_diag": paired_bootstrap(reorth, diag),
            "reorth_minus_exact": paired_bootstrap(reorth, exact),
            "wilcoxon_reorth_gt_diag_p": float(w.pvalue),
            "correct_4_6": {"reorth": int(round(reorth.sum() * 3)),
                            "diag": int(round(diag.sum() * 3)),
                            "exact": int(round(exact.sum() * 3))}}


def main_arms(data_dir: str = DATA_DIR) -> dict:
    """원고 표 3 — 본실험 네 조건의 시드 군집 구간과 대각 대비 짝지은 차이."""
    from scipy.stats import wilcoxon
    arms = {"diagonal": ("diagonal", "resampled"), "exact": ("exact", "resampled"),
            "qr": ("qr", "resampled"), "fixed": ("exact", "fixed_audit")}
    f = {k: seed_fraction(load(data_dir, "v2", arm=arm_key(a), condition=c))
         for k, (a, c) in arms.items()}
    out = {k: cluster_mean(v) for k, v in f.items()}
    for k in ("exact", "qr"):
        out[f"{k}_minus_diagonal"] = paired_bootstrap(f[k], f["diagonal"])
        out[f"wilcoxon_{k}_gt_diagonal_p"] = float(
            wilcoxon(f[k], f["diagonal"], alternative="greater").pvalue)
    return out


def e12(data_dir: str = DATA_DIR) -> dict:
    f, lad, counts = {}, {}, {}
    for key, tag in E12_CELLS.items():
        recs = load(data_dir, tag, arm=arm_key("exact"))
        f[key] = seed_fraction(recs)
        deep = [r for r in recs if r["mode"] in WALL_MODES]
        lad["/".join(key)] = ladder_counts(deep)
        counts["/".join(key)] = int(round(f[key].sum() * len(WALL_MODES)))
    L, H = "low", "high"
    return {
        "correct_4_6": counts,
        "ladder_4_6": lad,
        "rule1_resampled_high_minus_low": paired_bootstrap(f[(H, "resampled")], f[(L, "resampled")]),
        "rule1_fixed_high_minus_low": paired_bootstrap(f[(H, "fixed")], f[(L, "fixed")]),
        "rule2_low_resampled_minus_fixed": paired_bootstrap(f[(L, "resampled")], f[(L, "fixed")]),
        "rule2_high_resampled_minus_fixed": paired_bootstrap(f[(H, "resampled")], f[(H, "fixed")]),
        "rule3_interaction": interaction_bootstrap(f[(H, "resampled")], f[(L, "resampled")],
                                                   f[(H, "fixed")], f[(L, "fixed")]),
    }


# E13 — 사전규정 2026-09-24. 대비 구간은 각 구조의 **예측 벽**부터 모드 6까지다.
E13_STRUCTURES = {"e13a_thick": 4, "e13b_clamped2": 3}


def e13(data_dir: str = DATA_DIR) -> dict:
    out = {}
    for tag, wall in E13_STRUCTURES.items():
        modes = tuple(range(wall, 7))
        f = {a: seed_fraction(load(data_dir, tag, arm=arm_key(a)), modes)
             for a in ("diagonal", "exact", "qr")}
        out[tag] = {"modes": list(modes),
                    "exact_minus_diagonal": paired_bootstrap(f["exact"], f["diagonal"]),
                    "qr_minus_diagonal": paired_bootstrap(f["qr"], f["diagonal"])}
    return out


# E15 — 사전규정 2026-09-30. 판정량은 게이트 없는 정확도다: 몬테카를로 훈련은 감사 격자
# 이력을 흔들어 가우스 훈련에서 보정한 τ가 옮겨 가지 않기 때문이다.
E15_TAGS = {"exact_mc": "e15b", "rowan_mc": "e15r"}


def accurate(rec, mac_min: float = 0.9, elam_max: float = 0.05) -> bool:
    """게이트 없는 정확: 하한 게이트는 두고 수렴 게이트만 뺀다."""
    import math
    if rec["outcome"] == "invalid_quadrature":
        return False
    e = rec["e_lam"]
    return (rec["matched_mode"] == rec["mode"] and rec["mac"] >= mac_min
            and math.isfinite(e) and e <= elam_max)


def _seed_share(recs, pred, modes=WALL_MODES) -> np.ndarray:
    by = {}
    for r in recs:
        if r["mode"] in modes:
            by.setdefault(int(r["seed"]), []).append(pred(r))
    return np.array([np.mean(by[s]) for s in sorted(by)])


def e15_from_records(recs_exact, recs_rowan, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """주 판정: 짝지은 차 D = acc(정확·MC) − acc(Rowan·MC), 모드 4–6, 시드 군집 부트스트랩.
    규칙 1: 0 제외·양수, 규칙 2: 0 포함, 규칙 3: 0 제외·음수. 기전 관측은 판정이 아니다."""
    fb, fr = _seed_share(recs_exact, accurate), _seed_share(recs_rowan, accurate)
    ci = paired_bootstrap(fb, fr, B, seed)
    verdict = "rule1" if ci[1] > 0 else "rule3" if ci[2] < 0 else "rule2"
    deep_r = [r for r in recs_rowan if r["mode"] in WALL_MODES]
    fail = {}
    for r in deep_r:
        if not accurate(r):
            fail[r["outcome"]] = fail.get(r["outcome"], 0) + 1
    pm = lambda rs: float(np.median([r["projected_mass_fraction"] for r in rs
                                     if r["mode"] in WALL_MODES]))
    off = {}
    for m in (3, 4, 5, 6):
        v = [r["batch_offdiag"] for r in recs_rowan
             if r["mode"] == m and r.get("batch_offdiag") is not None
             and np.isfinite(r["batch_offdiag"])]
        off[m] = float(np.median(v)) if v else None
    n = lambda rs: int(sum(accurate(r) for r in rs if r["mode"] in WALL_MODES))
    return {"acc_4_6": {"exact_mc": n(recs_exact), "rowan_mc": n(recs_rowan)},
            "contrast": ci, "verdict": verdict,
            "rowan_failure_classes": fail,
            "projected_mass_median": {"exact_mc": pm(recs_exact), "rowan_mc": pm(recs_rowan)},
            "rowan_batch_offdiag_median_by_mode": off}


def e15(data_dir: str = DATA_DIR) -> dict | None:
    try:
        b = load(data_dir, E15_TAGS["exact_mc"])
        r = load(data_dir, E15_TAGS["rowan_mc"])
    except FileNotFoundError:
        return None
    return e15_from_records(b, r)


# ---------------------------------------------------------------- E16

# 사전규정 2026-09-30-p5-e16-prespecification.md. arm마다 프로세스 하나라 태그도 arm마다다.
E16_TAGS = {"diagonal": "e16sq_diag", "reorth_diagonal": "e16sq_reorth",
            "exact": "e16sq_exact", "qr": "e16sq_qr"}
E16_MODE = 3                # 첫 축퇴쌍 바로 다음 단계
E16_GUARD_MODE2_MIN = 10    # 정확 arm 모드 2 correct 하한 — 미달이면 계기 고장, 판정 안 함
E16_FLOOR_MIN = 10          # 정확 arm 모드 3 correct가 이보다 적으면 검출력 부족을 공시
MIX_LOW, MIX_HIGH = 0.2, 0.8
MIX_MIN_IQR, MIX_MIN_TAIL = 0.2, 4


def poisson_binomial_pmf(P) -> np.ndarray:
    """성공확률이 행마다 다른 독립 시행의 성공 수 분포. P (R, n) → (R, n + 1)."""
    P = np.atleast_2d(np.asarray(P, float))
    out = np.zeros((P.shape[0], P.shape[1] + 1))
    out[:, 0] = 1.0
    for j in range(P.shape[1]):
        p = P[:, j:j + 1]
        out[:, 1:j + 2] = out[:, 1:j + 2] * (1 - p) + out[:, :j + 1] * p
        out[:, 0] *= (1 - p[:, 0])
    return out


def e14_transfer(x, observed: int, obs: dict, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """E14 등록 모형으로 표본 밖 시드들의 correct 수를 예측한다.

    계수의 시드 군집 부트스트랩(수렴한 재표본)마다 Poisson-이항 분포를 만들고 그 혼합의
    중앙 95 % 범위 [lo, hi]를 낸다 — 계수 불확실성을 전파한 예측 구간이다. 관측이 범위 안이면
    `consistent`, 아래면 `below`(법칙보다 해롭다), 위면 `above`(법칙보다 덜 해롭다)."""
    beta, draws = _e14_draws(obs, B, seed)
    x = np.asarray(x, float)
    G = np.column_stack([np.ones_like(x), x])
    if len(draws) == 0 or not np.isfinite(beta).all():
        return {"observed": int(observed), "verdict": "not_estimable",
                "n_replicates": int(len(draws))}
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    pmf = poisson_binomial_pmf(sig(draws @ G.T)).mean(0)
    cdf = np.cumsum(pmf)
    lo, hi = int(np.searchsorted(cdf, 0.025)), int(np.searchsorted(cdf, 0.975))
    return {"observed": int(observed), "expected": float(sig(G @ beta).sum()),
            "lo": lo, "hi": hi, "n": int(len(x)),
            "verdict": "below" if observed < lo else "above" if observed > hi else "consistent",
            "b0": float(beta[0]), "b1": float(beta[1]), "n_replicates": int(len(draws))}


def _slope(m, v):
    mc = m - m.mean(-1, keepdims=True)
    den = (mc ** 2).sum(-1)
    num = (mc * (v - v.mean(-1, keepdims=True))).sum(-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def mixing_regression(mu, y, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """시드 수준 OLS 기울기 y ~ μ와 시드 부트스트랩 구간(분산 0인 재표본은 버린다).

    규칙 a: 구간이 0 제외·양수(혼합을 따른다 = 성분 공유), b: 0 포함, c: 0 제외·음수.
    μ가 한쪽에 몰리면 기울기를 식별할 수 없으므로 사분위범위와 양 끝 시드 수로 `informative`를
    함께 낸다."""
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    idx = resample_index(len(mu), B, seed)
    draws = _slope(mu[idx], y[idx])
    q1, q3 = np.percentile(mu, [25, 75])
    n_lo, n_hi = int((mu < MIX_LOW).sum()), int((mu > MIX_HIGH).sum())
    fin = draws[np.isfinite(draws)]
    if fin.size == 0:                 # 모든 시드의 μ가 같다 — 기울기가 정의되지 않는다
        nan = float("nan")
        return {"slope": (nan, nan, nan), "verdict": "not_estimable",
                "iqr_mu": float(q3 - q1), "n_low": n_lo, "n_high": n_hi,
                "informative": False}
    pt, lo, hi = _ci(float(_slope(mu[None], y[None])[0]), fin)
    return {"slope": (pt, lo, hi),
            "verdict": "rule_a" if lo > 0 else "rule_c" if hi < 0 else "rule_b",
            "iqr_mu": float(q3 - q1), "n_low": n_lo, "n_high": n_hi,
            "informative": bool(q3 - q1 >= MIX_MIN_IQR and n_lo >= MIX_MIN_TAIL
                                and n_hi >= MIX_MIN_TAIL)}


def _by_seed(rows, key, pred) -> np.ndarray:
    by = {int(r["seed"]): key(r) for r in rows if pred(r)}
    return np.array([by[s] for s in sorted(by)], float)


def _c12(span_rows) -> np.ndarray:
    """k = 2 저장 스팬(모드 1·2)의 최대 |C| — 쌍 안의 비직교. 단계 3을 앞둔 값이어야 한다."""
    for r in span_rows:
        if int(r["k"]) == 2 and int(r["stage_mode"]) != 3:
            raise ValueError(f"k = 2 행의 stage_mode가 3이 아니다: {r['stage_mode']}")
    return _by_seed(span_rows, lambda r: float(r["max_offdiag"]), lambda r: int(r["k"]) == 2)


def e16_from_records(recs: dict, spans: dict, obs14: dict, B: int = B_DEFAULT,
                     seed: int = SEED) -> dict:
    """사전규정 §3을 그대로 적용한다. recs·spans는 arm → 기록·시드별 스팬 행."""
    correct = lambda r: float(r["outcome"] == "correct")
    counts = {a: {m: int(sum(r["outcome"] == "correct" for r in rs if r["mode"] == m))
                  for m in sorted({r["mode"] for r in rs})} for a, rs in recs.items()}
    ok = counts["exact"].get(2, 0) >= E16_GUARD_MODE2_MIN
    at = lambda a: _by_seed(recs[a], correct, lambda r: r["mode"] == E16_MODE)
    ci = paired_bootstrap(at("exact"), at("diagonal"), B, seed)
    verdict = ("not_read" if not ok else "rule1" if ci[1] > 0
               else "rule3" if ci[2] < 0 else "rule2")
    fails = {}
    for a, rs in recs.items():
        f = {}
        for r in rs:
            if r["mode"] == E16_MODE and r["outcome"] != "correct":
                f[r["outcome"]] = f.get(r["outcome"], 0) + 1
        fails[a] = f
    c12 = _c12(spans["diagonal"])
    x = np.log10(c12)
    n3 = counts["diagonal"].get(E16_MODE, 0)
    mu = _by_seed(recs["diagonal"], lambda r: 4.0 * r["component_fractions"][1]
                  * r["component_fractions"][2], lambda r: r["mode"] == 1)
    mix = mixing_regression(mu, x, B, seed)
    med = lambda v: float(np.median(v)) if len(v) else None
    # 1단계에는 사영이 없어 네 arm이 같은 계산이다 — 성분 비율이 비트 단위로 같아야 한다(서술).
    frac1 = lambda rs: [tuple(r["component_fractions"]) for r in
                        sorted((r for r in rs if r["mode"] == 1), key=lambda r: int(r["seed"]))]
    same_mu = all(frac1(rs) == frac1(recs["diagonal"]) for rs in recs.values())
    base = (obs14["structure"] == 0) & (obs14["k"] == 2)      # 규칙 b의 비교 기준(서술)
    mix.update({"median_C_low": med(c12[mu < MIX_LOW]), "median_C_high": med(c12[mu > MIX_HIGH]),
                "baseline_k2_median_C": med(10.0 ** obs14["x"][base])})
    return {"correct_by_mode": counts, "instrument_ok": bool(ok),
            "primary": {"contrast": ci, "verdict": verdict,
                        "floor": counts["exact"].get(E16_MODE, 0) < E16_FLOOR_MIN},
            "other_contrasts": {f"exact_minus_{a}": paired_bootstrap(at("exact"), at(a), B, seed)
                                for a in ("reorth_diagonal", "qr") if a in recs},
            "mode3_failure_forms": fails,
            "e14_transfer": e14_transfer(x, n3, obs14, B, seed),
            "e14_transfer_k2_only": e14_transfer(x, n3, _subset(obs14, obs14["k"] == 2), B, seed),
            "mixing": mix, "mu_identical_across_arms": bool(same_mu),
            "c12_median_by_arm": {a: float(np.median(_c12(sp))) for a, sp in spans.items()},
            "mu": mu.tolist(), "log10_c12": x.tolist()}


def e16(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {a: load(data_dir, t) for a, t in E16_TAGS.items()}
        spans = {}
        for a, t in E16_TAGS.items():
            with open(os.path.join(data_dir, f"p5_span_per_seed_{t}.csv"), encoding="utf-8") as f:
                spans[a] = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    return e16_from_records(recs, spans, e14_observations(data_dir))


# ---------------------------------------------------------------- E17–E20

# 사전규정 2026-10-02-p5-e17-e20-prespecification.md.
def unpaired_bootstrap(a, b, B: int = B_DEFAULT, seed: int = SEED):
    """독립 두 집단 a − b의 (점, 2.5 %, 97.5 %). 집단마다 시드를 따로 재표집한다 — 구조가 다르면
    같은 시드 번호라도 짝이 아니다(E18)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(B, len(a)))
    ib = rng.integers(0, len(b), size=(B, len(b)))
    return _ci(a.mean() - b.mean(), a[ia].mean(1) - b[ib].mean(1))


def _rule(ci) -> str:
    return "rule1" if ci[1] > 0 else "rule3" if ci[2] < 0 else "rule2"


def _same_sign_excluding_zero(*cis) -> bool:
    return all(_excludes_zero(c) for c in cis) and len({c[0] > 0 for c in cis}) == 1


def _n_acc(recs, modes=WALL_MODES) -> int:
    return int(sum(accurate(r) for r in recs if r["mode"] in modes))


# E17 — 정확 arm의 훈련 구적 2 × 2. (배치, 재추첨 주기) → (태그, arm 부분일치, 조건).
E17_CELLS = {("gauss", 250): ("v2", "exact", "resampled"), ("mc", 1): ("e15b", None, None),
             ("gauss", 1): ("e17_gauss1", None, None), ("mc", 250): ("e17_mc250", None, None)}


def e17_from_cells(cells: dict, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """판정량은 모드 4–6의 게이트 없는 정확도(E15와 같다). 네 칸 모두 같은 시드 0–19라 짝짓는다.
    규칙 1: 배치 대비(MC − 가우스)가 두 주기에서 같은 부호로 0 제외. 규칙 2: 주기 대비(매 반복 −
    250회)가 두 배치에서 같은 부호로 0 제외. 규칙 3: 상호작용이 0 제외. 셋 다 아니면 분해 불가."""
    acc = {k: _seed_share(v, accurate) for k, v in cells.items()}
    g250, g1, m250, m1 = acc[("gauss", 250)], acc[("gauss", 1)], acc[("mc", 250)], acc[("mc", 1)]
    c = {"placement_at_every1": paired_bootstrap(m1, g1, B, seed),
         "placement_at_every250": paired_bootstrap(m250, g250, B, seed),
         "frequency_at_gauss": paired_bootstrap(g1, g250, B, seed),
         "frequency_at_mc": paired_bootstrap(m1, m250, B, seed),
         "interaction": interaction_bootstrap(m1, m250, g1, g250, B, seed)}
    r1 = _same_sign_excluding_zero(c["placement_at_every1"], c["placement_at_every250"])
    r2 = _same_sign_excluding_zero(c["frequency_at_gauss"], c["frequency_at_mc"])
    r3 = _excludes_zero(c["interaction"])
    return {"acc_4_6": {f"{p}_{f}": _n_acc(v) for (p, f), v in cells.items()}, "contrasts": c,
            "verdict": {"rule1": r1, "rule2": r2, "rule3": r3,
                        "indecomposable": not (r1 or r2 or r3)}}


def e17(data_dir: str = DATA_DIR) -> dict | None:
    try:
        cells = {k: load(data_dir, t, arm=None if a is None else arm_key(a), condition=cond)
                 for k, (t, a, cond) in E17_CELLS.items()}
    except FileNotFoundError:
        return None
    return e17_from_cells(cells)


# E18 — 근접 축퇴(b/h = 1.05) 대 정확 축퇴(E16). 같은 τ = 0.5, 같은 판정량(correct).
def e18_from_records(recs18: dict, spans18_diag, recs16_diag, obs14, B: int = B_DEFAULT,
                     seed: int = SEED) -> dict:
    """주 판정: acc₃(대각, E18) − acc₃(대각, E16), 짝짓지 않는 부트스트랩. 규칙 1(양수): 축퇴가
    아니면 벽이 풀린다 — E16의 추가 해는 정확한 축퇴의 것. 규칙 2: 0 포함. 규칙 3: 음수.
    보조: E18에서 E14 관계의 이전(S1과 같은 계산)."""
    correct = lambda r: float(r["outcome"] == "correct")
    at3 = lambda rs: _by_seed(rs, correct, lambda r: r["mode"] == E16_MODE)
    a18, a16 = at3(recs18["diagonal"]), at3(recs16_diag)
    ci = unpaired_bootstrap(a18, a16, B, seed)
    c12 = _c12(spans18_diag)
    n3 = int(a18.sum())
    fails = {}
    for r in recs18["diagonal"]:
        if r["mode"] == E16_MODE and r["outcome"] != "correct":
            fails[r["outcome"]] = fails.get(r["outcome"], 0) + 1
    out = {"primary": {"contrast": ci, "verdict": _rule(ci)},
           "correct_mode3": {"e18_diagonal": n3, "e16_diagonal": int(a16.sum())},
           "e14_transfer": e14_transfer(np.log10(c12), n3, obs14, B, seed),
           "c12_median": float(np.median(c12)), "mode3_failure_forms_diagonal": fails}
    if "exact" in recs18:
        out["d3_within_e18"] = paired_bootstrap(at3(recs18["exact"]), a18, B, seed)
    return out


def e18(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {"diagonal": load(data_dir, "e18_diag"), "exact": load(data_dir, "e18_exact")}
        with open(os.path.join(data_dir, "p5_span_per_seed_e18_diag.csv"), encoding="utf-8") as f:
            spans = list(csv.DictReader(f))
        r16 = load(data_dir, E16_TAGS["diagonal"])
    except FileNotFoundError:
        return None
    return e18_from_records(recs, spans, r16, e14_observations(data_dir))


# E19 — 정사각 단면 정확 arm의 비틀림·축방향 순서 역전. 예산 2배 / 입력 정규화.
E19_MODES = (5, 6)


def e19_from_records(cond_recs: dict, recs16_exact, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """조건마다 acc(모드 5–6, 게이트 없음) − E16 정확 arm, 시드 짝지음. 게이트가 예산에 불변이
    아니므로(원고 5절) 수렴 게이트를 쓰지 않는다. 서술: 5단계 정합 군집."""
    base = _seed_share(recs16_exact, accurate, modes=E19_MODES)
    out = {"e16_exact_acc_5_6": _n_acc(recs16_exact, E19_MODES)}
    for name, rs in cond_recs.items():
        acc = _seed_share(rs, accurate, modes=E19_MODES)
        ci = paired_bootstrap(acc, base, B, seed)
        cl = {}
        for r in rs:
            if r["mode"] == 5:
                k = "-".join(map(str, r.get("cluster") or [r["matched_mode"]]))
                cl[k] = cl.get(k, 0) + 1
        out[name] = {"contrast": ci, "verdict": _rule(ci), "acc_5_6": _n_acc(rs, E19_MODES),
                     "stage5_clusters": cl}
    return out


def e19(data_dir: str = DATA_DIR) -> dict | None:
    try:
        conds = {t: load(data_dir, t) for t in ("e19_budget", "e19_norm")}
        r16 = load(data_dir, E16_TAGS["exact"])
    except FileNotFoundError:
        return None
    return e19_from_records(conds, r16)


# E20 — Rowan 규약 × MC에서 표본 크기 4배.
def e20_from_records(recs20, recs15r, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """D = acc(모드 4–6, 게이트 없음, B = 4,536) − acc(E15-r, B = 1,134), 시드 짝지음."""
    a, b = _seed_share(recs20, accurate), _seed_share(recs15r, accurate)
    ci = paired_bootstrap(a, b, B, seed)
    off = {}
    for m in (3, 4, 5, 6):
        v = [r["batch_offdiag"] for r in recs20 if r["mode"] == m
             and r.get("batch_offdiag") is not None and np.isfinite(r["batch_offdiag"])]
        off[m] = float(np.median(v)) if v else None
    return {"contrast": ci, "verdict": _rule(ci),
            "acc_4_6": {"e20r_b4": _n_acc(recs20), "e15r": _n_acc(recs15r)},
            "batch_offdiag_median_by_mode": off}


def e20(data_dir: str = DATA_DIR) -> dict | None:
    try:
        r20, r15 = load(data_dir, "e20r_b4"), load(data_dir, E15_TAGS["rowan_mc"])
    except FileNotFoundError:
        return None
    return e20_from_records(r20, r15)



# E21 — L-브래킷(사전규정 2026-10-04-p5-e21-prespecification.md). 상자가 아닌 기하에서의 재현.
E21_TAGS = {"diagonal": "e21lb_diag", "exact": "e21lb_exact", "qr": "e21lb_qr"}
E21_MODES = (4, 5, 6)          # 성분 공유 필요조건이 지목한 벽(k* = 3의 다음 모드)부터
E21_GUARD_MODE2_MIN = 10       # 2단계는 저장이 하나라 규칙에 불변 — 미달이면 계기 고장(E16과 같다)
E21_FLOOR_MIN = 10             # 정확 arm 모드 4–6 correct 합(60 중)이 이보다 적으면 검출력 부족 공시
E21_STAGES = (2, 3)            # E14 법칙이 적합된 단계


def _span_c(span_rows, k: int) -> np.ndarray:
    """단계 k 저장 스팬의 최대 |C|, 시드 순. 단계 k의 목표 모드는 k + 1이어야 한다."""
    for r in span_rows:
        if int(r["k"]) == k and int(r["stage_mode"]) != k + 1:
            raise ValueError(f"k = {k} 행의 stage_mode가 {k + 1}이 아니다: {r['stage_mode']}")
    return _by_seed(span_rows, lambda r: float(r["max_offdiag"]), lambda r: int(r["k"]) == k)


def elam_shape_right(recs, mac_min: float = 0.9, elam_max: float = 0.05) -> dict:
    """모드별, 형상은 맞은 셀(정합 = 목표, MAC ≥ mac_min, 하한 게이트 통과)의 e_λ — 개수, 중앙값,
    e_λ > elam_max인 셀 수(형상은 맞고 Λ만 문턱 밖). 모서리 특이(E21)·재료 계면(E22)의 값을 본다."""
    out = {}
    for m in sorted({r["mode"] for r in recs}):
        e = [r["e_lam"] for r in recs if r["mode"] == m and r["outcome"] != "invalid_quadrature"
             and r["matched_mode"] == m and r["mac"] >= mac_min and np.isfinite(r["e_lam"])]
        out[m] = {"n": len(e), "median": float(np.median(e)) if e else None,
                  "over_threshold": int(sum(v > elam_max for v in e))}
    return out


def _replication(recs: dict, spans: dict, obs14: dict, modes, B: int, seed: int) -> dict:
    """새 구조 하나에서의 재현 판정(E21·E22 공통). recs·spans는 arm → 기록·시드별 스팬 행.

    주 판정 D = acc(정확, modes) − acc(대각, modes), correct, 시드 짝지음(E13과 같은 판정량).
    보조 S1은 E14 법칙의 관측 단위(시드 × k ∈ {2, 3})를 그대로 모아 대각 arm의 모드 3·4 correct
    합을 예측 범위와 비교한다."""
    def by_mode(rs, pred):
        return {m: int(sum(pred(r) for r in rs if r["mode"] == m))
                for m in sorted({r["mode"] for r in rs})}
    counts = {a: by_mode(rs, lambda r: r["outcome"] == "correct") for a, rs in recs.items()}
    acc = {a: by_mode(rs, accurate) for a, rs in recs.items()}
    ok = counts["exact"].get(2, 0) >= E21_GUARD_MODE2_MIN
    f = {a: seed_fraction(rs, modes) for a, rs in recs.items()}
    ci = paired_bootstrap(f["exact"], f["diagonal"], B, seed)
    n_exact = sum(counts["exact"].get(m, 0) for m in modes)
    fails = {}
    for a, rs in recs.items():
        g = {}
        for r in rs:
            if r["mode"] in modes and r["outcome"] != "correct":
                g[r["outcome"]] = g.get(r["outcome"], 0) + 1
        fails[a] = g
    xs = {k: np.log10(_span_c(spans["diagonal"], k)) for k in E21_STAGES}
    x = np.concatenate([xs[k] for k in E21_STAGES])
    obs = {k: counts["diagonal"].get(k + 1, 0) for k in E21_STAGES}
    k2 = _subset(obs14, obs14["k"] == 2)
    return {"correct_by_mode": counts, "accurate_by_mode": acc, "instrument_ok": bool(ok),
            "primary": {"contrast": ci, "verdict": "not_read" if not ok else _rule(ci),
                        "modes": list(modes), "floor": n_exact < E21_FLOOR_MIN},
            "other_contrasts": {
                "qr_minus_diagonal": paired_bootstrap(f["qr"], f["diagonal"], B, seed),
                "exact_minus_qr": paired_bootstrap(f["exact"], f["qr"], B, seed)},
            "failure_forms_4_6": fails,
            "e14_transfer": e14_transfer(x, sum(obs.values()), obs14, B, seed),
            "e14_transfer_k2_only": e14_transfer(x, sum(obs.values()), k2, B, seed),
            "e14_transfer_by_stage": {k: e14_transfer(xs[k], obs[k], obs14, B, seed)
                                      for k in E21_STAGES},
            "c_median_by_arm": {a: {k: float(np.median(_span_c(sp, k))) for k in E21_STAGES}
                                for a, sp in spans.items()},
            "elam_shape_right": {a: elam_shape_right(rs) for a, rs in recs.items()}}


def e21_from_records(recs: dict, spans: dict, obs14: dict, B: int = B_DEFAULT,
                     seed: int = SEED) -> dict:
    """사전규정 2026-10-04 E21 §3을 그대로 적용한다(판정 모드 4–6)."""
    return _replication(recs, spans, obs14, E21_MODES, B, seed)


def e21(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {a: load(data_dir, t) for a, t in E21_TAGS.items()}
        spans = {}
        for a, t in E21_TAGS.items():
            with open(os.path.join(data_dir, f"p5_span_per_seed_{t}.csv"), encoding="utf-8") as f:
                spans[a] = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    return e21_from_records(recs, spans, e14_observations(data_dir))



def training_exploitation(recs) -> dict:
    """모드별 훈련 구적 착취의 지문. 훈련 격자에서 잰 Λ가 참조 고유값 아래인 셀 수(정확히 적분했다면
    Rayleigh 하한 때문에 나올 수 없다)와, 실패 셀의 감사/훈련 Λ 비 중앙값. E21에서 사후로 본 것을
    E21b에서 사전 등록한다."""
    out = {}
    for m in sorted({r["mode"] for r in recs}):
        sel = [r for r in recs if r["mode"] == m]
        fail = [r["Lam"] / r["Lam_train"] for r in sel if not accurate(r) and r["Lam_train"] > 0]
        out[m] = {"train_below_ref": int(sum(r["Lam_train"] < r["Lam_ref"] for r in sel)),
                  "median_audit_over_train_failing": float(np.median(fail)) if fail else None}
    return out


# E21b — E21 정확 arm의 재추첨 주기 1(사전규정 2026-10-06-p5-e21b-prespecification.md).
E21B_TAG = "e21lb_exact_g1"


def e21b_from_records(recs_g1, recs_exact, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """D = acc₄₋₆(매 반복) − acc₄₋₆(E21 정확), 게이트 없는 정확도, 시드 짝지음(E17·E19와 같은 판정량).
    규칙 1(0 제외, 양수) → 매 반복 재추첨이 L에서 정확 사영의 복구를 되살린다."""
    a = _seed_share(recs_g1, accurate, E21_MODES)
    b = _seed_share(recs_exact, accurate, E21_MODES)
    ci = paired_bootstrap(a, b, B, seed)
    named = {E21B_TAG: recs_g1, E21_TAGS["exact"]: recs_exact}
    return {"contrast": ci, "verdict": _rule(ci),
            "acc_4_6": {k: _n_acc(rs, E21_MODES) for k, rs in named.items()},
            "accurate_by_mode": {k: {m: int(sum(accurate(r) for r in rs if r["mode"] == m))
                                     for m in sorted({r["mode"] for r in rs})}
                                 for k, rs in named.items()},
            "exploitation": {k: training_exploitation(rs) for k, rs in named.items()},
            "ladder_unresolved": {k: {m: ladder_counts([r for r in rs if r["mode"] == m])["unresolved"]
                                      for m in sorted({r["mode"] for r in rs})}
                                  for k, rs in named.items()}}


def e21b(data_dir: str = DATA_DIR) -> dict | None:
    try:
        g1, ex = load(data_dir, E21B_TAG), load(data_dir, E21_TAGS["exact"])
    except FileNotFoundError:
        return None
    return e21b_from_records(g1, ex)


# E22 — 두 층 이종재료(사전규정 2026-10-04-p5-e22-prespecification.md). E21과 같은 규칙에 같은 기하의
# 균질 구조(E13-a)와의 비교를 서술로 더한다. 두 구조의 τ가 달라 비교는 게이트 없는 정확도로 한다.
E22_TAGS = {"diagonal": "e22bi_diag", "exact": "e22bi_exact", "qr": "e22bi_qr"}
E22_MODES = (4, 5, 6)          # 성분 공유 필요조건이 지목한 벽(k* = 3의 다음 모드)부터
E22_COMPARE_TAG = "e13a_thick"


def e22_from_records(recs: dict, spans: dict, obs14: dict, recs13a: dict | None = None,
                     B: int = B_DEFAULT, seed: int = SEED) -> dict:
    out = _replication(recs, spans, obs14, E22_MODES, B, seed)
    if recs13a is not None:
        out["vs_e13a_gate_free"] = {
            a: unpaired_bootstrap(_seed_share(recs[a], accurate, E22_MODES),
                                  _seed_share(recs13a[a], accurate, E22_MODES), B, seed)
            for a in recs if a in recs13a}
        out["e13a_elam_shape_right"] = {a: elam_shape_right(rs) for a, rs in recs13a.items()}
    return out


def e22(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {a: load(data_dir, t) for a, t in E22_TAGS.items()}
        spans = {}
        for a, t in E22_TAGS.items():
            with open(os.path.join(data_dir, f"p5_span_per_seed_{t}.csv"), encoding="utf-8") as f:
                spans[a] = list(csv.DictReader(f))
        r13 = {a: load(data_dir, E22_COMPARE_TAG, arm=arm_key(a)) for a in E22_TAGS}
    except FileNotFoundError:
        return None
    return e22_from_records(recs, spans, e14_observations(data_dir), r13)


def _excludes_zero(t) -> bool:
    return t[1] * t[2] > 0


def e12_verdict(r: dict) -> dict:
    """사전규정 2026-09-26 §3의 네 규칙을 그대로 적용한다."""
    r1 = all(_excludes_zero(r[k]) and r[k][0] > 0 for k in
             ("rule1_resampled_high_minus_low", "rule1_fixed_high_minus_low"))
    r2 = all(_excludes_zero(r[k]) for k in
             ("rule2_low_resampled_minus_fixed", "rule2_high_resampled_minus_fixed"))
    r3 = _excludes_zero(r["rule3_interaction"])
    return {"rule1": r1, "rule2": r2, "rule3": r3,
            "indecomposable": not (r1 or r2 or r3)}


def main(data_dir: str = DATA_DIR, outdir: str | None = None) -> dict:
    from . import manifest
    d = manifest.ensure_outdir(data_dir if outdir is None else outdir)
    obs = e14_observations(data_dir)
    res = {"seed": SEED, "B": B_DEFAULT,
           "main_arms": main_arms(data_dir),
           "e11": e11(data_dir),
           "e12": e12(data_dir),
           "e13": e13(data_dir),
           "e15": e15(data_dir),
           "e16": e16(data_dir),
           "e17": e17(data_dir), "e18": e18(data_dir), "e19": e19(data_dir), "e20": e20(data_dir),
           "e21": e21(data_dir), "e21b": e21b(data_dir), "e22": e22(data_dir),
           "e14_registered": e14_registered(obs),
           "e14_fixed_effects": e14_fixed_effects(obs),
           "e14_sensitivity_post_hoc": e14_sensitivity(obs)}
    # 그림용 곡선 — 렌더러는 계산하지 않으므로 여기서 쓴다.
    cur = e14_logistic_curve(obs, np.linspace(*E14_CURVE_RANGE))
    res["e14_curve"] = {k: [float(v) for v in cur[k]] for k in ("x", "fit", "lo", "hi")}
    res["e12_verdict"] = e12_verdict(res["e12"])
    manifest.write_json(os.path.join(d, "p5_stats.json"), res)
    manifest.write_json(os.path.join(d, "manifest_p5_stats.json"), manifest.build({
        "driver": "p5_stats.main", "reads": "p5_neural_*_records.jsonl, p5_span_per_seed_*.csv",
        "B": B_DEFAULT, "seed": SEED, "wall_modes": list(WALL_MODES),
        "ladder_resolved": LADDER_RESOLVED}))
    return res


if __name__ == "__main__":
    main()
