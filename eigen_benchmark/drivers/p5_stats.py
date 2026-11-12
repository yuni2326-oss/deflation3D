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


def _vs_condition(recs_new, recs_base, new_tag: str, base_tag: str, B: int, seed: int) -> dict:
    """L-브래킷 정확 arm 두 조건의 대비. D = acc₄₋₆(조건) − acc₄₋₆(기준), 게이트 없는 정확도, 시드 짝지음
    (E17·E19와 같은 판정량). 착취(하한 위반)와 사다리 미해상을 두 조건에 나란히 붙인다."""
    a = _seed_share(recs_new, accurate, E21_MODES)
    b = _seed_share(recs_base, accurate, E21_MODES)
    ci = paired_bootstrap(a, b, B, seed)
    named = {new_tag: recs_new, base_tag: recs_base}
    return {"contrast": ci, "verdict": _rule(ci),
            "acc_4_6": {k: _n_acc(rs, E21_MODES) for k, rs in named.items()},
            "accurate_by_mode": {k: {m: int(sum(accurate(r) for r in rs if r["mode"] == m))
                                     for m in sorted({r["mode"] for r in rs})}
                                 for k, rs in named.items()},
            "exploitation": {k: training_exploitation(rs) for k, rs in named.items()},
            "ladder_unresolved": {k: {m: ladder_counts([r for r in rs if r["mode"] == m])["unresolved"]
                                      for m in sorted({r["mode"] for r in rs})}
                                  for k, rs in named.items()}}


def _vs_e21_exact(recs_new, recs_exact, new_tag: str, B: int, seed: int) -> dict:
    """대 E21 정확 arm(E21b·E21c·E21e 공통)."""
    return _vs_condition(recs_new, recs_exact, new_tag, E21_TAGS["exact"], B, seed)


def e21b_from_records(recs_g1, recs_exact, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """규칙 1(0 제외, 양수) → 매 반복 재추첨이 L에서 정확 사영의 복구를 되살린다."""
    return _vs_e21_exact(recs_g1, recs_exact, E21B_TAG, B, seed)


# E21c — E21 정확 arm의 격자족을 촘촘하게(사전규정 2026-10-08-p5-e21c-prespecification.md).
E21C_TAG = "e21lb_exact_dense"


def e21c_from_records(recs_dense, recs_exact, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """규칙 1(0 제외, 양수) → 촘촘한 격자족이 L에서 정확 사영의 복구를 되살린다(밀도 부족이 원인)."""
    return _vs_e21_exact(recs_dense, recs_exact, E21C_TAG, B, seed)


def e21c(data_dir: str = DATA_DIR) -> dict | None:
    try:
        dense, ex = load(data_dir, E21C_TAG), load(data_dir, E21_TAGS["exact"])
    except FileNotFoundError:
        return None
    return e21c_from_records(dense, ex)




# E21d — E21b의 점 배치를 균등 몬테카를로로(사전규정 2026-10-09-p5-e21d-prespecification.md).
E21D_TAG = "e21lb_exact_mc1"


def e21d_from_records(recs_mc, recs_g1, recs_exact, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """판정은 대 E21b(같은 매 반복 재추첨, 점 배치만 다르다). `vs_e21_exact`는 같은 판정량을 E21 정확 arm과
    시드끼리 짝지어 읽는다(사전규정 §3.1)."""
    out = _vs_condition(recs_mc, recs_g1, E21D_TAG, E21B_TAG, B, seed)
    ci = paired_bootstrap(_seed_share(recs_mc, accurate, E21_MODES),
                          _seed_share(recs_exact, accurate, E21_MODES), B, seed)
    out["vs_e21_exact"] = {"contrast": ci, "verdict": _rule(ci),
                           "acc_4_6": {E21D_TAG: _n_acc(recs_mc, E21_MODES),
                                       E21_TAGS["exact"]: _n_acc(recs_exact, E21_MODES)}}
    return out


def e21d(data_dir: str = DATA_DIR) -> dict | None:
    try:
        mc, g1 = load(data_dir, E21D_TAG), load(data_dir, E21B_TAG)
        ex = load(data_dir, E21_TAGS["exact"])
    except FileNotFoundError:
        return None
    return e21d_from_records(mc, g1, ex)

# E21e — E21 정확 arm에 기준 구조의 고해상 격자족(사전규정 2026-10-09-p5-e21e-prespecification.md).
E21E_TAG = "e21lb_exact_hires"


def e21e_from_records(recs_hires, recs_exact, recs_dense, B: int = B_DEFAULT,
                      seed: int = SEED) -> dict:
    """판정은 E21b·E21c와 같다(대 E21 정확). `vs_e21c`는 같은 판정량을 E21c와 시드끼리 짝지어 읽는다 —
    기준 구조의 밀도를 넘는 밀도가 더 살리는지(사전규정 §3.1)."""
    out = _vs_e21_exact(recs_hires, recs_exact, E21E_TAG, B, seed)
    ci = paired_bootstrap(_seed_share(recs_hires, accurate, E21_MODES),
                          _seed_share(recs_dense, accurate, E21_MODES), B, seed)
    out["vs_e21c"] = {"contrast": ci, "verdict": _rule(ci),
                      "acc_4_6": {E21E_TAG: _n_acc(recs_hires, E21_MODES),
                                  E21C_TAG: _n_acc(recs_dense, E21_MODES)}}
    return out


def e21e(data_dir: str = DATA_DIR) -> dict | None:
    try:
        hires, ex = load(data_dir, E21E_TAG), load(data_dir, E21_TAGS["exact"])
        dense = load(data_dir, E21C_TAG)
    except FileNotFoundError:
        return None
    return e21e_from_records(hires, ex, dense)


# E21f — L-브래킷 모드 6의 지도 적합(사전규정 2026-10-10-p5-e21f-prespecification.md).
E21F_FILES = {6: "p5_e1_oracle_fit_e21f_m6.csv", 8: "p5_e1_oracle_fit_e21f_m8.csv"}
E21F_CONTROL_RULE = 0.05


def e21f_from_rows(rows6, rows8, lam_e21d6) -> dict:
    """주 판정: 모드 6에 맞춘 망을 E21d의 저장 모드 1–5로 정확 사영한 Rayleigh 몫의 중앙값 대 E21d 모드 6 단계가
    실제로 도달한 Λ의 중앙값. 낮으면 규칙 1(더 낮은 함수가 시행공간에 있었는데 최적화가 놓쳤다), 아니면 규칙 2
    (이 예산에서 표현의 한계). 대조: 모드 8 적합의 맞춘 격자 밖 e_λ 중앙값이 0.05 미만이면 적합 절차가 그
    정확도를 낸다."""
    ld = float(np.median([float(r["Lam_deflated"]) for r in rows6]))
    le = float(np.median([float(v) for v in lam_e21d6]))
    e6 = float(np.median([float(r["e_lam_check_0"]) for r in rows6]))
    e8 = float(np.median([float(r["e_lam_check_0"]) for r in rows8]))
    return {"lam_deflated_median": ld, "lam_e21d_median": le,
            "verdict": "rule1" if ld < le else "rule2",
            "n_below_e21d": int(sum(float(r["Lam_deflated"]) < le for r in rows6)),
            "e_lam_held_out_median": {"6": e6, "8": e8},
            "control_ok": bool(e8 < E21F_CONTROL_RULE)}


def e21f(data_dir: str = DATA_DIR) -> dict | None:
    import csv
    try:
        rows = {m: list(csv.DictReader(open(os.path.join(data_dir, f), encoding="utf-8")))
                for m, f in E21F_FILES.items()}
        mc = load(data_dir, E21D_TAG)
    except FileNotFoundError:
        return None
    return e21f_from_rows(rows[6], rows[8], [r["Lam"] for r in mc if int(r["mode"]) == 6])


# E21g — 입력 정규화(사전규정 2026-10-13-p5-e21g-prespecification.md).
E21G_TAGS = {"mc": "e21lb_exact_mc1_norm", "gauss": "e21lb_exact_norm"}


def e21g_from_records(recs_mc_norm, recs_mc, recs_gauss_norm, recs_exact, B: int = B_DEFAULT,
                      seed: int = SEED) -> dict:
    """주 판정 `mode6`: 모드 6의 게이트 없는 정확, MC + 정규화 대 E21d(시드 짝지음). 보조 `mc_4_6`: 같은 두 조건의
    모드 4–6. 보조 `gauss_4_6`: 가우스 + 정규화 대 E21 정확의 모드 4–6(`_vs_condition`)."""
    a = _seed_share(recs_mc_norm, accurate, (6,))
    b = _seed_share(recs_mc, accurate, (6,))
    ci = paired_bootstrap(a, b, B, seed)
    mode6 = {"contrast": ci, "verdict": _rule(ci),
             "acc_6": {E21G_TAGS["mc"]: _n_acc(recs_mc_norm, (6,)), E21D_TAG: _n_acc(recs_mc, (6,))}}
    mc = _vs_condition(recs_mc_norm, recs_mc, E21G_TAGS["mc"], E21D_TAG, B, seed)
    ga = _vs_condition(recs_gauss_norm, recs_exact, E21G_TAGS["gauss"], E21_TAGS["exact"], B, seed)
    return {"mode6": mode6, "mc_4_6": mc, "gauss_4_6": ga}


def e21g(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {k: load(data_dir, t) for k, t in E21G_TAGS.items()}
        mc, ex = load(data_dir, E21D_TAG), load(data_dir, E21_TAGS["exact"])
    except FileNotFoundError:
        return None
    return e21g_from_records(recs["mc"], mc, recs["gauss"], ex)


# E21h — L-브래킷 모드 6 단계의 학습 일정과 망 폭(사전규정 2026-10-14-p5-e21h-prespecification.md). 새 실행은
# 앞 모드 1–5를 저장된 망으로 두고 모드 6 단계만 다시 풀었으므로 기록은 모드 6뿐이다.
E21H_TAGS = {"sched_norm": "e21lb_s6_norm_x4", "sched_raw": "e21lb_s6_raw_x4",
             "wide_norm": "e21lb_s6_norm_w256"}
E21H_BASE = {"sched_norm": E21G_TAGS["mc"], "sched_raw": E21D_TAG, "wide_norm": E21G_TAGS["mc"]}


def e21h_from_records(recs_sched_norm, recs_e21g_mc, recs_sched_raw, recs_e21d, recs_wide_norm,
                      B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """세 대비 모두 모드 6의 게이트 없는 정확 — 새 조건 대 그 조건이 앞 모드를 가져온 실행, 시드 짝지음.
    주 판정은 `sched_norm`(정규화, 반복 4배 대 E21g-mc)이고 나머지 둘은 보조 판정이다."""
    def c6(new, base, cond):
        a, b = _seed_share(new, accurate, (6,)), _seed_share(base, accurate, (6,))
        ci = paired_bootstrap(a, b, B, seed)
        return {"contrast": ci, "verdict": _rule(ci),
                "acc_6": {E21H_TAGS[cond]: _n_acc(new, (6,)), E21H_BASE[cond]: _n_acc(base, (6,))}}
    return {"primary": "sched_norm",
            "sched_norm": c6(recs_sched_norm, recs_e21g_mc, "sched_norm"),
            "sched_raw": c6(recs_sched_raw, recs_e21d, "sched_raw"),
            "wide_norm": c6(recs_wide_norm, recs_e21g_mc, "wide_norm")}


def e21h(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {k: load(data_dir, t) for k, t in E21H_TAGS.items()}
        g_mc, d_mc = load(data_dir, E21G_TAGS["mc"]), load(data_dir, E21D_TAG)
    except FileNotFoundError:
        return None
    return e21h_from_records(recs["sched_norm"], g_mc, recs["sched_raw"], d_mc, recs["wide_norm"])


# E23 — 학습률이 높은 구간을 묶은 일정과 Fourier 특징(사전규정 2026-10-17-p5-e23-prespecification.md). 둘 다
# E21g-mc의 저장된 모드 1–5에서 모드 6 단계만 다시 풀었다. 주 판정은 lowtail 대 E21g-mc, 보조는 lowtail 대
# E21h sched_norm(같은 저장 모드, 같은 시드)과 fourier 대 E21g-mc다.
E23_TAGS = {"lowtail": "e21lb_s6_norm_x4lo", "fourier": "e21lb_s6_norm_ff"}


def e23_from_records(recs_lowtail, recs_e21g_mc, recs_sched_norm, recs_fourier,
                     B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """세 대비 모두 모드 6의 게이트 없는 정확을 시드끼리 짝지어 비교한다."""
    def c6(new, base, new_tag, base_tag):
        a, b = _seed_share(new, accurate, (6,)), _seed_share(base, accurate, (6,))
        ci = paired_bootstrap(a, b, B, seed)
        return {"contrast": ci, "verdict": _rule(ci),
                "acc_6": {new_tag: _n_acc(new, (6,)), base_tag: _n_acc(base, (6,))}}
    return {"primary": "lowtail",
            "lowtail": c6(recs_lowtail, recs_e21g_mc, E23_TAGS["lowtail"], E21G_TAGS["mc"]),
            "lowtail_vs_sched": c6(recs_lowtail, recs_sched_norm, E23_TAGS["lowtail"],
                                   E21H_TAGS["sched_norm"]),
            "fourier": c6(recs_fourier, recs_e21g_mc, E23_TAGS["fourier"], E21G_TAGS["mc"])}


def e23(data_dir: str = DATA_DIR) -> dict | None:
    try:
        lo, ff = (load(data_dir, E23_TAGS[k]) for k in ("lowtail", "fourier"))
        g_mc, sn = load(data_dir, E21G_TAGS["mc"]), load(data_dir, E21H_TAGS["sched_norm"])
    except FileNotFoundError:
        return None
    return e23_from_records(lo, g_mc, sn, ff)

def e21b(data_dir: str = DATA_DIR) -> dict | None:
    try:
        g1, ex = load(data_dir, E21B_TAG), load(data_dir, E21_TAGS["exact"])
    except FileNotFoundError:
        return None
    return e21b_from_records(g1, ex)


# E24 — 강형식 물리정보 arm(사전규정 2026-10-17-p5-e24-prespecification.md). 두 해석기는 같은 초기화·같은 내부 표본을
# 쓰므로 시드끼리 짝짓는다. 주 판정은 대수 층(저장된 낮은 모드로 돌아간 셀의 몫, 모드 2–6), 보조는 계기 층(사다리 불안정,
# 모드 4–6)과 최적화 층(모드 6 정확). 판정 구조는 box(파일럿 규칙 P가 바꾸면 thick)와 L-브래킷이다.
E24_TAGS = {"box": "e24_pinn_box", "thick": "e24_pinn_thick", "lbracket": "e24_pinn_lb"}
E24_COMPARATOR = {"box": ("e7_ladder_a", "exact", "resampled"), "thick": ("e13a_thick", "exact", "resampled"),
                  "lbracket": ("e21lb_exact_mc1_norm", None, None)}
E24_BASIN_MODES = (2, 3, 4, 5, 6)


def returned_to_stored(rec, mac_min: float = 0.9) -> bool:
    """게이트 없는 '저장된 낮은 모드로 돌아감' — 짝 모드가 단계보다 낮고 MAC ≥ 0.9. 하한 위반 셀은 세지 않는다."""
    return (rec["outcome"] != "invalid_quadrature" and rec["matched_mode"] < rec["mode"]
            and rec["mac"] >= mac_min)


def ladder_unstable(rec) -> bool:
    s = rec.get("ladder_tail_spread")
    return s is not None and s > LADDER_RESOLVED


def e24_from_records(pinn: dict, var: dict, box_key: str = "box", B: int = B_DEFAULT,
                     seed: int = SEED) -> dict:
    """`pinn`·`var`는 {구조: 기록}. 세 층의 대비를 두 구조 모두 계산하고, 판정 구조를 표시한다."""
    def pair(a_recs, b_recs, pred, modes):
        a, b = _seed_share(a_recs, pred, modes), _seed_share(b_recs, pred, modes)
        ci = paired_bootstrap(a, b, B, seed)
        n = lambda rs: int(sum(pred(r) for r in rs if r["mode"] in modes))
        return {"contrast": ci, "verdict": _rule(ci), "count": {"strong": n(a_recs), "variational": n(b_recs)},
                "cells": int(len(a) * len(modes))}
    out = {"primary": f"algebra@{box_key}", "judged": {"algebra": box_key, "instrument": box_key,
                                                        "optimization": "lbracket"}}
    for st, recs in pinn.items():
        v = var[st]
        out[st] = {"algebra": pair(recs, v, returned_to_stored, E24_BASIN_MODES),
                   "instrument": pair(recs, v, ladder_unstable, WALL_MODES),
                   "optimization": pair(recs, v, accurate, (6,)),
                   "accurate_by_mode": {m: {"strong": _n_acc(recs, (m,)), "variational": _n_acc(v, (m,))}
                                        for m in range(1, 7)}}
    return out


def e24(data_dir: str = DATA_DIR) -> dict | None:
    box_key = "thick" if os.path.exists(os.path.join(data_dir, f"p5_neural_{E24_TAGS['thick']}_records.jsonl")) \
        else "box"
    try:
        pinn = {k: load(data_dir, E24_TAGS[k]) for k in (box_key, "lbracket")}
        var = {}
        for k in (box_key, "lbracket"):
            tag, arm, cond = E24_COMPARATOR[k]
            var[k] = load(data_dir, tag, arm=None if arm is None else arm_key(arm), condition=cond)
    except FileNotFoundError:
        return None
    return e24_from_records(pinn, var, box_key)


# E25 — T-브래킷 전향 적용(사전규정 2026-10-17-p5-e25-prespecification.md). 주 판정은 프로토콜 arm(처방 반영) 대
# 원 설정 정확 arm의 모드 1–6 게이트 없는 정확, 시드 짝지음. 보조는 원 설정의 정확 − 대각(모드 4–6), 성분 공유 규칙의
# 시작 모드, 등록 법칙(식 9)의 전이(게이트 없는 정확으로 셈), 처방 단계의 전후. T의 τ는 L에서 빌린 값이라 게이트를 쓰지 않는다.
E25_TAGS = {"orig_diag": "e25tb_diag", "orig_exact": "e25tb_exact", "protocol": "e25tb_protocol"}
E25_PREDICTED_ONSET = 4
E25_ONSET_MAX = 2              # 시작 모드 = 게이트 없는 정확이 2/20 이하로 떨어지는 첫 모드


def e25_combine(protocol, remedy=None) -> list:
    """처방을 반영한 프로토콜 arm의 기록. remedy = (k, 단계 k 기록, 단계 k+1–6 기록 또는 None)."""
    if remedy is None:
        return list(protocol)
    k, step1, step2 = remedy
    out = [r for r in protocol if r["mode"] < k] + [r for r in step1 if r["mode"] == k]
    if k < 6:
        if step2 is None:
            raise ValueError("k < 6인 처방에는 이어 푼 단계의 기록이 필요하다")
        out += [r for r in step2 if r["mode"] > k]
    return out


def first_drop_mode(recs, n_max: int = E25_ONSET_MAX):
    by = {m: _n_acc(recs, (m,)) for m in sorted({r["mode"] for r in recs})}
    return next((m for m in sorted(by) if by[m] <= n_max), None)


def e25_from_records(recs: dict, span_diag, obs14: dict, remedy=None, B: int = B_DEFAULT,
                     seed: int = SEED) -> dict:
    """recs = {orig_diag, orig_exact, protocol: 기록}, span_diag = 대각 arm의 시드별 스팬 행."""
    prot = e25_combine(recs["protocol"], remedy)
    modes = (1, 2, 3, 4, 5, 6)
    pa, oa = _seed_share(prot, accurate, modes), _seed_share(recs["orig_exact"], accurate, modes)
    ci = paired_bootstrap(pa, oa, B, seed)
    det = paired_bootstrap(_seed_share(recs["orig_exact"], accurate, WALL_MODES),
                           _seed_share(recs["orig_diag"], accurate, WALL_MODES), B, seed)
    xs = {k: np.log10(_span_c(span_diag, k)) for k in E21_STAGES}
    x = np.concatenate([xs[k] for k in E21_STAGES])
    obs = {k: _n_acc(recs["orig_diag"], (k + 1,)) for k in E21_STAGES}
    onset = first_drop_mode(recs["orig_diag"])
    out = {"primary": {"contrast": ci, "verdict": _rule(ci), "modes": list(modes)},
           "accurate_by_mode": {"protocol": {m: _n_acc(prot, (m,)) for m in modes},
                                "protocol_first_pass": {m: _n_acc(recs["protocol"], (m,)) for m in modes},
                                **{a: {m: _n_acc(rs, (m,)) for m in modes} for a, rs in recs.items()
                                   if a != "protocol"}},
           "detector": {"contrast": det, "verdict": _rule(det), "modes": list(WALL_MODES)},
           "onset": {"predicted": E25_PREDICTED_ONSET, "observed": onset,
                     "hit": bool(onset == E25_PREDICTED_ONSET)},
           "e14_transfer": e14_transfer(x, sum(obs.values()), obs14, B, seed),
           "c_median_diag": {k: float(np.median(_span_c(span_diag, k))) for k in E21_STAGES},
           "train_violations": {a: int(sum(1 for r in rs if r["mode"] >= 3
                                           and r.get("lam_margin") is not None
                                           and r["Lam_train"] < r["Lam"] / (1.0 + r["lam_margin"]) * (1 - 1e-6)))
                                for a, rs in recs.items()},
           "remedy": None}
    if remedy is not None:
        k = remedy[0]
        a, b = _seed_share(prot, accurate, (k,)), _seed_share(recs["protocol"], accurate, (k,))
        rc = paired_bootstrap(a, b, B, seed)
        out["remedy"] = {"stage": int(k), "contrast": rc, "verdict": _rule(rc)}
    return out


def e25(data_dir: str = DATA_DIR) -> dict | None:
    try:
        recs = {a: load(data_dir, t) for a, t in E25_TAGS.items()}
        with open(os.path.join(data_dir, f"p5_span_per_seed_{E25_TAGS['orig_diag']}.csv"), encoding="utf-8") as f:
            span = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    remedy = None
    for k in range(4, 7):
        p1 = os.path.join(data_dir, f"p5_neural_e25tb_remedy_k{k}_records.jsonl")
        if os.path.exists(p1):
            step2 = (load(data_dir, f"e25tb_remedy_k{k}_rest") if k < 6 else None)
            remedy = (k, load(data_dir, f"e25tb_remedy_k{k}"), step2)
            break
    return e25_from_records(recs, span, e14_observations(data_dir), remedy)


# E26 — 프로토콜 전/후를 네 구조로(사전규정 2026-10-18-p5-e26-prespecification.md). 구조마다 D = acc₁₋₆(MC 매 반복·정확)
# − acc₁₋₆(원 설정 정확 arm), 게이트 없는 정확, 시드 짝지음. fourier_raw는 모드 6, E21d 대비. 두 재풀이(fourier_raw,
# fourier_base)는 재풀이 일관성 검사의 판독량 — 시드별 Λ 상대 감소 (Λ_원 − Λ_재)/Λ_원과 허용오차 0.05를 넘은 시드 수 — 을 같이 적는다.
E26_TAGS = {"thick": "e26_thick_mc", "clamped2": "e26_cc_mc", "near": "e26_near_mc", "bilayer": "e26_bi_mc",
            "fourier_raw": "e21lb_s6_raw_ff", "fourier_base": "e15b_s6_raw_ff", "tbracket_raw": "e26_tb_raw_mc"}
E26_RESOLVE_BASE = {"fourier_raw": "e21lb_exact_mc1", "fourier_base": "e15b"}
E26_RESOLVE_TOL = 0.05
E26_ORIGINAL = {"thick": ("e13a_thick", "exact", "resampled"), "clamped2": ("e13b_clamped2", "exact", "resampled"),
                "near": ("e18_exact", None, None), "bilayer": ("e22bi_exact", None, None)}


def resolve_drop(resolved, original, mode: int = 6, tol: float = E26_RESOLVE_TOL) -> dict:
    """재풀이 일관성 검사의 판독량. 같은 저장 모드에서 다시 푼 단계의 감사 Λ가 원래 단계보다 얼마나 내려갔는가."""
    a = {int(r["seed"]): r["Lam"] for r in resolved if r["mode"] == mode}
    b = {int(r["seed"]): r["Lam"] for r in original if r["mode"] == mode}
    if sorted(a) != sorted(b):
        raise ValueError(f"재풀이와 원 실행의 시드가 다르다: {sorted(a)} 대 {sorted(b)}")
    d = np.array([(b[s] - a[s]) / b[s] for s in sorted(a)])
    return {"drop": d.tolist(), "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max()),
            "n_above_tol": int((d > tol).sum()), "n_lowered": int((d > 0).sum()), "tol": tol}


def e26_tbracket(raw, orig_exact, protocol_norm, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """E26d(사전규정 §2.5) — T-브래킷 원좌표 프로토콜. 주: acc₁₋₆(원좌표 프로토콜) − acc₁₋₆(원 설정 정확), 보조: 굽힘 모드
    1·3에서 acc(원좌표) − acc(정규화 프로토콜, E25), 모드 1·3의 e_λ 중앙(형상이 맞는 셀). 시드 짝지음, 게이트 없는 정확."""
    modes = (1, 2, 3, 4, 5, 6)
    a, b = _seed_share(raw, accurate, modes), _seed_share(orig_exact, accurate, modes)
    main = paired_bootstrap(a, b, B, seed)
    out = {"contrast": main, "verdict": _rule(main),
           "accurate_by_mode": {"raw": {m: _n_acc(raw, (m,)) for m in modes},
                                "orig_exact": {m: _n_acc(orig_exact, (m,)) for m in modes}}}
    if protocol_norm is not None:
        a, b = _seed_share(raw, accurate, (1, 3)), _seed_share(protocol_norm, accurate, (1, 3))
        ci = paired_bootstrap(a, b, B, seed)
        out["coordinates"] = {"contrast": ci, "verdict": _rule(ci),
                              "accurate_by_mode_norm": {m: _n_acc(protocol_norm, (m,)) for m in modes}}
        for name, recs in (("raw", raw), ("norm", protocol_norm)):
            for m in (1, 3):
                e = [r["e_lam"] for r in recs if r["mode"] == m and r["matched_mode"] == m and r["mac"] >= 0.9]
                out["coordinates"][f"e_lam_median_{name}_m{m}"] = float(np.median(e)) if e else None
    return out


def lowest_lambda_pick(runs: dict, mode: int = 6) -> dict:
    """재풀이 검사의 "낮은 해를 지킨다" 규칙 — 시드마다 여러 풀이 가운데 감사 Λ가 가장 낮은 것을 고른다(참조를 읽지
    않는 선택). 고른 풀이의 게이트 없는 정확 수와 시드별로 고른 실행 이름을 낸다."""
    by = {name: {int(r["seed"]): r for r in recs if r["mode"] == mode} for name, recs in runs.items()}
    seeds = sorted(next(iter(by.values())))
    if any(sorted(v) != seeds for v in by.values()):
        raise ValueError("풀이마다 시드가 다르다")
    pick = {s_: min(by, key=lambda name: by[name][s_]["Lam"]) for s_ in seeds}
    return {"accurate": int(sum(accurate(by[pick[s_]][s_]) for s_ in seeds)), "n": len(seeds),
            "picked": {s_: pick[s_] for s_ in seeds},
            "accurate_each": {name: int(sum(accurate(by[name][s_]) for s_ in seeds)) for name in by}}


E26_RAW_PATH = ("e21lb_exact_mc1", "e21lb_s6_raw_x4", "e21lb_s6_raw_ff")
E26_NORM_PATH = ("e21lb_exact_mc1_norm", "e21lb_s6_norm_x4", "e21lb_s6_norm_x4lo", "e21lb_s6_norm_ff")


def e26_from_records(proto: dict, orig: dict, ff_raw=None, e21d=None, B: int = B_DEFAULT, seed: int = SEED,
                     ff_base=None, e15b=None) -> dict:
    modes = (1, 2, 3, 4, 5, 6)
    out = {}
    for st_, recs in proto.items():
        a, b = _seed_share(recs, accurate, modes), _seed_share(orig[st_], accurate, modes)
        ci = paired_bootstrap(a, b, B, seed)
        out[st_] = {"contrast": ci, "verdict": _rule(ci),
                    "accurate_by_mode": {"protocol": {m: _n_acc(recs, (m,)) for m in modes},
                                         "original": {m: _n_acc(orig[st_], (m,)) for m in modes}}}
    if ff_raw is not None:
        a, b = _seed_share(ff_raw, accurate, (6,)), _seed_share(e21d, accurate, (6,))
        ci = paired_bootstrap(a, b, B, seed)
        out["fourier_raw"] = {"contrast": ci, "verdict": _rule(ci),
                              "acc_6": {"fourier_raw": _n_acc(ff_raw, (6,)), "e21d": _n_acc(e21d, (6,))},
                              "resolve": resolve_drop(ff_raw, e21d)}
    if ff_base is not None:
        a, b = _seed_share(ff_base, accurate, (6,)), _seed_share(e15b, accurate, (6,))
        ci = paired_bootstrap(a, b, B, seed)
        out["fourier_base"] = {"contrast": ci, "verdict": _rule(ci),
                               "acc_6": {"fourier_base": _n_acc(ff_base, (6,)), "e15b": _n_acc(e15b, (6,))},
                               "resolve": resolve_drop(ff_base, e15b)}
    return out


def e26(data_dir: str = DATA_DIR) -> dict | None:
    proto, orig = {}, {}
    for k in ("thick", "clamped2", "near", "bilayer"):
        try:
            proto[k] = load(data_dir, E26_TAGS[k])
            tag, arm, cond = E26_ORIGINAL[k]
            orig[k] = load(data_dir, tag, arm=None if arm is None else arm_key(arm), condition=cond)
        except FileNotFoundError:
            proto.pop(k, None)
    try:
        ff, d_ = load(data_dir, E26_TAGS["fourier_raw"]), load(data_dir, E26_RESOLVE_BASE["fourier_raw"])
    except FileNotFoundError:
        ff = d_ = None
    try:
        fb, b_ = load(data_dir, E26_TAGS["fourier_base"]), load(data_dir, E26_RESOLVE_BASE["fourier_base"])
    except FileNotFoundError:
        fb = b_ = None
    try:
        tb = load(data_dir, E26_TAGS["tbracket_raw"])
    except FileNotFoundError:
        tb = None
    if not proto and ff is None and fb is None and tb is None:
        return None
    out = e26_from_records(proto, orig, ff, d_, ff_base=fb, e15b=b_)
    if tb is not None:
        def _maybe(tag):
            try:
                return load(data_dir, tag)
            except FileNotFoundError:
                return None
        oe, pn = _maybe(E25_TAGS["orig_exact"]), _maybe(E25_TAGS["protocol"])
        if oe is not None:
            out["tbracket_raw"] = e26_tbracket(tb, oe, pn)
    # 표 10의 L-브래킷 프로토콜 경로(보조 판독, 사전규정 §3.1): 원좌표는 E26a가 있어야, 정규화는 기존 자료로.
    for key, tags in (("lbracket_raw_path", E26_RAW_PATH), ("lbracket_norm_path", E26_NORM_PATH)):
        try:
            out[key] = lowest_lambda_pick({t: load(data_dir, t) for t in tags})
        except FileNotFoundError:
            pass
    return out


# E26e — 첫 통과 단계의 재풀이 검사(사전규정 2026-10-22-p5-e26e-prespecification.md). 단계 k의 첫 풀이(E26b)와 재풀이를 시드끼리
# 짝지어, 검사가 울린 시드(Λ 감소 > 0.05)를 첫 풀이의 부류(정확·더 높은 모드·그 밖)로 나누고 정확의 전후를 센다.
def higher_mode(rec, mac_min: float = 0.9) -> bool:
    return rec["outcome"] != "invalid_quadrature" and rec["matched_mode"] > rec["mode"] and rec["mac"] >= mac_min


E26E_MIN_HIGHER = 5


def e26e_rule_stages(first, min_higher: int = E26E_MIN_HIGHER) -> list:
    """사전규정 E26e의 규칙 — 첫 통과에서 셀 5/20 이상이 더 높은 모드에 자리 잡은(짝 모드 > 단계, MAC ≥ 0.9) 단계 2–6."""
    return [k for k in range(2, 7) if sum(higher_mode(r) for r in first if r["mode"] == k) >= min_higher]


def e26e_from_records(first, resolved, k: int, tol: float = E26_RESOLVE_TOL) -> dict:
    a = {int(r["seed"]): r for r in first if r["mode"] == k}
    b = {int(r["seed"]): r for r in resolved if r["mode"] == k}
    if sorted(a) != sorted(b):
        raise ValueError("첫 풀이와 재풀이의 시드가 다르다")
    rd = resolve_drop(resolved, first, mode=k, tol=tol)
    flag = {s_: d > tol for s_, d in zip(sorted(a), rd["drop"])}
    cls = {s_: ("accurate" if accurate(a[s_]) else "higher" if higher_mode(a[s_]) else "other") for s_ in a}
    table = {c: {"n": sum(1 for s_ in a if cls[s_] == c), "flagged": sum(1 for s_ in a if cls[s_] == c and flag[s_]),
                 "recovered": sum(1 for s_ in a if cls[s_] == c and accurate(b[s_]))} for c in ("accurate", "higher", "other")}
    keep = {s_: (b[s_] if flag[s_] else a[s_]) for s_ in a}       # 울리면 낮은 해(재풀이)를, 아니면 첫 풀이를 지킨다
    return {"stage": int(k), "resolve": rd, "by_first_class": table,
            "accurate_first": int(sum(accurate(a[s_]) for s_ in a)), "accurate_resolved": int(sum(accurate(b[s_]) for s_ in b)),
            "accurate_after_test": int(sum(accurate(keep[s_]) for s_ in keep))}


def e26e(data_dir: str = DATA_DIR) -> dict | None:
    out = {}
    for cond, tag in E26_TAGS.items():
        if cond not in ("thick", "clamped2", "near", "bilayer"):
            continue
        for k in range(2, 7):
            try:
                out[f"{cond}_s{k}"] = e26e_from_records(load(data_dir, tag), load(data_dir, f"{tag}_s{k}_ff"), k)
            except FileNotFoundError:
                pass
    return out or None


# E27 — 양단 고정 상자의 프로토콜 전파와 정사각 단면의 첫 통과(사전규정 2026-10-23-p5-e27-prespecification.md).
# 선택 규칙은 참조를 읽지 않는다: 시드마다 사다리가 안정한(꼬리 폭 ≤ 10⁻³) 후보 가운데 감사 Λ가 가장 낮은 것, 동률이면
# 후보 순서(first, ff, long, norm), 안정한 후보가 없으면 전 후보 가운데 감사 Λ가 가장 낮은 것.
E27_LADDER_TOL = 1e-3
E27_RULE = ("per seed, the candidate of lowest audit eigenvalue among those whose ladder tail spread is at most 1e-3 "
            "(ties broken by candidate order; if none is stable, lowest audit eigenvalue overall); no reference is read")
E27_SELECTION = "p5_e27_cc_selection.json"
E27_SQ_TAG = "e27_sq_mc"
E27_CC_ORIGINAL = ("e13b_clamped2", "exact", "resampled")
E27_CC_FIRST = "e26_cc_mc"


def _stable(rec, tol: float = E27_LADDER_TOL) -> bool:
    v = rec.get("ladder_tail_spread")
    return v is not None and np.isfinite(v) and v <= tol


def e27_select_stage(recs_by_path: dict, k: int, tol: float = E27_LADDER_TOL) -> dict:
    """단계 k의 후보 기록(경로 → 기록 목록)에서 시드마다 고른 경로. 경로의 순서가 동률을 깬다."""
    by = {p_: {int(r["seed"]): r for r in recs if r["mode"] == int(k)} for p_, recs in recs_by_path.items()}
    seeds = sorted(next(iter(by.values())))
    if not seeds or any(sorted(v) != seeds for v in by.values()):
        raise ValueError(f"단계 {k}: 후보마다 시드가 다르거나 비어 있다")
    order = list(recs_by_path)
    return {s_: min(order, key=lambda p_: (not _stable(by[p_][s_], tol), by[p_][s_]["Lam"], order.index(p_)))
            for s_ in seeds}


def e27_from_records(cands: dict, selection: dict, orig, first_pass, aux: dict | None = None,
                     tol: float = E26_RESOLVE_TOL, B: int = B_DEFAULT, seed: int = SEED, check_s3: bool = True,
                     higher: dict | None = None) -> dict:
    """cands = {단계: {경로: 기록}}, selection = {단계: {시드: 경로}}. 고른 풀이로 모드 1–6을 세고(시드 짝지은 대비 둘),
    단계마다 재풀이 검사를 읽는다: 첫 풀이 대비 고른 풀이의 Λ 감소가 허용오차를 넘으면 울린 것이다.

    `higher` = {단계: {시드: bool}}(E28 사전규정 §8)이면 그 단계의 첫 풀이가 더 높은 모드인지를 기록 대신 그것으로 정한다 —
    단계 k만 푼 실행의 기록은 참조 모드 1–k로만 짝지어 더 높은 모드를 식별하지 못한다. 정확 판정은 기록 그대로다."""
    modes = tuple(sorted(cands))
    sel_recs = []
    stages = {}
    for k in modes:
        by = {p_: {int(r["seed"]): r for r in recs if r["mode"] == k} for p_, recs in cands[k].items()}
        seeds = sorted(by["first"])
        pick = {int(s_): p_ for s_, p_ in selection[k].items()}
        if sorted(pick) != seeds:
            raise ValueError(f"단계 {k}의 선택과 기록의 시드가 다르다")
        chosen = {s_: by[pick[s_]][s_] for s_ in seeds}
        sel_recs += list(chosen.values())
        first = by["first"]
        drop = {s_: (first[s_]["Lam"] - chosen[s_]["Lam"]) / first[s_]["Lam"] for s_ in seeds}
        flag = {s_: drop[s_] > tol for s_ in seeds}
        hk = None if higher is None else higher.get(k)
        cls = {s_: ("accurate" if accurate(first[s_]) else
                    "higher" if (higher_mode(first[s_]) if hk is None else bool(hk[s_])) else "other")
               for s_ in seeds}
        stages[k] = {
            "accurate_each": {p_: int(sum(accurate(by[p_][s_]) for s_ in seeds)) for p_ in by},
            "repairs_of_higher": {p_: int(sum(1 for s_ in seeds if cls[s_] == "higher" and accurate(by[p_][s_])))
                                  for p_ in by if p_ != "first"},
            "selected_each": {p_: int(sum(1 for s_ in seeds if pick[s_] == p_)) for p_ in by},
            "accurate_selected": int(sum(accurate(chosen[s_]) for s_ in seeds)),
            "unstable_selected": int(sum(not _stable(chosen[s_]) for s_ in seeds)),
            "invalid_selected": int(sum(chosen[s_]["outcome"] == "invalid_quadrature" for s_ in seeds)),
            "first_class": {c: int(sum(1 for s_ in seeds if cls[s_] == c)) for c in ("accurate", "higher", "other")},
            "flagged": int(sum(flag.values())),
            "false_positive": int(sum(1 for s_ in seeds if cls[s_] == "accurate" and flag[s_])),
            "missed": int(sum(1 for s_ in seeds if cls[s_] == "higher" and accurate(chosen[s_]) and not flag[s_])),
            "drop": [float(drop[s_]) for s_ in seeds]}
    a = _seed_share(sel_recs, accurate, modes)
    out = {"accurate_by_mode": {m: int(sum(accurate(r) for r in sel_recs if r["mode"] == m)) for m in modes},
           "stages": stages}
    if check_s3 and 3 in cands and 2 in selection:
        # 점검(사전규정 §3.3) — 단계 2에서 첫 통과의 망을 고른 시드는 단계 3 `first`가 첫 통과의 단계 3을 되풀이한다.
        same = sorted(int(s_) for s_, p_ in selection[2].items() if p_ == "first")
        fp3 = {int(r["seed"]): r for r in first_pass if r["mode"] == 3}
        e3 = {int(r["seed"]): r for r in cands[3]["first"] if r["mode"] == 3}
        out["check_s3_first"] = {
            "n": len(same),
            "same_class": int(sum((fp3[s_]["outcome"], fp3[s_]["matched_mode"]) == (e3[s_]["outcome"], e3[s_]["matched_mode"])
                                  for s_ in same)),
            "max_rel_dlam": float(max((abs(fp3[s_]["Lam"] - e3[s_]["Lam"]) / fp3[s_]["Lam"] for s_ in same), default=0.0))}
    out["accurate_total"] = int(sum(out["accurate_by_mode"].values()))
    for key, ref in (("primary", orig), ("vs_first_pass", first_pass)):
        ci = paired_bootstrap(a, _seed_share(ref, accurate, modes), B, seed)
        out[key] = {"contrast": ci, "verdict": _rule(ci),
                    "reference_by_mode": {m: _n_acc(ref, (m,)) for m in modes}}
    if aux:
        out["aux"] = {}
        for k, paths in aux.items():
            first = {int(r["seed"]): r for r in cands[k]["first"] if r["mode"] == k}
            for p_, recs in paths.items():
                b = {int(r["seed"]): r for r in recs if r["mode"] == k}
                out["aux"][f"s{k}_{p_}"] = {
                    "accurate": int(sum(accurate(b[s_]) for s_ in b)),
                    "repairs_of_higher": int(sum(1 for s_ in b if higher_mode(first[s_]) and accurate(b[s_]))),
                    "flagged_vs_first": int(sum(1 for s_ in b
                                                if (first[s_]["Lam"] - b[s_]["Lam"]) / first[s_]["Lam"] > tol))}
    return out


def e27_square_from_records(sq, orig16, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    modes = (1, 2, 3, 4, 5, 6)
    ci = paired_bootstrap(_seed_share(sq, accurate, modes), _seed_share(orig16, accurate, modes), B, seed)
    return {"contrast": ci, "verdict": _rule(ci),
            "accurate_by_mode": {"first_pass": {m: _n_acc(sq, (m,)) for m in modes},
                                 "original": {m: _n_acc(orig16, (m,)) for m in modes}},
            "higher_by_mode": {m: int(sum(higher_mode(r) for r in sq if r["mode"] == m)) for m in modes}}


def e27(data_dir: str = DATA_DIR) -> dict | None:
    out = {}
    try:
        sq, o16 = load(data_dir, E27_SQ_TAG), load(data_dir, E16_TAGS["exact"])
        out["square"] = e27_square_from_records(sq, o16)
    except FileNotFoundError:
        pass
    p = os.path.join(data_dir, E27_SELECTION)
    if os.path.exists(p):
        sel = json.load(open(p, encoding="utf-8"))
        done = sorted(int(k) for k in sel["stages"])
        if done == list(range(1, 7)):
            from .run_p5_neural import e27_candidates, E27_AUX_PATHS, e27_tag
            cands = {k: {p_: load(data_dir, t) for p_, t in e27_candidates(k).items()} for k in done}
            for k in done:       # 선택 파일이 규칙대로인지 기록에서 다시 계산해 확인한다
                if {str(s_): v for s_, v in e27_select_stage(cands[k], k).items()} != sel["stages"][str(k)]["choice"]:
                    raise ValueError(f"단계 {k}의 선택 파일이 규칙과 다르다")
            aux = {}
            for k, paths in E27_AUX_PATHS.items():
                try:
                    aux[k] = {p_: load(data_dir, e27_tag(k, p_)) for p_ in paths}
                except FileNotFoundError:
                    pass
            tag, arm, cond = E27_CC_ORIGINAL
            out["clamped2"] = e27_from_records(
                cands, {k: sel["stages"][str(k)]["choice"] for k in done},
                load(data_dir, tag, arm=arm_key(arm), condition=cond), load(data_dir, E27_CC_FIRST), aux or None)
    return out or None


# E28 — end-to-end 프로토콜, 같은 예산의 재시작 비교, 근축퇴 전파, T-브래킷 음성 대조, 강형식 견인력 가중
# (사전규정 2026-10-23-p5-e28-prespecification.md). 선택 규칙은 E27과 같다(`e27_select_stage`).
E28_SELECTION = "p5_e28_cc_selection.json"
E28_NEAR_SELECTION = "p5_e28_nr_selection.json"
E28_T_STRICT = 0.06          # 명제 2가 허용오차 0.05의 부정확을 보증하는 감소 기준(보조 판독)


def dispositions(chosen: dict, first: dict, tol: float = E26_RESOLVE_TOL) -> dict:
    """참조를 읽지 않는 단계 상태(사전규정 §3.2) — 시드마다 계기(고른 해의 사다리 안정), 재풀이 상태(첫 풀이보다 허용오차를
    넘게 낮은 해가 있었는가), 처리(보고 가능 = 사다리 안정, 판정 보류 = 안정한 후보 없음). 참조 정확과 교차한다."""
    out = {"stable": 0, "improved": 0, "no_further_improvement": 0, "reportable": 0, "undetermined": 0,
           "reportable_accurate": 0, "reportable_inaccurate": 0, "undetermined_accurate": 0}
    for s_, r in chosen.items():
        st = _stable(r)
        imp = (first[s_]["Lam"] - r["Lam"]) / first[s_]["Lam"] > tol
        out["stable"] += st
        out["improved" if imp else "no_further_improvement"] += 1
        if st:
            out["reportable"] += 1
            out["reportable_accurate" if accurate(r) else "reportable_inaccurate"] += 1
        else:
            out["undetermined"] += 1
            out["undetermined_accurate"] += int(accurate(r))
    return out


def restart_pick(first_recs, restarts: dict, k: int) -> dict:
    """같은 예산의 단순 대안 — 재초기화 K회 가운데 감사 Λ가 가장 낮은 해(그리고 첫 풀이까지 넣은 최저). 참조를 읽지 않는다."""
    f = {int(r["seed"]): r for r in first_recs if r["mode"] == int(k)}
    rs = {name: {int(r["seed"]): r for r in recs if r["mode"] == int(k)} for name, recs in restarts.items()}
    seeds = sorted(f)
    if any(sorted(v) != seeds for v in rs.values()):
        raise ValueError("재시작마다 시드가 다르다")
    best = {s_: min((v[s_] for v in rs.values()), key=lambda r: r["Lam"]) for s_ in seeds}
    best_all = {s_: min([f[s_], best[s_]], key=lambda r: r["Lam"]) for s_ in seeds}
    return {"accurate_each": {n: int(sum(accurate(v[s_]) for s_ in seeds)) for n, v in rs.items()},
            "accurate_lowest_restart": int(sum(accurate(best[s_]) for s_ in seeds)),
            "accurate_lowest_with_first": int(sum(accurate(best_all[s_]) for s_ in seeds)),
            "flagged_vs_first": int(sum((f[s_]["Lam"] - best[s_]["Lam"]) / f[s_]["Lam"] > E26_RESOLVE_TOL for s_ in seeds)),
            "lowest_restart_records": [best[s_] for s_ in seeds]}


def e28_cc_from_records(cands: dict, selection: dict, orig, first_pass, restarts: dict | None = None,
                        B: int = B_DEFAULT, seed: int = SEED, higher: dict | None = None) -> dict:
    """E28a — E27과 같은 판독(주 판정 D_full·보조 D_first, 단계별 재풀이 검사)에 단계별 처리 상태와 같은 예산의 재시작 비교를 더한다.
    `higher`는 단계 2–6 첫 풀이의 더 높은 모드 분류(사전규정 §8, `p5_exploit.e28_stage_mac`)다."""
    out = e27_from_records(cands, selection, orig, first_pass, None, B=B, seed=seed, check_s3=False, higher=higher)
    out["dispositions"] = {}
    out["flags_strict"] = {}
    for k in sorted(cands):
        by = {p_: {int(r["seed"]): r for r in recs if r["mode"] == k} for p_, recs in cands[k].items()}
        pick = {int(s_): p_ for s_, p_ in selection[k].items()}
        chosen = {s_: by[pick[s_]][s_] for s_ in by["first"]}
        out["dispositions"][k] = dispositions(chosen, by["first"])
        out["flags_strict"][k] = int(sum((by["first"][s_]["Lam"] - chosen[s_]["Lam"]) / by["first"][s_]["Lam"] > E28_T_STRICT
                                         for s_ in chosen))
    if restarts:
        out["restarts"] = {}
        for k, rr in restarts.items():
            rp = restart_pick(cands[k]["first"], rr, k)
            longk = cands[k].get("long")
            if longk is not None:
                a = _seed_share(rp.pop("lowest_restart_records"), accurate, (k,))
                b = _seed_share(longk, accurate, (k,))
                ci = paired_bootstrap(a, b, B, seed)
                rp["vs_long"] = {"contrast": ci, "verdict": _rule(ci), "accurate_long": _n_acc(longk, (k,))}
            else:
                rp.pop("lowest_restart_records")
            out["restarts"][k] = rp
    return out


def e28_lb_from_records(first, restarts: dict, fourfold, fourier, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """E28b — L-브래킷 단계 6. 주 판정: acc₆(재초기화 4회 중 최저 Λ) − acc₆(4배 일정, 같은 예산). 보조: − acc₆(Fourier 재풀이)."""
    rp = restart_pick(first, restarts, 6)
    a = _seed_share(rp.pop("lowest_restart_records"), accurate, (6,))
    out = {**rp}
    for key, ref in (("vs_fourfold", fourfold), ("vs_fourier", fourier)):
        ci = paired_bootstrap(a, _seed_share(ref, accurate, (6,)), B, seed)
        out[key] = {"contrast": ci, "verdict": _rule(ci), "accurate_reference": _n_acc(ref, (6,))}
    out["accurate_first"] = _n_acc(first, (6,))
    return out


def e28_near_from_records(cands: dict, selection: dict, first_pass) -> dict:
    """E28c — 근축퇴 단면의 단계 5 선택을 단계 6으로 전파. 판독: 단계 6의 경로별·선택 정확 수, 첫 통과 단계 6 대비."""
    out = {}
    for k in (5, 6):
        by = {p_: {int(r["seed"]): r for r in recs if r["mode"] == k} for p_, recs in cands[k].items()}
        pick = {int(s_): p_ for s_, p_ in selection[k].items()}
        chosen = {s_: by[pick[s_]][s_] for s_ in by["first"]}
        out[k] = {"accurate_each": {p_: int(sum(accurate(v[s_]) for s_ in v)) for p_, v in by.items()},
                  "accurate_selected": int(sum(accurate(r) for r in chosen.values())),
                  "selected_each": {p_: int(sum(1 for s_ in pick if pick[s_] == p_)) for p_ in by},
                  "dispositions": dispositions(chosen, by["first"])}
    out["first_pass_stage6"] = _n_acc(first_pass, (6,))
    return out


def e28_tb_from_records(first, resolved: dict) -> dict:
    """E28d — T-브래킷 음성 대조. 단계 k마다 첫 통과(E26d)와 Fourier 재풀이를 시드끼리 짝지어 거짓 양성(정확한 첫 풀이가
    울린 수)과 고른 해의 정확을 센다."""
    out, fp, n_acc_first, n_sel = {}, 0, 0, 0
    for k, recs in sorted(resolved.items()):
        a = {int(r["seed"]): r for r in first if r["mode"] == k}
        b = {int(r["seed"]): r for r in recs if r["mode"] == k}
        if sorted(a) != sorted(b):
            raise ValueError(f"단계 {k}: 시드가 다르다")
        flag = {s_: (a[s_]["Lam"] - b[s_]["Lam"]) / a[s_]["Lam"] > E26_RESOLVE_TOL for s_ in a}
        pick = {s_: min([a[s_], b[s_]], key=lambda r: (not _stable(r), r["Lam"])) for s_ in a}
        st = {"accurate_first": int(sum(accurate(a[s_]) for s_ in a)),
              "accurate_resolved": int(sum(accurate(b[s_]) for s_ in b)),
              "flagged": int(sum(flag.values())),
              "false_positive": int(sum(1 for s_ in a if accurate(a[s_]) and flag[s_])),
              "accurate_selected": int(sum(accurate(r) for r in pick.values())),
              "max_drop": float(max((a[s_]["Lam"] - b[s_]["Lam"]) / a[s_]["Lam"] for s_ in a))}
        out[k] = st
        fp += st["false_positive"]; n_acc_first += st["accurate_first"]; n_sel += st["accurate_selected"]
    out["total"] = {"false_positive": fp, "accurate_first": n_acc_first, "accurate_selected": n_sel}
    return out


def e28_pinn_from_records(w10, w1, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """E28e — 강형식 견인력 가중 10 대 1(E24 thick). 주 판정: 셀마다 log10(|Λ − λ|/Λ)의 시드 평균 차(가중 10 − 1).
    보조: acc₁₋₆ 차, 모드 2–6에서 더 낮은 모드에 짝지어진 셀의 비율 차."""
    def mism(recs):
        by = {}
        for r in recs:
            by.setdefault(int(r["seed"]), []).append(np.log10(max(abs(r["Lam"] - r["lam_param"]) / abs(r["Lam"]), 1e-16)))
        return np.array([np.mean(by[s_]) for s_ in sorted(by)])
    out = {}
    ci = paired_bootstrap(mism(w10), mism(w1), B, seed)
    out["mismatch"] = {"contrast": ci, "verdict": _rule(ci),
                       "median_rel": {"w10": float(np.median([abs(r["Lam"] - r["lam_param"]) / abs(r["Lam"]) for r in w10])),
                                      "w1": float(np.median([abs(r["Lam"] - r["lam_param"]) / abs(r["Lam"]) for r in w1]))}}
    modes = (1, 2, 3, 4, 5, 6)
    ci = paired_bootstrap(_seed_share(w10, accurate, modes), _seed_share(w1, accurate, modes), B, seed)
    out["accuracy"] = {"contrast": ci, "verdict": _rule(ci), "w10": _n_acc(w10, modes), "w1": _n_acc(w1, modes)}
    lower = lambda r: r["matched_mode"] < r["mode"] and r["mac"] >= 0.9
    ci = paired_bootstrap(_seed_share(w10, lower, (2, 3, 4, 5, 6)), _seed_share(w1, lower, (2, 3, 4, 5, 6)), B, seed)
    out["lower_mode"] = {"contrast": ci, "verdict": _rule(ci)}
    out["shape_match"] = {"w10": int(sum(r["mac"] >= 0.9 for r in w10)), "w1": int(sum(r["mac"] >= 0.9 for r in w1))}
    return out


def e28_higher_from_stage_mac(data_dir: str, stages) -> tuple:
    """사전규정 §8 — 단계 2–6 첫 풀이의 더 높은 모드 분류를 재채점 파일(`p5_exploit_e28_cc_s<k>_first_stage_mac.csv`)에서
    읽는다. 파일이 없거나 재채점이 실행의 채점을 되풀이하지 않은 단계는 넣지 않고(그 단계는 기록으로 분류된다) 출처에 적는다."""
    hi, src = {}, {}
    for k in stages:
        if int(k) == 1:
            src[1] = "records (first pass, modes 1-6)"
            continue
        t = f"e28_cc_s{int(k)}_first"
        pm = os.path.join(data_dir, f"manifest_p5_exploit_{t}_stage_mac.json")
        pc = os.path.join(data_dir, f"p5_exploit_{t}_stage_mac.csv")
        if not (os.path.exists(pm) and os.path.exists(pc)):
            src[int(k)] = "records (modes 1-k only; stage MAC missing)"
            continue
        if not json.load(open(pm, encoding="utf-8")).get("reproduces_record"):
            src[int(k)] = "records (stage MAC did not reproduce the run)"
            continue
        with open(pc, encoding="utf-8") as f:
            hi[int(k)] = {int(r["seed"]): r["higher"] in ("True", "true", "1") for r in csv.DictReader(f)}
        src[int(k)] = "stage MAC against FEM modes 1-8"
    return hi, src


def e28(data_dir: str = DATA_DIR) -> dict | None:
    out = {}
    def _maybe(tag):
        try:
            return load(data_dir, tag)
        except FileNotFoundError:
            return None
    from .run_p5_neural import (e28_candidates, e28_aux_paths, e28_tag, e28_near_candidates, E28_LB_REINIT,
                                E21H_TAGS, E26_TAGS as RUN_E26_TAGS)
    p = os.path.join(data_dir, E28_SELECTION)
    if os.path.exists(p):
        sel = json.load(open(p, encoding="utf-8"))
        done = sorted(int(k) for k in sel["stages"])
        if done == list(range(1, 7)):
            cands = {k: {p_: load(data_dir, t) for p_, t in e28_candidates(k).items()} for k in done}
            for k in done:
                if {str(s_): v for s_, v in e27_select_stage(cands[k], k).items()} != sel["stages"][str(k)]["choice"]:
                    raise ValueError(f"E28a 단계 {k}의 선택 파일이 규칙과 다르다")
            restarts = {}
            for k in (2, 3):
                rr = {pp: _maybe(e28_tag(k, pp)) for pp in e28_aux_paths(k)}
                if rr and all(v is not None for v in rr.values()):
                    restarts[k] = rr
            tag, arm, cond = E27_CC_ORIGINAL
            hi, src = e28_higher_from_stage_mac(data_dir, done)
            out["clamped2"] = e28_cc_from_records(cands, {k: sel["stages"][str(k)]["choice"] for k in done},
                                                  load(data_dir, tag, arm=arm_key(arm), condition=cond),
                                                  load(data_dir, E27_CC_FIRST), restarts or None, higher=hi or None)
            out["clamped2"]["higher_source"] = src
    rr = {f"reinit{r}": _maybe(f"e28_lb_s6_reinit{r}") for r in E28_LB_REINIT}
    if all(v is not None for v in rr.values()):
        out["lbracket_restart"] = e28_lb_from_records(load(data_dir, "e21lb_exact_mc1"), rr,
                                                      load(data_dir, E21H_TAGS["sched_raw"]),
                                                      load(data_dir, RUN_E26_TAGS["fourier_raw"]))
    pn = os.path.join(data_dir, E28_NEAR_SELECTION)
    if os.path.exists(pn):
        seln = json.load(open(pn, encoding="utf-8"))
        if sorted(int(k) for k in seln["stages"]) == list(range(1, 7)):
            candsn = {k: {p_: load(data_dir, t) for p_, t in e28_near_candidates(k).items()} for k in (5, 6)}
            out["near"] = e28_near_from_records(candsn, {k: seln["stages"][str(k)]["choice"] for k in (5, 6)},
                                                load(data_dir, RUN_E26_TAGS["near"]))
    tb = {k: _maybe(f"e28_tb_s{k}_ff") for k in range(1, 7)}
    tb = {k: v for k, v in tb.items() if v is not None}
    if tb:
        out["tbracket_control"] = e28_tb_from_records(load(data_dir, RUN_E26_TAGS["tbracket_raw"]), tb)
    w10 = _maybe("e28_pinn_thick_w10")
    if w10 is not None:
        out["pinn_traction"] = e28_pinn_from_records(w10, load(data_dir, "e24_pinn_thick"))
    return out or None


# ── E29 (사전규정 2026-10-30-p5-e29-prespecification.md) — 한 단계 더 ───────────────────────────────────────
# 첫 통과의 저장 모드 1–6에서 푼 단계 7을 순서 검사의 경쟁자로 더한다. 단계 7이 사다리를 통과하고 감사 Λ가 단계 6의
# (1 − t)배보다 작으면 단계 6을 울린다(참조를 읽지 않는다). 채점은 실행 뒤에 교차한다: 단계 6·7의 출력은 FEM 모드
# 1–10 재채점(`p5_exploit.e29_stage_mac`)으로 식별하고, 단계 1–5는 기록의 짝을 쓴다.
E29_FIRST = {"lbracket": "e21lb_exact_mc1", "near": "e26_near_mc"}
E29_ORDER_T = 0.05
E29_PREDICTION = {"lbracket": {"flags_stage6_at_least": 14}, "near": {"false_positives": 0, "flags_on_inaccurate": 0}}


def e29_ident(data_dir: str, tag: str) -> tuple:
    """FEM 재채점 파일의 {시드: (짝 모드, MAC)}와 출처. 파일이 없거나 재채점이 실행을 되풀이하지 않으면 빈 사전이다
    (그때는 기록의 짝 — 참조 모드 1–k — 을 쓴다)."""
    pm = os.path.join(data_dir, f"manifest_p5_exploit_{tag}_stage_mac.json")
    pc = os.path.join(data_dir, f"p5_exploit_{tag}_stage_mac.csv")
    if not (os.path.exists(pm) and os.path.exists(pc)):
        return {}, "records (stage MAC missing)"
    man = json.load(open(pm, encoding="utf-8"))
    if not man.get("reproduces_record"):
        return {}, "records (stage MAC did not reproduce the run)"
    with open(pc, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ({int(r["seed"]): (int(r["mode_ref"]), float(r["mac_ref"])) for r in rows},
            f"stage MAC against FEM modes 1-{int(man['n_ref'])}")


def _reordered_accurate(outs: list, lamref: dict, n_report: int = 6) -> int:
    """outs = [(Λ, 짝 모드, MAC, 군집 또는 None)]. 감사 Λ 순으로 세워 순위 1–n_report를 채점한다(`p5_posthoc.order_check`의
    정렬 판독과 같은 규칙: 짝이 순위이거나 순위가 군집 안, MAC ≥ 0.9, 그 순위의 참조 고유값에 대한 e_λ ≤ 0.05)."""
    n = 0
    for rank, (lam, mm, mac, cl) in zip(range(1, n_report + 1), sorted(outs, key=lambda o: o[0])):
        shape = mm == rank or rank in (cl or ())
        n += bool(shape and mac >= 0.9 and abs(lam - lamref[rank]) / lamref[rank] <= 0.05)
    return n


def e29_from_records(first: list, stage7: list, ident7: dict | None = None, ident6: dict | None = None,
                     t: float = E29_ORDER_T) -> dict:
    """first = 첫 통과 기록(단계 1–6), stage7 = 단계 7 기록, ident7·ident6 = {시드: (짝 모드, MAC)}(FEM 재채점, 없으면 기록).
    시드마다 단계 6의 울림과 그 정확(기록), 단계 7의 식별, 단계 7이 새로 울리는 앞 단계(단계 1–6 사다리로는 울리지 않던
    것), 단계 7을 더한 정렬 판독(순위 1–6)을 낸다."""
    lad = {}
    for r in first:
        lad.setdefault(int(r["seed"]), {})[int(r["mode"])] = r
    s7 = {int(r["seed"]): r for r in stage7 if int(r["mode"]) == 7}
    seeds = sorted(lad)
    if sorted(s7) != seeds or any(sorted(lad[s_]) != list(range(1, 7)) for s_ in seeds):
        raise ValueError("E29: 첫 통과(단계 1–6)와 단계 7의 시드가 맞지 않는다")
    ident7, ident6 = ident7 or {}, ident6 or {}
    rows, modes7 = [], {}
    for s_ in seeds:
        L, r7 = lad[s_], s7[s_]
        lam = {k: float(L[k]["Lam"]) for k in range(1, 7)}
        lam7 = float(r7["Lam"])
        stable7 = _stable(r7)
        flag6 = bool(stable7 and lam7 < (1 - t) * lam[6])
        old = {j for j in range(1, 6) if any(_stable(L[i]) and lam[i] < (1 - t) * lam[j] for i in range(j + 1, 7))}
        new = [j for j in range(1, 6) if stable7 and lam7 < (1 - t) * lam[j] and j not in old]
        m7, mac7 = ident7.get(s_, (int(r7["matched_mode"]), float(r7["mac"])))
        m6, mac6 = ident6.get(s_, (int(L[6]["matched_mode"]), float(L[6]["mac"])))
        key = m7 if mac7 >= 0.9 else ("mixture" if mac7 >= 0.5 else "unidentified")
        modes7[str(key)] = modes7.get(str(key), 0) + 1
        lamref = {k: float(L[k]["Lam_ref"]) for k in range(1, 7)}
        outs = [(lam[k], int(L[k]["matched_mode"]), float(L[k]["mac"]), tuple(L[k].get("cluster") or ()) or None)
                for k in range(1, 6)] + [(lam[6], m6, mac6, None)]
        rows.append({"seed": s_, "lam6": lam[6], "lam7": lam7, "drop": 1 - lam7 / lam[6], "stable7": bool(stable7),
                     "flag6": flag6, "accurate6": bool(accurate(L[6])), "mode6": m6, "mac6": mac6, "mode7": m7,
                     "mac7": mac7, "new_earlier_flags": new,
                     "reordered6": _reordered_accurate(outs, lamref),
                     "reordered7": _reordered_accurate(outs + [(lam7, m7, mac7, None)], lamref)})
    acc6 = [r for r in rows if r["accurate6"]]
    return {"seeds": len(rows), "stage7_stable": sum(r["stable7"] for r in rows),
            "flags_stage6": sum(r["flag6"] for r in rows),
            "stage6_accurate": len(acc6), "false_positives": sum(r["flag6"] for r in acc6),
            "flags_on_inaccurate": sum(r["flag6"] for r in rows if not r["accurate6"]),
            "new_earlier_flags": sum(len(r["new_earlier_flags"]) for r in rows),
            "stage7_modes": dict(sorted(modes7.items())),
            "reordered_accurate_six_outputs": sum(r["reordered6"] for r in rows),
            "reordered_accurate_seven_outputs": sum(r["reordered7"] for r in rows),
            "flagged_min_drop": min((r["drop"] for r in rows if r["flag6"]), default=None),
            "accurate_max_drop": max((r["drop"] for r in acc6), default=None),
            "per_seed": rows}


def e29(data_dir: str = DATA_DIR) -> dict | None:
    from .run_p5_neural import E29_TAGS
    out = {}
    for s_, tag in E29_TAGS.items():
        if not os.path.exists(os.path.join(data_dir, f"manifest_p5_neural_{tag}.json")):
            continue
        id7, src7 = e29_ident(data_dir, tag)
        id6, src6 = e29_ident(data_dir, f"{E29_FIRST[s_]}_s6")
        r = e29_from_records(load(data_dir, E29_FIRST[s_]), load(data_dir, tag), id7, id6)
        pred = E29_PREDICTION[s_]
        if s_ == "lbracket":
            r["verdict"] = {"prediction": pred, "hit": r["flags_stage6"] >= pred["flags_stage6_at_least"]}
        else:
            r["verdict"] = {"prediction": pred, "hit": (r["false_positives"] == pred["false_positives"]
                                                        and r["flags_on_inaccurate"] == pred["flags_on_inaccurate"])}
        r["ident_source"] = {"stage6": src6, "stage7": src7}
        out[s_] = r
    return out or None


# ── E30 (사전규정 2026-10-30-p5-e30-prespecification.md) — 단계적 배분 ────────────────────────────────────
# 1회차(두 경로 + 단계 7) → 순서 검사 → 울린 시드만 2회차(j*에서 네 경로, 그 뒤 두 경로). 판정은 최종 사슬의 단계 1–6 정확과
# 시드마다 센 비용(그 시드의 절차가 부른 실행의 GPU-시간 합 / 첫 통과의 GPU-시간)의 시드 평균이다.
E30_T = 0.05
E30_RULE = (E27_RULE + "; restricted per seed to its eligible paths (round 2 adds the Fourier and held-rate paths only "
            "at the seed's flagged stage j*)")
E30_ORDER_RULE = ("stage j (1-6) is flagged when a later stage i (up to the look-ahead stage 7) of the seed's chain has "
                  "ladder tail spread at most 1e-3 and audit eigenvalue below (1 - 0.05) times that of stage j; j* is the "
                  "earliest flagged stage; no reference is read")
E30_FIRST = {"cc": E27_CC_FIRST, "nr": "e26_near_mc"}
E30_GO = {"accurate_at_least": 118, "mean_cost_ratio_at_most": 2.5}
E30_PREDICTION = {"cc": {"go": True, "d_full": "rule2", "false_positives": 0},
                  "nr": {"d_first": "rule1", "false_positives": 0}}


def e30_select_stage(recs_by_path: dict, k: int, eligible: dict, tol: float = E27_LADDER_TOL) -> dict:
    """E27 규칙을 시드마다 고를 수 있는 경로(`eligible` = {시드: 경로들})로 좁힌다. 경로의 순서가 동률을 깬다."""
    by = {p_: {int(r["seed"]): r for r in recs if r["mode"] == int(k)} for p_, recs in recs_by_path.items()}
    order = list(recs_by_path)
    out = {}
    for s_, ok in sorted(eligible.items()):
        ps = [p_ for p_ in order if p_ in ok]
        if not ps:
            raise ValueError(f"단계 {k} 시드 {s_}: 고를 경로가 없다")
        out[int(s_)] = min(ps, key=lambda p_: (not _stable(by[p_][int(s_)], tol), by[p_][int(s_)]["Lam"], order.index(p_)))
    return out


def e30_order_flags(chain: dict, t: float = E30_T, last_reported: int = 6) -> list:
    """시드 하나의 사슬 {단계: 기록}에서 순서 검사가 울린 단계(1–last_reported). 참조를 읽지 않는다."""
    ks = sorted(chain)
    return [j for j in ks if j <= last_reported and any(
        _stable(chain[i]) and chain[i]["Lam"] < (1 - t) * chain[j]["Lam"] for i in ks if i > j)]


def e30_from_records(cands1: dict, sel1: dict, plan: dict, cands2: dict | None, sel2: dict | None, cost: dict,
                     tags1: dict, tags2: dict | None, first_pass: list, first_cost: float, compare: dict | None = None,
                     new_tags=None, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """cands = {단계: {경로: 기록 목록}}, sel = {단계: {시드: 경로}}, tags = {(단계, 경로): 태그}, cost = {태그: GPU-시간}.
    `compare` = {이름: 단계 1–6 기록 목록}(시드 짝지은 비교 사슬). `new_tags`는 이 실험에서 새로 돈 실행(앙상블 GPU-시간
    합계용, 없으면 모든 태그). 계획의 울림은 1회차 사슬에서 다시 계산해 확인한다."""
    by1 = {k: {p_: {int(r["seed"]): r for r in recs if r["mode"] == k} for p_, recs in v.items()} for k, v in cands1.items()}
    by2 = ({k: {p_: {int(r["seed"]): r for r in recs if r["mode"] == k} for p_, recs in v.items()}
            for k, v in cands2.items()} if cands2 else {})
    stages = sorted(by1)
    seeds = sorted(by1[1]["first"])
    flagged = {int(s_): int(j) for s_, j in plan["flagged"].items()}
    if flagged and not by2:
        raise ValueError("계획에 울린 시드가 있는데 2회차 기록이 없다")
    r1 = {s_: {k: by1[k][sel1[k][s_]][s_] for k in stages} for s_ in seeds}
    for s_ in seeds:
        if e30_order_flags(r1[s_]) != list(plan["flags"][str(s_)]):
            raise ValueError(f"시드 {s_}: 계획의 울림이 1회차 사슬과 다르다")
    final, reproduce = {}, {"compared": 0, "max_rel_dlam": 0.0, "mismatch": 0}
    for s_ in seeds:
        j = flagged.get(s_)
        final[s_] = {k: (by2[k][sel2[k][s_]][s_] if j is not None and k >= j else r1[s_][k]) for k in stages}
        for k in by2:                                   # 2회차에서 아직 j*에 이르지 않은 시드는 1회차를 되풀이해야 한다
            if j is not None and k >= j:
                continue
            a, b = by2[k][sel1[k][s_]][s_], r1[s_][k]
            rel = abs(a["Lam"] - b["Lam"]) / abs(b["Lam"])
            reproduce["compared"] += 1
            reproduce["max_rel_dlam"] = max(reproduce["max_rel_dlam"], float(rel))
            reproduce["mismatch"] += bool(rel > 0 or a["mac"] != b["mac"])
    rep = range(1, 7)
    acc = lambda ch: {k: int(sum(accurate(ch[s_][k]) for s_ in seeds)) for k in rep}
    flags_final = {s_: e30_order_flags(final[s_]) for s_ in seeds}
    fp = lambda ch, fl: int(sum(accurate(ch[s_][j]) for s_ in seeds for j in fl[s_]))
    flags_r1 = {s_: list(plan["flags"][str(s_)]) for s_ in seeds}
    # 비용 — 그 시드의 절차가 부른 실행만. 2회차의 j*에서 first·norm은 1회차와 같은 계산이라 다시 세지 않는다.
    c1 = sum(cost[tags1[(k, p_)]] for k in stages for p_ in by1[k])
    per_seed = {}
    for s_ in seeds:
        c = c1
        j = flagged.get(s_)
        if j is not None:
            c += sum(cost[tags2[(j, p_)]] for p_ in by2[j] if p_ not in by1[j])
            c += sum(cost[tags2[(k, p_)]] for k in stages if k > j for p_ in by2[k])
        per_seed[s_] = c / first_cost
    ratio = np.array([per_seed[s_] for s_ in seeds])
    ens = sum(cost[t] for t in (set(tags1.values()) | set((tags2 or {}).values()) if new_tags is None else new_tags))
    fin_recs = [final[s_][k] for s_ in seeds for k in rep]
    out = {"accurate_by_stage": acc(final), "accurate": int(sum(acc(final).values())),
           "round1_accurate_by_stage": acc(r1), "round1_accurate": int(sum(acc(r1).values())),
           "first_pass_accurate": _n_acc(first_pass, tuple(rep)),
           "flagged_seeds": {str(s_): j for s_, j in sorted(flagged.items())},
           "flags_round1": int(sum(len(v) for v in flags_r1.values())), "false_positives_round1": fp(r1, flags_r1),
           "flags_final": int(sum(len(v) for v in flags_final.values())), "false_positives_final": fp(final, flags_final),
           "flags_final_by_seed": {str(s_): v for s_, v in flags_final.items() if v},
           "round2": {str(s_): {"stage": j, "chosen": sel2[j][s_], "accurate_before": bool(accurate(r1[s_][j])),
                                "accurate_after": bool(accurate(final[s_][j])),
                                "drop": float(1 - final[s_][j]["Lam"] / r1[s_][j]["Lam"])}
                      for s_, j in sorted(flagged.items())},
           "round2_reproduces_round1": reproduce,
           "selected_each_round1": {k: {p_: int(sum(sel1[k][s_] == p_ for s_ in seeds)) for p_ in by1[k]} for k in stages},
           "cost": {"first_pass_gpu_h": float(first_cost), "round1_gpu_h_per_seed": float(c1),
                    "mean_ratio": float(ratio.mean()), "median_ratio": float(np.median(ratio)),
                    "max_ratio": float(ratio.max()), "per_seed_ratio": {str(s_): float(per_seed[s_]) for s_ in seeds},
                    "ensemble_gpu_h": float(ens)}}
    out["d_first"] = {"contrast": (ci := paired_bootstrap(_seed_share(fin_recs, accurate, tuple(rep)),
                                                          _seed_share(first_pass, accurate, tuple(rep)), B, seed)),
                      "verdict": _rule(ci)}
    for name, recs in (compare or {}).items():
        ci = paired_bootstrap(_seed_share(fin_recs, accurate, tuple(rep)), _seed_share(recs, accurate, tuple(rep)), B, seed)
        out[f"d_{name}"] = {"contrast": ci, "verdict": _rule(ci), "accurate_reference": _n_acc(recs, tuple(rep))}
    return out


def _gpu_hours(data_dir: str, tag: str) -> float:
    import datetime as dt
    m = json.load(open(os.path.join(data_dir, f"manifest_p5_neural_{tag}.json"), encoding="utf-8"))
    a = dt.datetime.fromisoformat(m["launch"]["timestamp_utc"])
    return (dt.datetime.fromisoformat(m["timestamp_utc"]) - a).total_seconds() / 3600


def _staged_judgment(data_dir: str, s: str, first_tag: str, compare: dict | None = None) -> dict | None:
    """단계적 배분 구조 하나의 판정량(`e30_from_records`). 1회차 단계 1–7과 계획(울림이 있으면 2회차 단계 j_min–7까지)이 끝났을
    때만 낸다. 단계 1의 first는 첫 통과를 다시 쓴 실행이라 비용은 같은 구조 단계 1의 norm 실행 시간으로 센다(같은 크기의 단계
    하나). 다시 쓴 실행(첫 통과 포함)은 앙상블 GPU-시간 합계에서 뺀다. E30과 E31이 같이 쓴다."""
    from .run_p5_neural import E30_LOOKAHEAD, e30_tag, e30_paths, e30_selection_name, e30_plan_name, _staged_spec
    get = lambda name: (json.load(open(os.path.join(data_dir, name), encoding="utf-8"))
                        if os.path.exists(os.path.join(data_dir, name)) else None)
    sel1, sel2, plan = get(e30_selection_name(s, 1)), get(e30_selection_name(s, 2)), get(e30_plan_name(s))
    ks = range(1, E30_LOOKAHEAD + 1)
    if plan is None or sel1 is None or any(str(k) not in sel1["stages"] for k in ks):
        return None
    r2 = range(int(plan["j_min"]), E30_LOOKAHEAD + 1) if plan["flagged"] else ()
    if r2 and (sel2 is None or any(str(k) not in sel2["stages"] for k in r2)):
        return None
    tags1 = {(k, p_): e30_tag(s, 1, k, p_) for k in ks for p_ in e30_paths(1, k)}
    tags2 = {(k, p_): e30_tag(s, 2, k, p_) for k in r2 for p_ in e30_paths(2, k, plan)} or None
    cands1 = {k: {p_: load(data_dir, tags1[(k, p_)]) for p_ in e30_paths(1, k)} for k in ks}
    cands2 = {k: {p_: load(data_dir, tags2[(k, p_)]) for p_ in e30_paths(2, k, plan)} for k in r2} or None
    sel = lambda f, rng: {k: {int(s_): v for s_, v in f["stages"][str(k)]["choice"].items()} for k in rng}
    alltags = set(tags1.values()) | set((tags2 or {}).values())
    first1 = e30_tag(s, 1, 1, "first")
    cost = {t: _gpu_hours(data_dir, t) for t in alltags if t != first1}
    cost[first1] = cost[tags1[(1, "norm")]]
    reused = set(_staged_spec(s)["reuse"].values())
    return e30_from_records(cands1, sel(sel1, ks), plan, cands2, sel(sel2, r2) if r2 else None, cost, tags1, tags2,
                            load(data_dir, first_tag), _gpu_hours(data_dir, first_tag), compare,
                            new_tags={t for t in alltags if t not in reused})


def e30(data_dir: str = DATA_DIR) -> dict | None:
    """구조마다 1회차 단계 1–7과 계획(울림이 있으면 2회차 단계 j_min–7까지)이 끝났을 때만 판정한다(`_staged_judgment`)."""
    from .run_p5_neural import E30_STRUCTURES, e28_candidates, E28_SELECTION
    out = {}
    for s in E30_STRUCTURES:
        compare = None
        if s == "cc" and os.path.exists(os.path.join(data_dir, E28_SELECTION)):
            s28 = json.load(open(os.path.join(data_dir, E28_SELECTION), encoding="utf-8"))
            recs = []
            for k in range(1, 7):
                c = {p_: {int(r["seed"]): r for r in load(data_dir, t) if r["mode"] == k} for p_, t in e28_candidates(k).items()}
                recs += [c[p_][int(s_)] for s_, p_ in s28["stages"][str(k)]["choice"].items()]
            compare = {"full": recs}
        r = _staged_judgment(data_dir, s, E30_FIRST[s], compare)
        if r is None:
            continue
        pred = E30_PREDICTION[s]
        if s == "cc":
            go = r["accurate"] >= E30_GO["accurate_at_least"] and r["cost"]["mean_ratio"] <= E30_GO["mean_cost_ratio_at_most"]
            r["verdict"] = {"go": bool(go), "go_rule": E30_GO, "prediction": pred,
                            "hit": {"go": go == pred["go"], "d_full": r["d_full"]["verdict"] == pred["d_full"],
                                    "false_positives": r["false_positives_round1"] == r["false_positives_final"] == 0}}
        else:
            r["verdict"] = {"prediction": pred,
                            "hit": {"d_first": r["d_first"]["verdict"] == pred["d_first"],
                                    "false_positives": r["false_positives_round1"] == r["false_positives_final"] == 0}}
        out[s] = r
    return out or None



# ── E30 보조 판독(보완 사전규정 2026-10-30-p5-review-addendum-prespecification.md §4.2–4.5) ──────────────────────
# 등록 판정(`e30`)은 그대로 두고, 검토 보고서(2026-10-30) §9.4–9.5의 양을 따로 낸다: 비열등(Δ = 0.05, D_full의 등록 구간 아래 끝),
# 완결성(Z = 단계 1–6이 모두 정확한 시드 수, 시드별 정확 비율 Y_s), 판정 품질(검사 가능/불가능, 거짓 양성, 미검출, 잠정·위쪽 잠정),
# 최종 부정확 셀의 유형(FEM 식별, `p5_exploit.b2`의 E30 판독), E30 출력의 Rayleigh–Ritz. 판정이 아니다.
E30_NI_DELTA = 0.05


def sorted_seed_accurate(first: list) -> dict:
    """첫 통과 기록의 정렬 판독을 시드마다 — 감사 Λ 순으로 세운 순위 1–6의 정확 수(`p5_posthoc.order_check`와 같은 규칙)."""
    lad = {}
    for r in first:
        if int(r["mode"]) <= 6:
            lad.setdefault(int(r["seed"]), {})[int(r["mode"])] = r
    out = {}
    for s_, L in sorted(lad.items()):
        lamref = {k: float(L[k]["Lam_ref"]) for k in L}
        n = 0
        for rank, k in zip(sorted(L), sorted(L, key=lambda k_: L[k_]["Lam"])):
            r = L[k]
            members = tuple(r.get("cluster") or (r["matched_mode"],))
            shape = r["outcome"] != "invalid_quadrature" and (r["matched_mode"] == rank or rank in members)
            n += bool(shape and r["mac"] >= 0.9 and abs(r["Lam"] - lamref[rank]) / lamref[rank] <= 0.05)
        out[s_] = n
    return out


def e30_chain_quality(chains: dict, t: float = E30_T) -> dict:
    """chains = {시드: {단계 1–7: 기록}}. 단계 1–6마다 정확(참조), 사다리 안정, 검사 가능(뒤 단계 i ≤ 7 가운데 사다리를 통과한
    것이 있다), 울림(`e30_order_flags`), 잠정(고른 해가 사다리를 통과하지 못했다)과 위쪽 잠정(앞 단계에 잠정이 있다)을 센다."""
    c = {k: 0 for k in ("accurate", "accurate_stable", "testable_accurate", "testable_accurate_flagged",
                        "testable_inaccurate", "testable_inaccurate_flagged", "untestable_accurate", "untestable_inaccurate",
                        "provisional", "upstream_provisional", "all_accurate_seeds")}
    share = []
    for s_, ch in sorted(chains.items()):
        flags = e30_order_flags(ch, t=t)
        prov, n = False, 0
        for j in range(1, 7):
            r = ch[j]
            a, stb = bool(accurate(r)), _stable(r)
            testable = any(_stable(ch[i]) for i in ch if i > j)
            key = ("testable_" if testable else "untestable_") + ("accurate" if a else "inaccurate")
            c[key] += 1
            if testable:
                c[key + "_flagged"] += j in flags
            c["accurate"] += a
            c["accurate_stable"] += a and stb
            c["upstream_provisional"] += prov
            if not stb:
                c["provisional"] += 1
                prov = True
            n += a
        share.append(n)
        c["all_accurate_seeds"] += n == 6
    c["seed_accurate_counts"] = {f"{k}/6": int(sum(v == k for v in share)) for k in range(7)}
    return c


def _ident_class(k: int, ident: dict) -> str:
    """부정확한 단계 k의 출력을 FEM 식별로 나눈다 — 고유값(형상은 맞다), 더 높은 모드, 낮은 모드, 섞임, 미식별."""
    mac, cl = float(ident["best_mac"]), [int(v) for v in ident["cluster"]]
    if mac >= 0.9:
        return "eigenvalue" if k in cl else ("higher" if min(cl) > k else "lower")
    return "mixture" if mac >= 0.5 else "unidentified"


def _e30_chains(data_dir: str, s: str) -> dict | None:
    """최종 사슬과 1회차 사슬 {이름: {시드: {단계 1–7: 기록}}} — `run_p5_neural.e30_chain_choice`로 다시 짓는다."""
    from .run_p5_neural import E30_LOOKAHEAD, _e30_files, e30_chain_choice
    sel1, sel2, plan = _e30_files(data_dir, s)
    if sel1 is None or plan is None or any(str(k) not in sel1["stages"] for k in range(1, E30_LOOKAHEAD + 1)):
        return None
    if plan["flagged"] and (sel2 is None or any(str(k) not in sel2["stages"]
                                                  for k in range(int(plan["j_min"]), E30_LOOKAHEAD + 1))):
        return None
    cache = {}

    def rec(tag, seed, k):
        if tag not in cache:
            cache[tag] = {(int(r["seed"]), int(r["mode"])): r for r in load(data_dir, tag)}
        return cache[tag][(seed, k)]

    seeds = sorted(int(v) for v in sel1["stages"]["1"]["choice"])
    out = {}
    for name, s2 in (("final", sel2 if plan["flagged"] else None), ("round1", None)):
        out[name] = {s_: {k: rec(e30_chain_choice(s, k, s_, sel1, s2, plan if s2 else None)[2], s_, k)
                          for k in range(1, E30_LOOKAHEAD + 1)} for s_ in seeds}
    return out


def e30_readouts(data_dir: str = DATA_DIR, res: dict | None = None, B: int = B_DEFAULT, seed: int = SEED) -> dict | None:
    """보완 사전규정 §4.2–4.5의 보조 판독. `res`는 `e30`의 결과(없으면 계산한다)."""
    from .run_p5_neural import E28_SELECTION, E30_STRUCTURES, e28_candidates
    res = e30(data_dir) if res is None else res
    if not res:
        return None
    rr_json = os.path.join(data_dir, "p5_exploit_rr_e30.json")
    rr_csv = os.path.join(data_dir, "p5_exploit_rr_e30.csv")
    rr = json.load(open(rr_json, encoding="utf-8")) if os.path.exists(rr_json) else {}
    rr_rows = []
    if os.path.exists(rr_csv):
        with open(rr_csv, encoding="utf-8") as f:
            rr_rows = list(csv.DictReader(f))
    out = {}
    for s in E30_STRUCTURES:
        if s not in res:
            continue
        ch = _e30_chains(data_dir, s)
        first = load(data_dir, E30_FIRST[s])
        srt = sorted_seed_accurate(first)
        q = {name: e30_chain_quality(c) for name, c in ch.items()}
        fp_by = {}
        for r in first:
            if int(r["mode"]) <= 6:
                fp_by.setdefault(int(r["seed"]), []).append(bool(accurate(r)))
        z = {"final": q["final"]["all_accurate_seeds"], "round1": q["round1"]["all_accurate_seeds"],
             "first_pass": int(sum(all(v) for v in fp_by.values())), "first_pass_sorted": int(sum(v == 6 for v in srt.values())),
             "first_pass_sorted_accurate": int(sum(srt.values()))}
        r_s = {"quality": q, "completeness": z}
        if s == "cc" and "d_full" in res[s]:
            lo = float(res[s]["d_full"]["contrast"][1])
            r_s["noninferiority"] = {"delta": E30_NI_DELTA, "contrast": res[s]["d_full"]["contrast"], "lower": lo,
                                     "noninferior": bool(lo > -E30_NI_DELTA)}
            if os.path.exists(os.path.join(data_dir, E28_SELECTION)):
                s28 = json.load(open(os.path.join(data_dir, E28_SELECTION), encoding="utf-8"))
                acc28 = {}
                for k in range(1, 7):
                    c_ = {p_: {int(r["seed"]): r for r in load(data_dir, t) if r["mode"] == k} for p_, t in e28_candidates(k).items()}
                    for s_, p_ in s28["stages"][str(k)]["choice"].items():
                        acc28.setdefault(int(s_), []).append(bool(accurate(c_[p_][int(s_)])))
                z["e28a_chain"] = int(sum(all(v) for v in acc28.values()))
        # 최종 부정확 셀의 유형(FEM 식별, b2의 최종 사슬 판독)
        lab = f"e30 {s} final seven"
        if lab in rr:
            ident = {int(x["seed"]): x.get("stage_ident") for x in rr[lab]["per_seed"]}
            types = {}
            for s_, c in ch["final"].items():
                for k in range(1, 7):
                    if not accurate(c[k]) and ident.get(s_):
                        t_ = _ident_class(k, ident[s_][k - 1])
                        types[t_] = types.get(t_, 0) + 1
            r_s["final_inaccurate_types"] = dict(sorted(types.items()))
        # E30 출력의 Rayleigh–Ritz(서술, 규칙 없음)
        fin = {s_: sum(bool(accurate(c[k])) for k in range(1, 7)) for s_, c in ch["final"].items()}
        rr_s = {}
        for name in ("final seven", "round1 seven", "pool"):
            lab = f"e30 {s} {name}"
            if lab not in rr:
                continue
            per = {}
            for row in rr_rows:
                if row["label"] == lab:
                    per[int(row["seed"])] = per.get(int(row["seed"]), 0) + (row["rr_accurate"] == "True")
            seeds = sorted(fin)
            a = np.array([per.get(s_, 0) / 6 for s_ in seeds])
            b = np.array([fin[s_] / 6 for s_ in seeds])
            rr_s[name] = {k_: rr[lab][k_] for k_ in ("rr_accurate", "rr_accurate_stable", "rr_all_accurate_seeds", "rr_fail",
                                                      "missing_modes", "dropped", "cond_max", "seconds_rr", "device")
                          if k_ in rr[lab]}
            rr_s[name]["vs_final_described"] = paired_bootstrap(a, b, B, seed)
        if rr_s:
            r_s["rayleigh_ritz"] = rr_s
        out[s] = r_s
    return out or None

# ── E31 (사전규정 2026-11-03-p5-e31-prespecification.md) — 정책을 고정한 보류 평가와 같은 예산 대조(B3) ────────────────────
# 보류 구조(양단 고정 h/L 0.07, L-브래킷 a 0.4)의 판정량은 E30과 같은 `_staged_judgment`이고, 첫 통과는 그 구조의 E31 첫 통과다.
# 예측(자료 전 기록)은 적중 여부만 적는다 — 투고처나 정책을 정하지 않는다. B3은 E30-cc 최종 사슬과 시드 짝으로 견준다.
E31_PREDICTION = {s: {"accurate_at_least": 114, "false_positives": 0, "mean_cost_ratio_at_most": 3.0,
                      "d_first_rule1_if_first_pass_at_most": 108} for s in ("cc07", "lb04")}
B3_PREDICTION = {"d_e30_final": "rule1", "accurate_at_most": 116}


def e31(data_dir: str = DATA_DIR) -> dict | None:
    """보류 구조마다 첫 통과와 단계적 배분(1회차, 계획, 울림이 있으면 2회차)이 끝났을 때만 판정한다."""
    from .run_p5_neural import E31_STRUCTURES, E31_FP_TAGS
    out = {}
    for s in E31_STRUCTURES:
        if not os.path.exists(os.path.join(data_dir, f"manifest_p5_neural_{E31_FP_TAGS[s]}.json")):
            continue
        r = _staged_judgment(data_dir, s, E31_FP_TAGS[s])
        if r is None:
            continue
        pred = E31_PREDICTION[s]
        cond = r["first_pass_accurate"] <= pred["d_first_rule1_if_first_pass_at_most"]
        r["verdict"] = {"prediction": pred, "hit": {
            "accurate": r["accurate"] >= pred["accurate_at_least"],
            "false_positives": r["false_positives_round1"] == r["false_positives_final"] == 0,
            "cost": r["cost"]["mean_ratio"] <= pred["mean_cost_ratio_at_most"],
            "d_first": (r["d_first"]["verdict"] == "rule1") if cond else None}}
        out[s] = r
    return out or None


def _sorted_from_ident(per_seed: list, lamref: dict, m: int) -> int:
    """B2 단계별 FEM 식별(`stage_ident`)에서 앞 m개 출력을 감사 Λ 순으로 세운 순위 1–6의 정확 수 — 순위 r의 출력이 모드 r(또는 r을
    담은 군집)이고 MAC ≥ 0.9, |Λ − λ*_r|/λ*_r ≤ 0.05(정렬 판독과 같은 규칙)."""
    n = 0
    for x in per_seed:
        srt = sorted(x["stage_ident"][:m], key=lambda d: d["lam"])[:6]
        n += sum(((d["best_mode"] == r) or (r in d["cluster"])) and d["best_mac"] >= 0.9
                 and abs(d["lam"] - lamref[r]) / lamref[r] <= 0.05 for r, d in enumerate(srt, 1))
    return int(n)


def e31_readouts(data_dir: str = DATA_DIR, res: dict | None = None, rr_name: str = "p5_exploit_rr_e31.json") -> dict | None:
    """E31 보조 판독(판정 아님) — 판정 품질·완결성(E30과 같은 양), 단계 7의 비교 가능성·검출, 첫 통과·1회차·최종 사슬의 정렬과
    Rayleigh–Ritz(`p5_exploit.b2(groups=('e31',))`), 최종 부정확 셀의 유형, B3의 Rayleigh–Ritz."""
    from .run_p5_neural import E31_FP_TAGS
    res = e31(data_dir) if res is None else res
    p = os.path.join(data_dir, rr_name)
    rr = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    keep = ("rr_accurate", "rr_accurate_stable", "rr_all_accurate_seeds", "rr_fail", "missing_modes", "dropped", "cond_max",
            "sorted_accurate", "first_accurate", "seconds_rr", "device")
    out = {}
    for s in (res or {}):
        ch = _e30_chains(data_dir, s)
        first = load(data_dir, E31_FP_TAGS[s])
        lamref = {int(r["mode"]): float(r["Lam_ref"]) for r in first if int(r["seed"]) == 0 and int(r["mode"]) <= 6}
        srt = sorted_seed_accurate(first)
        q = {name: e30_chain_quality(c) for name, c in ch.items()}
        fp_by = {}
        for r in first:
            if int(r["mode"]) <= 6:
                fp_by.setdefault(int(r["seed"]), []).append(bool(accurate(r)))
        r_s = {"quality": q,
               "completeness": {"final": q["final"]["all_accurate_seeds"], "round1": q["round1"]["all_accurate_seeds"],
                                "first_pass": int(sum(all(v) for v in fp_by.values())),
                                "first_pass_sorted": int(sum(v == 6 for v in srt.values())),
                                "first_pass_sorted_accurate": int(sum(srt.values()))},
               "stage7": {name: {"stable": int(sum(_stable(c[s_][7]) for s_ in c)),
                                 "stage6_flags": int(sum(6 in e30_order_flags(c[s_]) for s_ in c)),
                                 "stage6_inaccurate": int(sum(not accurate(c[s_][6]) for s_ in c))}
                          for name, c in ch.items()}}
        rr_s = {}
        for name, lab in (("first pass", f"first {s}"), ("final seven", f"e31 {s} final seven"),
                          ("round1 seven", f"e31 {s} round1 seven"), ("pool", f"e31 {s} pool")):
            if lab not in rr:
                continue
            rr_s[name] = {k_: rr[lab][k_] for k_ in keep if k_ in rr[lab]}
            per = rr[lab].get("per_seed") or []
            if per and all(x.get("stage_ident") for x in per):
                m = len(per[0]["stage_ident"])
                rr_s[name]["sorted_first_six"] = _sorted_from_ident(per, lamref, min(m, 6))
                if m >= 7:
                    rr_s[name]["sorted_seven"] = _sorted_from_ident(per, lamref, 7)
                    modes7 = [int(x["stage_ident"][6]["best_mode"]) for x in per]
                    rr_s[name]["stage7_modes"] = {str(v): modes7.count(v) for v in sorted(set(modes7))}
        if rr_s:
            r_s["rayleigh_ritz"] = rr_s
        lab = f"e31 {s} final seven"
        if lab in rr:
            ident = {int(x["seed"]): x.get("stage_ident") for x in rr[lab]["per_seed"]}
            types = {}
            for s_, c in ch["final"].items():
                for k in range(1, 7):
                    if not accurate(c[k]) and ident.get(s_):
                        t_ = _ident_class(k, ident[s_][k - 1])
                        types[t_] = types.get(t_, 0) + 1
            r_s["final_inaccurate_types"] = dict(sorted(types.items()))
        out[s] = r_s
    if "b3 cc" in rr:
        out["b3"] = {k_: rr["b3 cc"][k_] for k_ in keep if k_ in rr["b3 cc"]}
    return out or None


def b3_from_records(b3: list, final: list, round1: list, first: list, cost_b3: float, cost_first: float,
                    B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """B3(한 실행, 단계 1–6)과 E30-cc 최종·1회차 사슬, 첫 통과(E26b)의 시드 짝 비교. 대비는 E30 − B3(E30이 나으면 규칙 1)와
    B3 − 첫 통과다. 순서 검사·정렬은 판독으로만 센다(단계 7이 없어 단계 6은 검사할 수 없다)."""
    rep = tuple(range(1, 7))
    share = lambda recs: _seed_share(recs, accurate, rep)
    acc = {k: int(sum(accurate(r) for r in b3 if r["mode"] == k)) for k in rep}
    out = {"accurate_by_stage": acc, "accurate": int(sum(acc.values())), "e30_final_accurate": _n_acc(final, rep),
           "e30_round1_accurate": _n_acc(round1, rep), "first_pass_accurate": _n_acc(first, rep)}
    for name, recs in (("e30_final", final), ("e30_round1", round1)):
        ci = paired_bootstrap(share(recs), share(b3), B, seed)
        out[f"d_{name}"] = {"contrast": ci, "verdict": _rule(ci)}
    ci = paired_bootstrap(share(b3), share(first), B, seed)
    out["d_first"] = {"contrast": ci, "verdict": _rule(ci)}
    chains = {}
    for r in b3:
        if r["mode"] in rep:
            chains.setdefault(int(r["seed"]), {})[int(r["mode"])] = r
    flags = {s_: e30_order_flags(c) for s_, c in sorted(chains.items())}
    out["order_check"] = {"flags": int(sum(len(v) for v in flags.values())),
                          "false_positives": int(sum(accurate(chains[s_][j]) for s_, v in flags.items() for j in v)),
                          "inaccurate_stages_1_5": int(sum(not accurate(c[j]) for c in chains.values() for j in range(1, 6))),
                          "by_seed": {str(s_): v for s_, v in flags.items() if v}}
    out["sorted_accurate"] = int(sum(sorted_seed_accurate(b3).values()))
    out["cost"] = {"gpu_h": float(cost_b3), "first_pass_gpu_h": float(cost_first), "ratio": float(cost_b3 / cost_first)}
    return out


def b3(data_dir: str = DATA_DIR, res30: dict | None = None) -> dict | None:
    """B3 판정. E30-cc의 최종·1회차 사슬이 있어야 한다. `res30`(E30 판정)이 있으면 E30의 시드 평균 비용 비를 함께 적는다."""
    from .run_p5_neural import B3_TAG
    if not os.path.exists(os.path.join(data_dir, f"manifest_p5_neural_{B3_TAG}.json")):
        return None
    ch = _e30_chains(data_dir, "cc")
    if ch is None:
        return None
    flat = lambda c: [c[s_][k] for s_ in sorted(c) for k in range(1, 7)]
    r = b3_from_records(load(data_dir, B3_TAG), flat(ch["final"]), flat(ch["round1"]), load(data_dir, E30_FIRST["cc"]),
                        _gpu_hours(data_dir, B3_TAG), _gpu_hours(data_dir, E30_FIRST["cc"]))
    if res30 and "cc" in res30:
        r["cost"]["e30_mean_ratio"] = float(res30["cc"]["cost"]["mean_ratio"])
    r["verdict"] = {"prediction": B3_PREDICTION,
                    "hit": {"d_e30_final": r["d_e30_final"]["verdict"] == B3_PREDICTION["d_e30_final"],
                            "accurate": r["accurate"] <= B3_PREDICTION["accurate_at_most"]}}
    return r


# ── E31r(사전규정 2026-11-07-p5-e31r-prespecification.md) — E31-lb04 단계 6의 사후 재풀이 판독 ─────────────────────
# 울렸다면(가상: 20시드 모두 j* = 6) 2회차가 단계 6에서 고를 네 경로(1회차의 first·norm, E31r의 ff·long)에 2회차의 고르기 규칙
# (`e30_select_stage`)을 그대로 적용한다. 등록 판정이 아니다 — E31의 판정은 바뀌지 않는다. 예측은 자료 전에 적었다.
E31R_PREDICTION = {"long_accurate_at_least": 10, "stage6_accurate_at_least": 10, "d_round1": "rule1",
                   "selection_missed_at_most": 2}


def e31r_from_records(stage6: dict, chain: list, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """stage6 = {경로: 단계 6 기록 목록}(이 순서가 동률을 깬다), chain = 1회차 사슬의 단계 1–6 기록. 가상 사슬은 단계 6만 네 경로에서
    고른 해로 바꾼 것이다. `selection_missed`는 정확한 경로가 있었는데 고른 해가 부정확한 시드다."""
    k, rep = 6, tuple(range(1, 7))
    by = {p_: {int(r["seed"]): r for r in recs if int(r["mode"]) == k} for p_, recs in stage6.items()}
    paths = tuple(stage6)
    seeds = sorted(by[paths[0]])
    choice = e30_select_stage(stage6, k, {s_: paths for s_ in seeds})
    picked = {s_: by[choice[s_]][s_] for s_ in seeds}
    hyp = [r for r in chain if int(r["mode"]) != k] + [picked[s_] for s_ in seeds]
    share = lambda recs: _seed_share(recs, accurate, rep)
    ci = paired_bootstrap(share(hyp), share(chain), B, seed)
    shape6 = lambda r: int(r["matched_mode"]) == k and r["mac"] >= 0.9
    some = [s_ for s_ in seeds if any(accurate(by[p_][s_]) for p_ in paths)]
    count = lambda pred: {p_: int(sum(pred(by[p_][s_]) for s_ in seeds)) for p_ in paths}
    return {"accurate_by_path": count(accurate), "mode6_shape_by_path": count(shape6),
            "unstable_by_path": count(lambda r: not _stable(r)),
            "chosen": {p_: int(sum(choice[s_] == p_ for s_ in seeds)) for p_ in paths},
            "stage6_accurate": int(sum(accurate(picked[s_]) for s_ in seeds)),
            "seeds_with_an_accurate_path": len(some),
            "selection_missed": [s_ for s_ in some if not accurate(picked[s_])],
            "round1_accurate": _n_acc(chain, rep), "hypothetical_accurate": _n_acc(hyp, rep),
            "d_round1": {"contrast": ci, "verdict": _rule(ci)},
            "per_seed": {str(s_): {"chosen": choice[s_], **{p_: [float(by[p_][s_]["Lam"]), bool(accurate(by[p_][s_]))]
                                                             for p_ in paths}} for s_ in seeds}}


def e31r(data_dir: str = DATA_DIR) -> dict | None:
    """E31r 판독. ff·long 실행과 E31-lb04의 1회차 사슬이 있을 때만 낸다. 비용은 두 실행의 GPU-시간과 첫 통과 대비의 더한 비다."""
    from .run_p5_neural import E31_FP_TAGS, E31R_PATHS, E31R_STAGE, E31R_STRUCTURE, e30_tag, e31r_tag
    tags = {p_: e31r_tag(p_) for p_ in E31R_PATHS}
    if not all(os.path.exists(os.path.join(data_dir, f"manifest_p5_neural_{t}.json")) for t in tags.values()):
        return None
    ch = _e30_chains(data_dir, E31R_STRUCTURE)
    if ch is None:
        return None
    base = {p_: e30_tag(E31R_STRUCTURE, 1, E31R_STAGE, p_) for p_ in ("first", "norm")}
    stage6 = {p_: load(data_dir, t) for p_, t in {**base, **tags}.items()}
    chain = [ch["round1"][s_][k] for s_ in sorted(ch["round1"]) for k in range(1, 7)]
    r = e31r_from_records(stage6, chain)
    gh = {p_: _gpu_hours(data_dir, t) for p_, t in tags.items()}
    first = _gpu_hours(data_dir, E31_FP_TAGS[E31R_STRUCTURE])
    r["cost"] = {"gpu_h": gh, "first_pass_gpu_h": first, "added_ratio": float(sum(gh.values()) / first)}
    pred = E31R_PREDICTION
    r["verdict"] = {"prediction": pred, "hit": {
        "long": r["accurate_by_path"]["long"] >= pred["long_accurate_at_least"],
        "stage6": r["stage6_accurate"] >= pred["stage6_accurate_at_least"],
        "d_round1": r["d_round1"]["verdict"] == pred["d_round1"],
        "selection": len(r["selection_missed"]) <= pred["selection_missed_at_most"]}}
    return r


# E31p — E31-lb04 단계 6의 표현 판독(사전규정 2026-11-08-p5-e31p-prespecification.md). E21f와 같은 규칙의 사후 판독이고, E31의
# 등록 판정은 바뀌지 않는다. 예측은 데이터 전에 여기 적는다.
E31P_FILES = {6: "p5_e1_oracle_fit_e31p_lb04_m6.csv", 7: "p5_e1_oracle_fit_e31p_lb04_m7.csv"}
E31P_PREDICTION = {"verdict": "rule1", "n_below_own_stage6_at_least": 18, "control_ok": True}


def e31p_from_rows(rows6, rows7, stage6: dict, stage6_round1: dict | None = None) -> dict:
    """주 판정(E21f의 규칙): 모드 6에 맞춘 망을 E31-lb04 첫 통과의 저장 모드 1–5로 정확 사영한 Rayleigh 몫의 중앙값이 첫 통과
    단계 6의 Λ 중앙값보다 낮으면 규칙 1(더 낮은 함수가 시행공간에 있었는데 최적화가 놓쳤다), 아니면 규칙 2(정하지 못함).
    시드마다 같은 시드 단계 6의 Λ와도 견준다. 대조: 모드 7 적합의 맞춘 격자 밖 e_λ 중앙값이 0.05 미만이면 적합 절차가 그
    정확도를 낸다. `stage6_round1`(있으면)은 1회차 사슬 단계 6의 Λ — 서술용이다."""
    d6 = {int(r["seed"]): float(r["Lam_deflated"]) for r in rows6}
    d7 = {int(r["seed"]): float(r["Lam_deflated"]) for r in rows7}
    ld, le = float(np.median(list(d6.values()))), float(np.median(list(stage6.values())))
    med = lambda rows, key: float(np.median([float(r[key]) for r in rows]))
    e7 = med(rows7, "e_lam_check_0")
    out = {"lam_deflated_median": ld, "lam_stage6_median": le, "verdict": "rule1" if ld < le else "rule2",
           "n_below_own_stage6": int(sum(d6[s_] < stage6[s_] for s_ in d6)),
           "lam_deflated_range": [float(min(d6.values())), float(max(d6.values()))],
           "lam_stage6_range": [float(min(stage6.values())), float(max(stage6.values()))],
           "e_lam_held_out_median": {"6": med(rows6, "e_lam_check_0"), "7": e7},
           "mac_held_out_median": {"6": med(rows6, "mac_check_0"), "7": med(rows7, "mac_check_0")},
           "control_ok": bool(e7 < E21F_CONTROL_RULE),
           "lam_deflated_7_median": float(np.median(list(d7.values()))),
           "n7_below_own_stage6": int(sum(d7[s_] < stage6[s_] for s_ in d7))}
    if stage6_round1 is not None:
        out["n_below_round1_stage6"] = int(sum(d6[s_] < stage6_round1[s_] for s_ in d6))
    return out


def e31p(data_dir: str = DATA_DIR) -> dict | None:
    import csv
    from .run_p5_neural import E31_FP_TAGS
    try:
        rows = {m: list(csv.DictReader(open(os.path.join(data_dir, f), encoding="utf-8")))
                for m, f in E31P_FILES.items()}
        fp = load(data_dir, E31_FP_TAGS["lb04"])
    except FileNotFoundError:
        return None
    st6 = {int(r["seed"]): float(r["Lam"]) for r in fp if int(r["mode"]) == 6}
    ch = _e30_chains(data_dir, "lb04")
    r1 = {int(s_): float(c[6]["Lam"]) for s_, c in ch["round1"].items()} if ch else None
    r = e31p_from_rows(rows[6], rows[7], st6, r1)
    pred = E31P_PREDICTION
    r["verdict_prediction"] = {"prediction": pred, "hit": {
        "verdict": r["verdict"] == pred["verdict"],
        "n_below_own_stage6": r["n_below_own_stage6"] >= pred["n_below_own_stage6_at_least"],
        "control_ok": r["control_ok"] == pred["control_ok"]}}
    return r


# E32 — 대수 층 보강(사전규정 2026-11-09-p5-e32-prespecification.md). 저장 관행(감사 격자에서 직교화해 저장하고 쓸 때 자기 계수로
# 뺀다, 원고 식 (3))을 T-브래킷 원좌표 첫 통과(정확 사영·원시 저장)와 같은 시드·같은 표본에서, 기준 구조에서는 B = 283에서 같은
# 표본의 정확 사영 대조와 함께 돌린다. 판정량은 E15와 같은 게이트 없는 정확도(모드 4–6)이고, 드리프트는 실행이 훈련 표본 위에서
# 기록한 max|C|(`batch_offdiag`, 단계 마지막 10 % 평균)의 시드 중앙이다. 예측은 데이터 전에 여기 적는다.
E32_TAGS = {"tb_storage": "e32_tb_orth_mc", "b283_storage": "e32_b283_orth_mc", "b283_exact": "e32_b283_exact_mc"}
E32_TB_EXACT = "e26_tb_raw_mc"                 # E26d — 같은 설정의 정확 사영 첫 통과
E32_B_STORAGE = {283: "e32_b283_orth_mc", 1134: "e15r", 4536: "e20r_b4"}
E32_B_EXACT = {283: "e32_b283_exact_mc", 1134: "e15b"}
E32_DRIFT_STAGES = (4, 5, 6)
E32_DRIFT_FILE = "p5_exploit_e32_drift_prediction.csv"
# 자료 전 예측(드리프트 예측 `E32_DRIFT_FILE`을 본 뒤, 신경망 실행 전에 적었다 — 사전규정 §4).
E32_PREDICTION = {"tb_verdict": "rule1", "tb_storage_modes_at_most": {4: 10, 5: 10}, "tb_failing_mass_at_most": 0.01,
                  "tb_drift_factor": 1.25, "tb_drift_stages": [3, 4],
                  "b283_verdict": "rule1", "b283_exact_acc_4_6_at_least": 50,
                  "b283_ratio_range": [1.7, 2.5], "b283_ratio_stages": [4]}


def shape_kept(rec, mac_min: float = 0.9) -> bool:
    """목표 모드의 형상(짝 모드 = 단계, MAC ≥ 0.9)은 지켰는가 — 고유값만 틀린 셀을 가른다."""
    return int(rec["matched_mode"]) == int(rec["mode"]) and rec["mac"] >= mac_min


def _median_or_none(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def e32_arm(recs) -> dict:
    """실행 하나의 서술: 모드별 정확 셀, 모드 4–6 정확 셀, 모드별 남은 질량 중앙, 단계별 드리프트 중앙(저장 관행이면 저장 함수의,
    정확 사영이면 원시 출력의 비대각), 사다리 안정 셀, 정확하지 않은 모드 4–6 셀(형상을 지킨 수, 결과 분류, 남은 질량, e_λ)."""
    by_mode = lambda f: {m: f([r for r in recs if int(r["mode"]) == m]) for m in range(1, 7)}
    fail = [r for r in recs if r["mode"] in WALL_MODES and not accurate(r)]
    outcomes = {}
    for r in fail:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {"accurate_by_mode": by_mode(lambda c: int(sum(accurate(r) for r in c))), "acc_4_6": _n_acc(recs),
            "remaining_mass_median_by_mode": by_mode(lambda c: _median_or_none([r["projected_mass_fraction"] for r in c])),
            "remaining_mass_median_4_6": _median_or_none([r["projected_mass_fraction"] for r in recs
                                                          if r["mode"] in WALL_MODES]),
            "drift_median_by_stage": {m: _median_or_none([r.get("batch_offdiag") for r in recs if int(r["mode"]) == m])
                                      for m in (3, 4, 5, 6)},
            "ladder_stable": int(sum(_stable(r) for r in recs)),
            "failing_4_6": {
                "n": len(fail), "shape_kept": int(sum(shape_kept(r) for r in fail)), "outcomes": outcomes,
                "remaining_mass_median": _median_or_none([r["projected_mass_fraction"] for r in fail]),
                "e_lam_median_by_mode": {m: _median_or_none([r["e_lam"] for r in fail if int(r["mode"]) == m])
                                         for m in WALL_MODES}}}


def e32_pair(recs_exact, recs_storage, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """같은 시드·같은 표본에서 정확 사영(원시 저장)과 저장 관행. D = acc₄₋₆(정확) − acc₄₋₆(저장 관행), 시드 짝지음,
    규칙 1: 0 제외·양수, 규칙 2: 0 포함, 규칙 3: 0 제외·음수. 두 실행의 서술(`e32_arm`)을 함께 둔다."""
    ci = paired_bootstrap(_seed_share(recs_exact, accurate), _seed_share(recs_storage, accurate), B, seed)
    return {"contrast": ci, "verdict": _rule(ci), "exact": e32_arm(recs_exact), "storage": e32_arm(recs_storage)}


def e32_drift_law(drift: dict, stages=E32_DRIFT_STAGES) -> dict:
    """{B: {단계: 드리프트 중앙}} → 단계마다 log 드리프트의 log B 최소제곱 기울기(세 점 이상일 때)와 B 283 대 1,134의 비."""
    out = {}
    for k in stages:
        pts = sorted((int(B), v[k]) for B, v in drift.items() if v.get(k) is not None and v[k] > 0)
        slope = (float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
                 if len(pts) >= 3 else None)
        d = dict(pts)
        out[k] = {"slope": slope, "ratio_283_1134": (d[283] / d[1134] if 283 in d and 1134 in d else None),
                  "points": [[B, v] for B, v in pts]}
    return out


def e32_prediction_medians(rows, source: str, B: int) -> dict:
    """드리프트 예측 파일(`p5_exploit.e32_drift_prediction`)에서 한 출처·한 표본 크기의 단계별 시드 중앙."""
    out = {}
    for r in rows:
        if r["source"] == source and int(r["B"]) == int(B):
            out.setdefault(int(r["stage"]), []).append(float(r["drift"]))
    return {k: float(np.median(v)) for k, v in sorted(out.items())}


def e32_from_records(tb_storage, tb_exact, storage_by_B: dict, b283_exact, pred_rows=None,
                     B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """T-브래킷 쌍(주 판정 D_T), 기준 구조 B = 283의 쌍(D_283), 세 B의 드리프트 법칙. `storage_by_B`는 {B: 저장 관행 기록}."""
    out = {"tbracket": e32_pair(tb_exact, tb_storage, B, seed),
           "b283": e32_pair(b283_exact, storage_by_B[283], B, seed),
           "storage_by_B": {int(b): e32_arm(rs) for b, rs in storage_by_B.items()}}
    out["drift_law"] = e32_drift_law({b: v["drift_median_by_stage"] for b, v in out["storage_by_B"].items()})
    if pred_rows:
        out["drift_prediction"] = {
            "tbracket_from_e26d": e32_prediction_medians(pred_rows, E32_TB_EXACT, 1651),
            "b283_from_e15r": e32_prediction_medians(pred_rows, "e15r", 283),
            "b283_from_e15b": e32_prediction_medians(pred_rows, "e15b", 283)}
    return out


def e32_hits(r: dict, pred: dict) -> dict:
    """자료 전 예측(`E32_PREDICTION`)과 견준 적중. 키가 없는 예측은 매기지 않는다."""
    t, b, law = r["tbracket"], r["b283"], r["drift_law"]
    hit = {}
    if "tb_verdict" in pred:
        hit["tb_verdict"] = t["verdict"] == pred["tb_verdict"]
    if "tb_storage_acc_4_6_at_most" in pred:
        hit["tb_storage_acc_4_6"] = t["storage"]["acc_4_6"] <= pred["tb_storage_acc_4_6_at_most"]
    if "tb_storage_modes_at_most" in pred:
        hit["tb_storage_modes"] = all(t["storage"]["accurate_by_mode"][int(m)] <= n
                                      for m, n in pred["tb_storage_modes_at_most"].items())
    if "tb_failing_mass_at_most" in pred:
        m = t["storage"]["failing_4_6"]["remaining_mass_median"]
        hit["tb_failing_mass"] = m is not None and m <= pred["tb_failing_mass_at_most"]
    if "tb_drift_factor" in pred and r.get("drift_prediction"):
        f, pm, got = pred["tb_drift_factor"], r["drift_prediction"]["tbracket_from_e26d"], t["storage"]["drift_median_by_stage"]
        hit["tb_drift"] = all(got.get(k) is not None and pm[k] / f <= got[k] <= pm[k] * f for k in pred["tb_drift_stages"])
    if "b283_ratio_range" in pred:
        lo, hi = pred["b283_ratio_range"]
        hit["b283_ratio"] = all(law[k]["ratio_283_1134"] is not None and lo <= law[k]["ratio_283_1134"] <= hi
                                for k in pred["b283_ratio_stages"])
    if "b283_verdict" in pred:
        hit["b283_verdict"] = b["verdict"] == pred["b283_verdict"]
    if "b283_exact_acc_4_6_at_least" in pred:
        hit["b283_exact_acc_4_6"] = b["exact"]["acc_4_6"] >= pred["b283_exact_acc_4_6_at_least"]
    if "b283_storage_modes_at_most" in pred:
        hit["b283_storage_modes"] = all(b["storage"]["accurate_by_mode"][int(m)] <= n
                                        for m, n in pred["b283_storage_modes_at_most"].items())
    return hit


def e32(data_dir: str = DATA_DIR) -> dict | None:
    import csv
    try:
        tb = load(data_dir, E32_TAGS["tb_storage"])
        tbe = load(data_dir, E32_TB_EXACT)
        sto = {b: load(data_dir, t) for b, t in E32_B_STORAGE.items()}
        b283e = load(data_dir, E32_B_EXACT[283])
    except FileNotFoundError:
        return None
    p = os.path.join(data_dir, E32_DRIFT_FILE)
    rows = list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else None
    r = e32_from_records(tb, tbe, sto, b283e, rows)
    if E32_PREDICTION is not None:
        r["verdict_prediction"] = {"prediction": E32_PREDICTION, "hit": e32_hits(r, E32_PREDICTION)}
    return r


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
           "e21": e21(data_dir), "e21b": e21b(data_dir), "e21c": e21c(data_dir),
           "e21d": e21d(data_dir), "e21e": e21e(data_dir), "e21f": e21f(data_dir),
           "e21g": e21g(data_dir), "e21h": e21h(data_dir), "e23": e23(data_dir),
           "e24": e24(data_dir), "e25": e25(data_dir), "e26": e26(data_dir), "e26e": e26e(data_dir), "e27": e27(data_dir), "e28": e28(data_dir), "e29": e29(data_dir), "e30": (r30 := e30(data_dir)), "e30_readouts": e30_readouts(data_dir, r30),
           "e31": (r31 := e31(data_dir)), "e31_readouts": e31_readouts(data_dir, r31), "b3": b3(data_dir, r30),
           "e31r": e31r(data_dir), "e31p": e31p(data_dir), "e32": e32(data_dir),
           "e22": e22(data_dir),
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
