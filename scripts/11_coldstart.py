"""
scripts/11_coldstart.py
=======================
SkillBridge, Component 1: COLD-START PLACEMENT.
Owner: Muhammad Aarij (V01096775).  CSC 503 Data Mining, Summer 2026.


THE QUESTION
------------
    Given a person's competency profile, can we place them among 900 Canadian
    occupations better than we could WITHOUT machine learning?

Not "better than random". Random is the floor. The bar is what a competent analyst
does with no model at all: compare the person's self-ratings directly to each
occupation's official ratings on those same competencies.


A BUG THIS VERSION FIXES  (the most important lesson in this file)
------------------------------------------------------------------
The previous version tuned the fold-in ridge parameter lambda by maximising

        cos( p_folded , p_true )

on simulated cold starts. That objective is SCALE-INVARIANT: shrink p by 100x and the
cosine is unchanged. So the tuner cranked lambda all the way to the edge of the grid
(20.0) with no penalty, because heavy shrinkage cleans up p's DIRECTION for free.

But one of the mechanisms, mf_impute, does not use cosine. It RECONSTRUCTS:

        r_hat(d) = mu + b_you + b_d + p_you . q_d

That is SCALE-SENSITIVE. At lambda = 20, p_you is crushed toward zero, so

        r_hat(d)  ~=  mu + b_you + b_d

and b_you is a CONSTANT across all 181 descriptors, so it VANISHES the moment we centre
for the Pearson correlation. What remains is b_centred: the SAME VECTOR FOR EVERY PERSON.
At lambda = 20, mf_impute stops distinguishing people at all.

The evidence was sitting in plain sight. Running the query tool at lambda = 1.0 with a
Registered Nurse profile put 8 of the top 10 in NOC major group 31 (P@10 = 0.80). The
cold-start table, at lambda = 20, reported mf_impute at 0.21. Same model, same
mechanism, same data. Only lambda differed.

    LESSON 1: TUNE ON THE METRIC YOU WILL REPORT, NOT ON A PROXY FOR IT.
              A scale-invariant tuning objective silently destroyed a scale-sensitive
              downstream model.

    LESSON 2: A HYPERPARAMETER SWEEP WHOSE OPTIMUM SITS AT THE EDGE OF THE GRID IS A
              WARNING, NOT A RESULT.

Fixed here: lambda is tuned SEPARATELY FOR EACH MECHANISM, directly on P@10, by
simulating cold start on WARM occupations.


A SECOND ERROR THIS VERSION FIXES
----------------------------------
The previous headline comparison was  mf_impute  vs  raw_correlation.  But the results
table showed mf_latent beating everything at every profile size, and raw_distance
beating raw_correlation badly at small |S|. The experiment had been pointed at the
WEAKER model and the WEAKER baseline.

    LESSON 3: CHECK THAT YOUR HEADLINE COMPARISON POINTS AT YOUR STRONGEST MODEL AND
              YOUR STRONGEST BASELINE. Otherwise you are staging a fight you have
              already rigged.

Fixed here: the headline is  BEST MODEL  vs  BEST NO-LEARNING BASELINE, both selected
on the data rather than assumed.


THE METHODS
-----------
NO LEARNING (the bar):
    random           the floor. Verifies the metric code is honest.
    popular_group    always recommend the largest NOC group. A constant predictor.
                     Beating random is trivial. Beating THIS is the minimum bar for
                     claiming the model reads the query at all.
    raw_distance     -RMSE between the person's ratings and each occupation's ratings
                     on the SAME revealed competencies. Level-sensitive.
    raw_correlation  Pearson r on the revealed competencies. Level-invariant, matches
                     profile SHAPE.

WITH LEARNING (the fold-in):
    mf_latent        cosine( p_you , p_occupation ) in the learned latent space.
                     SCALE-INVARIANT, so it is robust to lambda.
    mf_impute        reconstruct all 181 of the person's ratings, then Pearson r against
                     each occupation's true 181-vector. 181 - |S| dimensions are IMPUTED.
                     SCALE-SENSITIVE, so lambda matters enormously.

mf_impute is the cleanest CONTROLLED counterpart to raw_correlation: identical metric,
identical ground truth, differing ONLY in whether the unrevealed dimensions are filled
in by the model. mf_latent is a different question ("who is near me in latent space?")
and is not directly comparable, but it is reported because it is the strongest method.


GROUND TRUTH
------------
NOC codes are hierarchical:
    digit 1     broad category  (10 groups, chance P@10 ~ 0.12)
    digits 1-2  major group     (45 groups, chance P@10 ~ 0.04)  <- the harder test
    digit 2     TEER, the required education tier

The model NEVER SEES NOC CODES. It sees occupation_id (an arbitrary integer),
descriptor_id, and a rating. Recovering Canada's occupational hierarchy from competency
ratings alone is genuine structure discovery, not circularity.


WHAT THIS DOES NOT TEST
------------------------
We use an OCCUPATION'S PROFILE AS A PROXY FOR A PERSON. Nobody has labelled "person X
fits occupation Y"; that data does not exist. So this measures whether the model can
place a coherent, expert-authored competency profile. It does NOT measure whether a real
human with a noisy, incomplete self-assessment gets good advice. Say so in the report.


USAGE
    python scripts/11_coldstart.py
    python scripts/11_coldstart.py --n-cold 90 --bootstrap 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from skillbridge.config import SEED, set_all_seeds
from skillbridge.models import OCC, DSC, RAT, Matrix, load_raw, MFRecommender, fold_in

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COMPONENT = "coldstart"

METHODS = ["random", "popular_group", "raw_distance", "raw_correlation",
           "mf_latent", "mf_impute"]
NO_LEARNING = ["random", "popular_group", "raw_distance", "raw_correlation"]
LEARNED = ["mf_latent", "mf_impute"]
LAMBDA_GRID = [0.05, 0.2, 1.0, 3.0, 10.0, 30.0, 100.0]


# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def noc_group(noc: str, digits: int) -> str:
    return str(noc).zfill(5)[:digits]


def pearson_rows(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pearson r between every row of A and the vector v.

    The centring is not cosmetic. Without it, occupations that rate EVERYTHING highly
    (surgeons, senior managers) correlate with every possible profile. That popularity
    bias is what made an early naive scorer return ophthalmologists for an early
    childhood educator.
    """
    Ac = A - A.mean(axis=1, keepdims=True)
    vc = v - v.mean()
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(vc)
    out = np.zeros(A.shape[0])
    ok = denom > 1e-9
    out[ok] = (Ac[ok] @ vc) / denom[ok]
    return out


def p_at_k(scores, rel, k=10):
    return float(rel[np.argsort(-scores)[:k]].mean())


def rank_metrics(scores, rel, ks=(5, 10)):
    order = np.argsort(-scores)
    r = rel[order]
    out = {f"p@{k}": float(r[:k].mean()) for k in ks}
    hits = np.flatnonzero(r)
    out["mrr"] = float(1.0 / (hits[0] + 1)) if len(hits) else 0.0
    return out


def boot_ci(x, n_boot=1000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def sample_profile(cells: pd.DataFrame, n: int, strategy: str, rng):
    """Which n of an occupation's 181 ratings does the 'person' reveal?

    core_first  Highest-rated first. REALISTIC (asked what you are good at, you name
                strengths) but it FAVOURS US, because core competencies are the most
                discriminative. Ablated below, never assumed.
    uniform     n at random. Unbiased, less realistic.
    """
    if n >= len(cells):
        return cells
    if strategy == "core_first":
        return cells.sort_values(RAT, ascending=False, kind="mergesort").head(n)
    return cells.iloc[rng.choice(len(cells), size=n, replace=False)]


# ═══════════════════════════════════════════════════════════════════════════
# SCORERS
# ═══════════════════════════════════════════════════════════════════════════

def score_all(mf, S, r_person, cand, R_full, n_desc, lam_latent, lam_impute, rng):
    """Every method's score vector over the candidate occupations.

    NOTE the two fold-ins. mf_latent is scale-INVARIANT (cosine) and mf_impute is
    scale-SENSITIVE (reconstruction), so they need DIFFERENT lambdas. Sharing one is
    exactly the bug that crippled the previous run.
    """
    R_cand_S = R_full[np.ix_(cand, S)]
    R_cand_full = R_full[cand]

    out = {}
    out["random"] = rng.random(len(cand))

    d = R_cand_S - r_person[None, :]
    out["raw_distance"] = -np.sqrt((d ** 2).mean(axis=1))
    out["raw_correlation"] = pearson_rows(R_cand_S, r_person)

    p_lat, _ = fold_in(mf, S, r_person, reg=lam_latent)
    P = mf.P[cand]
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(p_lat) + 1e-12
    out["mf_latent"] = (P @ p_lat) / den

    p_imp, b_imp = fold_in(mf, S, r_person, reg=lam_impute)
    d_all = np.arange(n_desc)
    r_hat = np.clip(mf.mu + b_imp + mf.bi[d_all] + mf.Q[d_all] @ p_imp, 0, 5)
    out["mf_impute"] = pearson_rows(R_cand_full, r_hat)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# LAMBDA TUNING  -- on the downstream metric, per mechanism
# ═══════════════════════════════════════════════════════════════════════════

def tune_lambda(mf, warm, warm_df, R_full, n_desc, groups, rng,
                n_probe=120, n_rev=20, seed=0):
    """Choose lambda by simulating cold start on WARM occupations and maximising P@10.

    We fold a warm occupation in AS IF it were new (using only n_rev of its ratings),
    rank the OTHER warm occupations, and score against NOC 2-digit ground truth. Then we
    pick the lambda that maximises P@10 -- THE METRIC WE ACTUALLY REPORT.

    Done SEPARATELY for mf_latent and mf_impute, because one is scale-invariant and the
    other is not. A single shared lambda is precisely what broke the previous run.

    CAVEAT, stated rather than hidden: MF was trained on these occupations, so their
    latent vectors are partly memorised and the absolute numbers here are optimistic.
    But the OPTIMISM IS UNIFORM ACROSS LAMBDA, and we only need the argmax. The cold set
    is never touched.
    """
    print("\n  TUNING the fold-in lambda  (simulated cold start on WARM occupations)")
    print("  objective: P@10 on NOC 2-digit -- THE METRIC WE REPORT, not a proxy for it")
    print("  " + "-" * 86)

    probe = rng.choice(warm, size=min(n_probe, len(warm)), replace=False)
    warm_set = set(warm.tolist())

    best = {}
    for mech in LEARNED:
        scores = []
        for lam in LAMBDA_GRID:
            vals = []
            for w in probe:
                w = int(w)
                cells = warm_df[warm_df[OCC] == w]
                if len(cells) < 3:
                    continue
                sel = sample_profile(cells, n_rev, "core_first",
                                     np.random.default_rng(seed * 7919 + w))
                S = sel[DSC].to_numpy()
                r_p = sel[RAT].to_numpy().astype(float)

                cand = np.array(sorted(warm_set - {w}))       # exclude self
                rel = (groups[cand] == groups[w]).astype(int)
                if rel.sum() == 0:
                    continue

                lam_lat = lam if mech == "mf_latent" else 1.0
                lam_imp = lam if mech == "mf_impute" else 1.0
                sc = score_all(mf, S, r_p, cand, R_full, n_desc, lam_lat, lam_imp,
                               np.random.default_rng(0))
                vals.append(p_at_k(sc[mech], rel, 10))
            scores.append(float(np.mean(vals)) if vals else np.nan)

        i = int(np.nanargmax(scores))
        best[mech] = LAMBDA_GRID[i]
        row = "  ".join(f"{lam:>5g}:{s:.3f}" for lam, s in zip(LAMBDA_GRID, scores))
        print(f"    {mech:<11} {row}")
        print(f"    {'':<11} -> lambda = {best[mech]}   (P@10 = {scores[i]:.3f})")
        if i in (0, len(LAMBDA_GRID) - 1):
            print(f"    {'':<11} !! optimum at the EDGE of the grid. Widen it before trusting this.")

    print()
    print(f"  mf_latent uses lambda = {best['mf_latent']:<6g} (cosine: SCALE-INVARIANT, tolerant of shrinkage)")
    print(f"  mf_impute uses lambda = {best['mf_impute']:<6g} (reconstruction: SCALE-SENSITIVE, "
          f"over-shrinking destroys it)")
    print("  Different mechanisms, different regularisation. Sharing one lambda was the bug.")
    return best


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cold", type=int, default=90)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reveals", nargs="+", type=int, default=[5, 10, 20, 40, 80, 181])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--fairness-n", type=int, default=20)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 100)
    print("  SkillBridge / Component 1 / COLD-START PLACEMENT")
    print("  Q: can we place a person into occupations better than we could WITHOUT")
    print("     machine learning? (the bar is raw matching, not random)")
    print("=" * 100)

    df, n_occ, n_desc, occ_names, desc_names, noc_of = load_raw(PROCESSED_DIR)
    R_full = np.zeros((n_occ, n_desc))
    R_full[df[OCC].to_numpy(), df[DSC].to_numpy()] = df[RAT].to_numpy()

    all_occ = np.arange(n_occ)
    cold = np.sort(rng.choice(all_occ, size=args.n_cold, replace=False))
    warm = np.setdiff1d(all_occ, cold)
    cold_set = set(cold.tolist())

    warm_df = df[~df[OCC].isin(cold_set)].reset_index(drop=True)
    cold_df = df[df[OCC].isin(cold_set)].reset_index(drop=True)

    print(f"\n  warm (training) : {len(warm):>4} occupations, {len(warm_df):>7,} cells")
    print(f"  COLD (unseen)   : {len(cold):>4} occupations, {len(cold_df):>7,} cells")
    print("  The cold occupations have NO latent vector. The model does not know they exist.")

    R = np.zeros((n_occ, n_desc))
    M = np.zeros((n_occ, n_desc))
    R[warm_df[OCC], warm_df[DSC]] = warm_df[RAT]
    M[warm_df[OCC], warm_df[DSC]] = 1.0
    mx = Matrix(n_occ, n_desc, warm_df, cold_df, R, M, occ_names, desc_names, noc_of)

    print(f"\n  training MF (k={args.k}) on the warm occupations only ...")
    mf = MFRecommender(k=args.k, epochs=args.epochs)
    mf.fit(mx, np.random.default_rng(args.seed))
    print(f"  done. inner-val RMSE {mf.val_rmse:.4f}, stopped at epoch {mf.stopped_at}")

    # ---- ground truth --------------------------------------------------------
    g1 = np.array([noc_group(noc_of[int(u)], 1) for u in range(n_occ)])
    g2 = np.array([noc_group(noc_of[int(u)], 2) for u in range(n_occ)])
    groups = {1: g1, 2: g2}

    print("\n  GROUND TRUTH   (the model never sees NOC codes)")
    for d in (1, 2):
        chance = float(np.mean([np.mean(groups[d][warm] == groups[d][int(u)]) for u in cold]))
        lab = "broad category" if d == 1 else "major group, HARDER"
        print(f"    NOC {d}-digit ({lab:<20}) : {len(set(groups[d]))} groups, "
              f"chance P@10 = {chance:.3f}")

    # ---- tune lambda PER MECHANISM on P@10 -----------------------------------
    lam = tune_lambda(mf, warm, warm_df, R_full, n_desc, g2, rng, seed=args.seed)

    # ---- run -----------------------------------------------------------------
    print("  running the experiment ...")
    recs = []
    for strategy in ("core_first", "uniform"):
        for n_rev in args.reveals:
            for u in cold:
                u = int(u)
                cells = cold_df[cold_df[OCC] == u]
                if len(cells) < 3:
                    continue
                prng = np.random.default_rng(args.seed * 100003 + u)
                sel = sample_profile(cells, n_rev, strategy, prng)
                if len(sel) < 2:
                    continue
                S = sel[DSC].to_numpy()
                r_p = sel[RAT].to_numpy().astype(float)

                sc = score_all(mf, S, r_p, warm, R_full, n_desc,
                               lam["mf_latent"], lam["mf_impute"], prng)

                for digits in (1, 2):
                    gg = groups[digits]
                    rel = (gg[warm] == gg[u]).astype(int)
                    if rel.sum() == 0:
                        continue
                    biggest = pd.Series(gg[warm]).value_counts().idxmax()
                    sc["popular_group"] = (gg[warm] == biggest).astype(float)

                    for meth in METHODS:
                        recs.append({"occupation_id": u, "n_revealed": n_rev,
                                     "strategy": strategy, "digits": digits,
                                     "method": meth, "prior": float(rel.mean()),
                                     **rank_metrics(sc[meth], rel)})

    res = pd.DataFrame(recs)
    res.to_csv(RESULTS_DIR / "coldstart_raw.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # TABLES
    # ═══════════════════════════════════════════════════════════════════════
    best_model = best_base = None

    for digits in (1, 2):
        sub = res[(res["digits"] == digits) & (res["strategy"] == "core_first")]
        if sub.empty:
            continue
        chance = sub["prior"].mean()
        lab = "BROAD CATEGORY (1-digit)" if digits == 1 else "MAJOR GROUP (2-digit) -- THE HARD TEST"

        print("\n" + "=" * 100)
        print(f"  P@10   ground truth: NOC {lab}   chance = {chance:.3f}")
        print(f"  elicitation: core_first")
        print("=" * 100)
        hdr = f"  {'|S|':>5} | " + " ".join(f"{m:>16}" for m in METHODS)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for n_rev in args.reveals:
            line = f"  {n_rev:>5} | "
            for m in METHODS:
                v = sub[(sub["n_revealed"] == n_rev) & (sub["method"] == m)]["p@10"]
                line += f"{v.mean():>16.3f} " if len(v) else f"{'-':>16} "
            print(line)

        if digits == 2:
            avg = sub.groupby("method")["p@10"].mean()
            best_model = avg[LEARNED].idxmax()
            best_base = avg[[m for m in NO_LEARNING if m != "random"]].idxmax()
            print(f"\n  Selected ON THE DATA, not assumed:")
            print(f"    strongest LEARNED model     : {best_model}  ({avg[best_model]:.3f} avg)")
            print(f"    strongest NO-LEARNING base  : {best_base}  ({avg[best_base]:.3f} avg)")

    # ═══════════════════════════════════════════════════════════════════════
    # HEADLINE: best model vs best no-learning baseline. PAIRED.
    # ═══════════════════════════════════════════════════════════════════════
    for digits in (1, 2):
        sub = res[(res["digits"] == digits) & (res["strategy"] == "core_first")]
        if sub.empty:
            continue
        print("\n" + "=" * 100)
        print(f"  HEADLINE  [NOC {digits}-digit]:  {best_model}  vs  {best_base}")
        print(f"  Does machine learning beat the best thing you can do WITHOUT it?")
        print(f"  Paired bootstrap over the {args.n_cold} cold occupations, 95% CI.")
        print("=" * 100)
        print(f"  {'|S|':>5} {best_base:>16} {best_model:>16} {'difference':>12} {'95% CI':>20}  verdict")
        print("  " + "-" * 92)

        for n_rev in args.reveals:
            a = sub[(sub["n_revealed"] == n_rev) & (sub["method"] == best_model)] \
                .set_index("occupation_id")["p@10"]
            b = sub[(sub["n_revealed"] == n_rev) & (sub["method"] == best_base)] \
                .set_index("occupation_id")["p@10"]
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            d = (a.loc[common] - b.loc[common]).to_numpy()
            m, lo, hi = boot_ci(d, args.bootstrap, seed=args.seed)
            verdict = ("ML WINS" if lo > 0 else
                       "no-learning WINS" if hi < 0 else
                       "not established")
            print(f"  {n_rev:>5} {b.mean():>16.3f} {a.mean():>16.3f} {m:>+12.3f} "
                  f"  [{lo:>+6.3f}, {hi:>+6.3f}]  {verdict}")

    # ═══════════════════════════════════════════════════════════════════════
    # CONTROLLED PAIR: mf_impute vs raw_correlation  (isolates IMPUTATION)
    # ═══════════════════════════════════════════════════════════════════════
    sub = res[(res["digits"] == 2) & (res["strategy"] == "core_first")]
    print("\n" + "=" * 100)
    print("  CONTROLLED PAIR:  mf_impute  vs  raw_correlation      [NOC 2-digit]")
    print("  Identical metric (Pearson r). Identical ground truth. The ONLY difference:")
    print("  mf_impute compares 181 dimensions with 181-|S| IMPUTED; raw_correlation")
    print("  compares only the |S| revealed. So the gap isolates THE VALUE OF IMPUTATION.")
    print("=" * 100)
    print(f"  {'|S|':>5} {'raw_corr':>10} {'mf_impute':>11} {'difference':>12} {'95% CI':>20}  verdict")
    print("  " + "-" * 84)
    crossover = None
    for n_rev in args.reveals:
        a = sub[(sub["n_revealed"] == n_rev) & (sub["method"] == "mf_impute")] \
            .set_index("occupation_id")["p@10"]
        b = sub[(sub["n_revealed"] == n_rev) & (sub["method"] == "raw_correlation")] \
            .set_index("occupation_id")["p@10"]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        d = (a.loc[common] - b.loc[common]).to_numpy()
        m, lo, hi = boot_ci(d, args.bootstrap, seed=args.seed)
        verdict = ("imputation HELPS" if lo > 0 else
                   "imputation HURTS" if hi < 0 else
                   "not established")
        if hi < 0 and crossover is None:
            crossover = n_rev
        print(f"  {n_rev:>5} {b.mean():>10.3f} {a.mean():>11.3f} {m:>+12.3f} "
              f"  [{lo:>+6.3f}, {hi:>+6.3f}]  {verdict}")
    if crossover:
        print(f"\n  CROSSOVER at ~{crossover} revealed ratings.")
        print("  Below it, imputing the unknown competencies HELPS: the model supplies")
        print("  information the person never gave us. Above it, the person has already told")
        print("  us nearly everything, and a 32-dim reconstruction is a LOSSY stand-in for")
        print("  values we could simply have read off. That is the product spec.")

    # ═══════════════════════════════════════════════════════════════════════
    # ELICITATION ABLATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print(f"  ABLATION: does asking about STRENGTHS beat asking at random?   [{best_model}, 2-digit]")
    print("=" * 100)
    print(f"  {'|S|':>5} {'core_first':>12} {'uniform':>10} {'gain':>8}")
    print("  " + "-" * 40)
    for n_rev in args.reveals:
        base = res[(res["digits"] == 2) & (res["method"] == best_model) &
                   (res["n_revealed"] == n_rev)]
        c = base[base["strategy"] == "core_first"]["p@10"]
        u_ = base[base["strategy"] == "uniform"]["p@10"]
        if len(c) and len(u_):
            print(f"  {n_rev:>5} {c.mean():>12.3f} {u_.mean():>10.3f} {c.mean()-u_.mean():>+8.3f}")
    print("\n  A positive gain means the ELICITATION STRATEGY is worth something on its own:")
    print("  asking 'what are you good at' extracts more signal than a random subset.")
    print("  That is a PRODUCT decision, not a modelling one.")

    # ═══════════════════════════════════════════════════════════════════════
    # FAIRNESS -- on the DEPLOYED mechanism
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print(f"  FAIRNESS AUDIT   [{best_model}, |S|={args.fairness_n}, NOC 2-digit]  -> Dharnesh")
    print("=" * 100)
    print("  TEER = 2nd digit of the NOC = the education tier the occupation requires.")
    print("    0 management  1 university  2 college  3 secondary+training  4 secondary  5 none")
    print()
    print("  RUN THIS ON THE MODEL THAT SHIPS. An earlier audit used a mechanism we later")
    print("  proved degenerate, and it manufactured a harm to low-credential workers that")
    print("  did not exist. Auditing the wrong model is worse than not auditing.")
    print()

    fair = res[(res["digits"] == 2) & (res["method"] == best_model) &
               (res["n_revealed"] == args.fairness_n) &
               (res["strategy"] == "core_first")].copy()
    if not fair.empty:
        fair["teer"] = fair["occupation_id"].map(lambda u: noc_group(noc_of[int(u)], 2)[1])
        print(f"  {'TEER':>5} {'n':>4} {'P@10':>7} {'95% CI':>18}")
        print("  " + "-" * 40)
        rows = []
        for teer, g in fair.groupby("teer"):
            m, lo, hi = boot_ci(g["p@10"].to_numpy(), args.bootstrap, seed=args.seed)
            note = "  (n<10: not interpretable)" if len(g) < 10 else ""
            print(f"  {teer:>5} {len(g):>4} {m:>7.3f}   [{lo:>5.3f}, {hi:>5.3f}]{note}")
            rows.append({"teer": teer, "n": len(g), "p@10": m, "lo": lo, "hi": hi})

        rep = pd.DataFrame(rows)
        big = rep[rep["n"] >= 10]
        if len(big) >= 2:
            gap = big["p@10"].max() - big["p@10"].min()
            rel = gap / max(big["p@10"].max(), 1e-9)
            overlap = big["lo"].max() <= big["hi"].min()
            worst = big.loc[big["p@10"].idxmin()]
            print(f"\n  Excluding tiers with n < 10:")
            print(f"    absolute disparity : {gap:.3f}")
            print(f"    relative disparity : {rel:.1%}   (proposal target: < 10%)")
            print(f"    CIs overlap        : {'YES' if overlap else 'NO'}")
            print(f"    worst-served tier  : TEER {worst['teer']}  "
                  f"P@10 {worst['p@10']:.3f}  (n={int(worst['n'])})")
            print()
            if not overlap:
                print("  => A REAL DISPARITY. The confidence intervals do not overlap.")
                print("     This is a finding with stakes: the affected workers have the least")
                print("     room to absorb bad career advice. It must be characterised, not")
                print("     buried -- and the mechanism behind it must be explained, not just")
                print("     the magnitude.")
            else:
                print("  => NOT STATISTICALLY DETECTABLE. The intervals overlap, so the spread")
                print("     is consistent with sampling noise. But with 10-30 occupations per")
                print("     tier we cannot RULE OUT a moderate effect. The honest phrasing is")
                print("     'not detected', NOT 'absent'.")
        rep.to_csv(RESULTS_DIR / "coldstart_fairness_by_teer.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, digits in zip(axes, (1, 2)):
        sub = res[(res["digits"] == digits) & (res["strategy"] == "core_first")]
        for m in METHODS:
            g = sub[sub["method"] == m].groupby("n_revealed")["p@10"].mean()
            if g.empty:
                continue
            bold = m in (best_model, best_base)
            ax.plot(g.index, g.values, marker="o", ms=4,
                    lw=2.6 if bold else 1.2,
                    ls="-" if m in LEARNED else "--",
                    alpha=1.0 if bold else 0.6, label=m)
        ax.axhline(sub["prior"].mean(), color="crimson", ls=":", lw=1, label="chance")
        ax.set_xscale("log")
        ax.set_xticks(args.reveals)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("competencies revealed  |S|")
        ax.set_ylabel("P@10")
        ax.set_title(f"NOC {digits}-digit ({'broad' if digits == 1 else 'major group, harder'})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Cold start: 90 occupations the model has NEVER seen.  Bold = headline pair.",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / "coldstart.png"
    fig.savefig(out, dpi=150)

    summ = (res.groupby(["digits", "strategy", "n_revealed", "method"])
              [["p@5", "p@10", "mrr", "prior"]].mean().reset_index())
    summ.to_csv(RESULTS_DIR / "coldstart_summary.csv", index=False)

    print(f"\n  figure  -> {out}")
    print(f"  results -> {RESULTS_DIR / 'coldstart_summary.csv'}")
    print(f"\n  lambda: mf_latent={lam['mf_latent']}  mf_impute={lam['mf_impute']}")


if __name__ == "__main__":
    main()
