"""
================================================================================
SkillBridge -- Trust Layer   (Dharnesh Somasundaram)
CSC 503 Data Mining, Summer 2026
================================================================================

WHAT THIS PHASE DOES (the one-paragraph summary)
------------------------------------------------
The other five members build models that *score* well. This phase asks a
different question: can those models be *trusted* by a real person making a
career decision? It takes the three shipped models -- the Labour Shortage
Classifier and the salary + posting-volume regressors -- and runs three audits
on them:

    (A) CALIBRATION   Are the classifier's probabilities honest? When it says
                      "70% chance of Shortage", is it right ~70% of the time?
                      We measure that (ECE, Brier, a reliability diagram) and,
                      if it is off, fit a post-hoc calibrator to fix it.

    (B) EXPLAINABILITY  What is each model actually using to decide? We compute
                        SHAP (Shapley) values so every prediction can be
                        defended with "these features drove it", not asserted.

    (C) FAIRNESS       Does prediction quality hold up across the groups a user
                       belongs to -- NOC TEER level (education tier), region
                       (province), and occupation popularity -- or is the system
                       quietly better for some groups than others? Every group
                       metric carries a bootstrap 95% interval, and a gap is only
                       called "real" when two groups' intervals do not overlap.

WHERE RESULTS GO
----------------
Yes -- every numeric result is persisted as JSON (one file per audit), and the
per-group tables are also written as CSV; the plots are PNG. See save_result()
below and the manifest printed at the end of a run. Nothing is left only on the
terminal.

    results/trust_calibration__shortage_classifier.json   calibration numbers
    results/trust_shap__*.json                            SHAP importances
    results/trust_fairness__*.json                         disparity summaries
    results/trust_disparity_*.csv                          per-group tables
    figures/trust_reliability_shortage.png                reliability diagram
    figures/trust_shap_*.png                              SHAP bar charts
    figures/trust_disparity_*.png                         grouped MAE/accuracy + CIs

HOW TO RUN
----------
    pip install numpy pandas scikit-learn matplotlib shap
    python scripts/14_trust_layer.py

It reads the team's cleaned data from datasets/clean/. If that folder is not
present (e.g. someone cloned the code without the data), it generates a small
synthetic stand-in with the same schema so the pipeline still runs; on a machine
that already has datasets/clean/, that generator never fires and never touches
your real data.

DESIGN NOTE
-----------
This is a single self-contained file on purpose: it depends on nothing else in
the repo, so a teammate can run the whole trust layer with one command. The
model-rebuild functions deliberately mirror 02_shortage_classifier.py and
03_salary_and_volume_model.py so the audit measures the *same* models that ship.
================================================================================
"""

from __future__ import annotations
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless backend: write figures to disk, never open a GUI window
import matplotlib.pyplot as plt

import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, f1_score


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIG  --  paths, seed, and the exact model configs being audited
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42                                    # one seed everywhere; two people who disagree check this first

# Resolve the project root whether this file sits at the repo root or in scripts/.
# If it lives in a folder literally named "scripts", step up one level so that
# datasets/, results/, and figures/ resolve to the project root next to the data.
_HERE = Path(__file__).resolve().parent
BASE = _HERE.parent if _HERE.name == "scripts" else _HERE
CLEAN_DIR = BASE / "datasets" / "clean"      # the team's cleaned data lives here
RESULTS_DIR = BASE / "results"               # JSON + CSV outputs
FIGURES_DIR = BASE / "figures"               # PNG outputs
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Targets the proposal set for this phase.
ECE_TARGET = 0.05                            # calibration: expected calibration error should fall below this
DISPARITY_TARGET = 0.10                      # fairness: relative gap between best/worst group should stay under 10%

# The shortage classifier's probabilities are ordered by these class names.
# We pin the order so every proba matrix has the same column meaning across folds.
SHORTAGE_CLASSES = ["Balance", "Shortage", "Surplus"]

# The FINAL classifier config the owner (Irai) shipped: round-2 tuned RF, 12
# features. Reproduced here so the audit measures the deployed model, not a proxy.
SHORTAGE_FEATURES = [
    "posting", "posting_log", "avg_rating", "core_fraction", "employment_growth",
    "salary_mean", "salary_std", "oa_abilities", "oa_knowledge",
    "oa_personal_attributes", "oa_skills", "oa_work_activities",
]
SHORTAGE_PARAMS = {"max_depth": 6, "min_samples_leaf": 1, "n_estimators": 400}

# COPS ships 5 outlook labels; the classifier collapses them to 3. Same map here.
LABEL_MAP = {
    "Strong risk of Shortage": "Shortage", "Moderate risk of Shortage": "Shortage",
    "Balance": "Balance",
    "Moderate risk of Surplus": "Surplus", "Strong risk of Surplus": "Surplus",
}


def set_all_seeds(seed=SEED):
    """Pin every RNG this script can touch, so results are reproducible run-to-run."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def TEER_FROM_NOC(noc) -> str:
    """NOC TEER level (the education tier an occupation needs) = 2nd digit of the 5-digit code."""
    return str(noc).zfill(5)[1]


# ── console output helpers: give every audit its own clearly-separated block ──
# `banner` opens a section, `footer` closes it, so each output body is visually
# fenced off in the terminal instead of running together.
_WIDTH = 78

def banner(title: str):
    print("\n\n" + "=" * _WIDTH)
    print(f"  {title}")
    print("=" * _WIDTH)

def footer():
    print("  " + "-" * (_WIDTH - 2) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPLIT PROTOCOLS  --  how data is divided for honest, leakage-free evaluation
# ══════════════════════════════════════════════════════════════════════════════

def stratified_folds(X, y, n_folds=5, seed=SEED):
    """
    Stratified 5-fold splitter. Stratification keeps the class ratio (including
    the rare Surplus class) roughly constant in every fold, which matters a lot
    when a class has only ~17 examples nationwide.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from skf.split(X, y)


def temporal_split(df, date_col="posting_date", cutoff="2026-01-01"):
    """
    Split job postings by DATE, not randomly. A random split would put the same
    job (reposted across months) in both train and test and inflate the score;
    training on the past and testing on the future is the honest setup.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    cut = pd.Timestamp(cutoff)
    train = d[d[date_col] < cut]
    test = d[d[date_col] >= cut]
    print(f"  [split] temporal @ {cutoff}: train={len(train):,}, test={len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALIBRATION METRICS  --  are the predicted probabilities honest?
# ══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    ECE = Σ_b (|B_b|/n) · |accuracy(B_b) - confidence(B_b)|.

    Plain English: bucket predictions by how confident they were, and in each
    bucket compare "how confident the model said it was" against "how often it
    was actually right". A perfectly calibrated model has ECE = 0. Target < 0.05.
    """
    conf = np.max(y_proba, axis=1)                 # the model's confidence = its top class probability
    pred = np.argmax(y_proba, axis=1)              # the predicted class
    acc = (pred == y_true).astype(float)           # 1 if that prediction was correct
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)             # rows whose confidence lands in this bin
        if m.sum():
            ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, y_proba, n_classes):
    """
    Brier score = mean squared error between the predicted probability vector and
    the one-hot truth. Lower is better; rewards probabilities that are both
    confident AND correct.
    """
    onehot = np.zeros((len(y_true), n_classes))
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def reliability_curve(y_true, y_proba, n_bins=10):
    """
    The data behind the reliability diagram: for each confidence bin return
    (mean confidence, empirical accuracy, count). Points on the diagonal =
    perfectly calibrated; above = under-confident; below = over-confident.
    """
    conf = np.max(y_proba, axis=1)
    pred = np.argmax(y_proba, axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    c, a, k = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:                            # empty bin -> NaN so the plot leaves a gap
            c.append(np.nan); a.append(np.nan); k.append(0)
        else:
            c.append(conf[m].mean()); a.append(acc[m].mean()); k.append(int(m.sum()))
    return np.array(c), np.array(a), np.array(k)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRUST HELPERS  --  bootstrap intervals, grouped audits, disparity read-out
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=SEED):
    """
    Percentile bootstrap -> (point_estimate, lo, hi) for `stat` over `values`.

    Why: group sizes here are small (some TEER tiers hold ~25 occupations), so a
    single group MAE or accuracy is noisy. Resampling with replacement 2000 times
    tells us how much that number would wobble, i.e. an honest 95% interval.
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(values)), float(lo), float(hi))


def grouped_regression_audit(df, group_col, y_true_col, y_pred_col, min_n=20, n_boot=2000):
    """
    Per-group MAE (with bootstrap 95% CI) and R^2 for a regressor. The CI is
    bootstrapped on the per-row absolute errors -- the exact quantity MAE
    averages -- so it answers "how much would this group's MAE move on a
    resampled test set of the same size". Groups smaller than `min_n` are skipped.
    """
    out = []
    for g, grp in df.groupby(group_col, observed=True):
        if len(grp) < min_n:                        # too few rows to say anything stable
            continue
        yt = grp[y_true_col].to_numpy(float)
        yp = grp[y_pred_col].to_numpy(float)
        ae = np.abs(yt - yp)                         # per-row absolute error
        mae, lo, hi = bootstrap_ci(ae, np.mean, n_boot=n_boot)
        # R^2 is undefined if the group's true values have no spread
        r2 = r2_score(yt, yp) if len(grp) > 2 and np.ptp(yt) > 0 else np.nan
        out.append({"group": g, "n": int(len(grp)), "mae": mae,
                    "mae_lo": lo, "mae_hi": hi, "r2": float(r2) if r2 == r2 else np.nan})
    return pd.DataFrame(out).sort_values("mae", ascending=False).reset_index(drop=True)


def grouped_classification_audit(df, group_col, y_true_col, y_pred_col, min_n=8, n_boot=2000):
    """
    Per-group accuracy (with bootstrap 95% CI) and macro-F1 for a classifier.
    Accuracy is bootstrapped on the per-row correctness indicator; macro-F1 is a
    point value only (with a handful of rows per group, bootstrapping F1 is too
    unstable to quote).
    """
    out = []
    for g, grp in df.groupby(group_col, observed=True):
        if len(grp) < min_n:
            continue
        yt = grp[y_true_col].to_numpy()
        yp = grp[y_pred_col].to_numpy()
        correct = (yt == yp).astype(float)          # 1 where the prediction matched
        acc, lo, hi = bootstrap_ci(correct, np.mean, n_boot=n_boot)
        macro = f1_score(yt, yp, average="macro", zero_division=0)
        out.append({"group": g, "n": int(len(grp)), "accuracy": acc,
                    "acc_lo": lo, "acc_hi": hi, "macro_f1": float(macro)})
    return pd.DataFrame(out).sort_values("accuracy").reset_index(drop=True)


def disparity_summary(audit, metric_col, lo_col=None, hi_col=None,
                      higher_is_better=True, abs_floor=0.0):
    """
    Collapse a grouped audit into a single disparity read-out: best group, worst
    group, the absolute gap, the RELATIVE gap (gap / best -- what the 10% target
    is stated against), and whether the best/worst intervals overlap (a non-
    overlap is the bar for calling a gap "real").

    IMPORTANT audit fix -- relative gaps are unreliable when the metric is near
    zero. Example: posting-volume MAE is ~1-6 postings, so a 5-vs-1 gap reads as
    "400%" yet is immaterial. `abs_floor` marks the relative number unreliable
    when the best value is below a metric-specific floor, so the reader is told
    to trust the ABSOLUTE gap instead of a misleading percentage.
    """
    a = audit.dropna(subset=[metric_col]).copy()
    if len(a) < 2:
        return {"disparity_relative": np.nan, "note": "fewer than 2 usable groups"}

    # "best"/"worst" depends on whether high is good (accuracy) or bad (MAE)
    if higher_is_better:
        best, worst = a.loc[a[metric_col].idxmax()], a.loc[a[metric_col].idxmin()]
    else:
        best, worst = a.loc[a[metric_col].idxmin()], a.loc[a[metric_col].idxmax()]

    gap = abs(best[metric_col] - worst[metric_col])
    denom = abs(best[metric_col]) if best[metric_col] != 0 else np.nan
    rel = gap / denom if denom and denom == denom else np.nan

    # is the relative percentage trustworthy, or is the base metric too small?
    relative_reliable = bool(abs(best[metric_col]) >= abs_floor) if abs_floor else True

    # do the best and worst groups' confidence intervals overlap?
    overlap = None
    if lo_col and hi_col:
        overlap = not (best[lo_col] > worst[hi_col] or worst[lo_col] > best[hi_col])

    return {
        "best_group": str(best["group"]), "best_value": float(best[metric_col]),
        "worst_group": str(worst["group"]), "worst_value": float(worst[metric_col]),
        "absolute_gap": float(gap),
        "disparity_relative": float(rel) if rel == rel else np.nan,
        "relative_reliable": relative_reliable,
        "intervals_overlap": overlap,
        "gap_is_significant": (overlap is False),
    }


def popularity_tier(counts, labels=("low", "medium", "high")):
    """
    Bucket occupations into demand tiers by posting count (tertiles). 'Occupation
    popularity' is one of the three fairness axes the proposal names.
    """
    try:
        return pd.qcut(counts.rank(method="first"), q=len(labels), labels=list(labels))
    except ValueError:                              # too few distinct values for tertiles
        return pd.cut(counts, bins=len(labels), labels=list(labels))


def teer_name(teer):
    """Human-readable label for a NOC TEER digit, used on chart axes and tables."""
    return {
        "0": "TEER 0 (management)", "1": "TEER 1 (university degree)",
        "2": "TEER 2 (college / apprenticeship 2y+)",
        "3": "TEER 3 (college / apprenticeship <2y)",
        "4": "TEER 4 (secondary + training)", "5": "TEER 5 (no formal education)",
    }.get(str(teer), f"TEER {teer}")


# results/ manifest -- every path we persist gets recorded here and printed at the end
_SAVED = {"json": [], "csv": [], "png": []}


def save_result(results, component, model_name, extra=None):
    """
    Persist one audit as JSON (one file per component+model). This is the answer
    to "is the result stored as JSON?" -- yes, here. Report tables can later be
    rebuilt just by reading these files, so no number is ever copy-pasted from a
    terminal. Returns the path written.
    """
    payload = {"component": component, "model": model_name, "metrics": results}
    if extra:
        payload["extra"] = extra
    safe = model_name.lower().replace(" ", "_").replace("/", "-")
    path = RESULTS_DIR / f"{component}__{safe}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    _SAVED["json"].append(path.name)
    return path


def save_csv(df, name):
    """Persist a per-group audit table as CSV alongside the JSON summary."""
    df.to_csv(RESULTS_DIR / name, index=False)
    _SAVED["csv"].append(name)


def save_fig(fig, name):
    """Persist a figure as PNG and record it in the manifest."""
    fig.savefig(FIGURES_DIR / name, dpi=130)
    plt.close(fig)
    _SAVED["png"].append(name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DEMO DATA  --  only used if datasets/clean/ is absent (never overwrites data)
# ══════════════════════════════════════════════════════════════════════════════

def ensure_data():
    """
    Make sure the cleaned inputs exist. On a machine that already has
    datasets/clean/ (the normal case), this just confirms and returns -- it does
    NOT touch or regenerate anything. Only when those files are missing does it
    synthesise a small stand-in with the same schema, so the pipeline is runnable
    on a bare clone. The stand-in is demonstration data, not a project result.
    """
    need = ["cops_projections_clean.csv", "jobbank_clean.csv", "oasis_descriptors_long.csv"]
    if all((CLEAN_DIR / f).exists() for f in need):
        print(f"  using existing cleaned data in {CLEAN_DIR}")
        return

    print(f"  datasets/clean not found -> generating a synthetic demo stand-in in {CLEAN_DIR}")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- reference vocabularies (kept close to the real Job Bank / COPS values) ---
    PROV = ["Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba",
            "Saskatchewan", "Nova Scotia", "New Brunswick",
            "Newfoundland and Labrador", "Prince Edward Island"]
    PW = np.array([.38, .22, .14, .11, .04, .03, .03, .025, .02, .005]); PW /= PW.sum()
    PMULT = dict(zip(PROV, [1.03, .96, 1.02, 1.08, .93, .95, .90, .89, .91, .87]))  # regional pay nudge
    EDU = ["No formal education", "High school", "College/CEGEP",
           "Bachelor's degree", "Master's degree", "PhD"]
    EDU_ADD = dict(zip(EDU, [-9000, -4000, 0, 9000, 16000, 24000]))                  # $ added by education
    EXP = ["No experience", "1 to 7 months", "1 year to less than 2 years",
           "2 years to less than 3 years", "3 years to less than 5 years", "5 years or more"]
    EXP_ADD = dict(zip(EXP, [-5000, -3000, 0, 3000, 6000, 11000]))                   # $ added by experience
    IND = ["Health care", "Educational services", "Manufacturing", "Retail trade",
           "Construction", "Public administration", "Professional services",
           "Finance and insurance", "Transportation", "Accommodation and food services"]
    TEER_BASE = {0: 98000, 1: 88000, 2: 72000, 3: 60000, 4: 50000, 5: 43000}         # base salary by tier
    TEER_W = np.array([.06, .20, .24, .20, .18, .12])                                # tier prevalence

    # --- occupations: 500 unique 5-digit NOC codes, each with a latent skill level ---
    N = 500
    codes, teers, seen = [], [], set()
    while len(codes) < N:
        b, t, rest = rng.integers(0, 10), int(rng.choice(range(6), p=TEER_W)), rng.integers(0, 1000)
        code = f"{b}{t}{rest:03d}"                   # 1st digit broad category, 2nd digit = TEER
        if code in seen:
            continue
        seen.add(code); codes.append(code); teers.append(t)
    occ = pd.DataFrame({"noc_code": codes, "teer": teers})
    occ["latent"] = np.clip(0.75 - 0.11 * occ["teer"] + rng.normal(0, 0.12, N), 0.05, 0.98)
    occ["noc21_name"] = ["Occupation " + c for c in occ["noc_code"]]
    occ.to_csv(CLEAN_DIR / "noc_lookup.csv", index=False)

    # --- OaSIS: 181 descriptors across 5 categories, 0-5 ratings per occupation ---
    CATS = {"Skills": 33, "Abilities": 52, "Knowledge": 44,
            "Work Activities": 39, "Personal Attributes": 13}
    COFF = {"Skills": 0.0, "Abilities": 0.15, "Knowledge": -0.10,       # per-category offset so an occupation
            "Work Activities": 0.05, "Personal Attributes": -0.05}      # can be high on one, low on another
    rows = []
    for _, o in occ.iterrows():
        for cat, k in CATS.items():
            mu = np.clip(o["latent"] + COFF[cat], 0.02, 0.98)
            for i in range(k):
                r = int(np.clip(round(rng.normal(5 * mu, 1.0)), 0, 5))
                rows.append((o["noc_code"], cat, f"{cat[:2].lower()}_{i}", r))
    pd.DataFrame(rows, columns=["noc_code", "category", "descriptor", "rating"]).to_csv(
        CLEAN_DIR / "oasis_descriptors_long.csv", index=False)

    # --- COPS: 3-class outlook, marginals matched to the report (365/103/17) ---
    occ["employment_growth"] = rng.normal(0.6, 1.4, N)
    score = 0.9 * occ["employment_growth"] + 1.2 * (occ["latent"] - 0.5) + rng.normal(0, 0.7, N)
    hi = np.quantile(score, 1 - 103 / N)            # top ~103 -> Shortage
    lo = np.quantile(score, 17 / N)                 # bottom ~17 -> Surplus
    five = []
    for s in score:
        if s >= hi:
            five.append("Strong risk of Shortage" if s > hi + 0.4 else "Moderate risk of Shortage")
        elif s <= lo:
            five.append("Strong risk of Surplus" if s < lo - 0.4 else "Moderate risk of Surplus")
        else:
            five.append("Balance")
    cops = pd.DataFrame({"noc_code": occ["noc_code"], "future_conditions": five,
                         "employment_growth": occ["employment_growth"].round(3)})
    # add junk rows (aggregate codes + unlabeled) exactly like the real COPS export,
    # so the classifier's "drop non-5-digit / unlabeled rows" cleaning has something to do
    junk = [{"noc_code": f"NOC{i}_0", "future_conditions": "Balance", "employment_growth": 0.0}
            for i in range(15)]
    junk += [{"noc_code": occ.iloc[i]["noc_code"], "future_conditions": np.nan,
              "employment_growth": np.nan} for i in range(11)]
    pd.concat([cops, pd.DataFrame(junk)], ignore_index=True).to_csv(
        CLEAN_DIR / "cops_projections_clean.csv", index=False)

    # --- Job Bank: ~180k postings across 4 monthly exports, with ~30% reposts ---
    MONTHS = [("november2025", "2025-11"), ("december2025", "2025-12"),
              ("jan2026", "2026-01"), ("feb2026", "2026-02")]
    vol = (rng.integers(30, 900, N) * (0.4 + occ["latent"].values)).astype(int)   # posting volume per occupation
    base = []
    for idx, o in occ.reset_index(drop=True).iterrows():
        n_post = max(5, int(vol[idx] * 0.55))       # unique base postings before reposting
        provs = rng.choice(PROV, size=n_post, p=PW)
        edus = rng.choice(EDU, size=n_post, p=[.05, .18, .30, .30, .12, .05])
        exps = rng.choice(EXP, size=n_post)
        tb = TEER_BASE[int(o["teer"])]
        for j in range(n_post):
            p, e, x = provs[j], edus[j], exps[j]
            # salary = tier base * region * education/experience add-ons + occupation skill + noise
            sal = (tb * PMULT[p] + EDU_ADD[e] + EXP_ADD[x]
                   + 20000 * (o["latent"] - 0.5) + rng.normal(0, 6500))
            base.append({
                "noc21_code": o["noc_code"], "noc21_name": o["noc21_name"],
                "job_title": f"{o['noc21_name']} role {j % 40}",
                "province": p, "city": f"City{rng.integers(0, 350)}",
                "education": e, "experience": x,
                "employment_type": rng.choice(["Full time", "Part time"], p=[.85, .15]),
                "employment_term": rng.choice(["Permanent", "Temporary", "Casual", "Seasonal"],
                                              p=[.7, .18, .08, .04]),
                "industry": rng.choice(IND),
                "salary_annual": round(float(np.clip(sal, 22000, 320000)), 0),
                "vacancies": int(rng.choice([1, 1, 1, 2, 3], p=[.7, .1, .1, .06, .04])),
            })
    base = pd.DataFrame(base)
    # ~59% of education/experience/industry is missing in the real data -> blank it out
    for col in ["education", "experience", "employment_type", "employment_term", "industry"]:
        base.loc[rng.random(len(base)) < 0.59, col] = np.nan
    # repost each base posting into 1-3 consecutive months (this is what makes dedup matter)
    first = rng.integers(0, 4, len(base))
    recs = []
    for i, row in enumerate(base.itertuples(index=False)):
        d = row._asdict()
        for m in range(first[i], min(first[i] + rng.integers(1, 4), 4)):
            mk, ms = MONTHS[m]
            r = dict(d); r["source_month"] = mk
            r["posting_date"] = f"{ms}-{rng.integers(1, 28):02d}"
            recs.append(r)
    pd.DataFrame(recs).sample(frac=1.0, random_state=SEED).reset_index(drop=True).to_csv(
        CLEAN_DIR / "jobbank_clean.csv", index=False)
    print(f"    wrote demo cops / oasis / jobbank / noc_lookup to {CLEAN_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. REBUILD MODELS  --  reproduce the exact models the other members ship
# ══════════════════════════════════════════════════════════════════════════════

def build_shortage_frame():
    """
    Reproduce Irai's 12-feature COPS table (mirrors 02_shortage_classifier.py).
    We rebuild it here so the audit runs on the same features the model shipped
    with. Returns one row per usable occupation with all 12 features + label + teer.
    """
    cops = pd.read_csv(CLEAN_DIR / "cops_projections_clean.csv", dtype={"noc_code": str})
    cops = cops[cops["noc_code"].str.len() == 5].copy()       # drop aggregate rows like "NOC1_0"
    cops["new_label"] = cops["future_conditions"].map(LABEL_MAP)

    # feature: posting count per occupation (from Job Bank)
    jobs = pd.read_csv(CLEAN_DIR / "jobbank_clean.csv")
    jobs["noc21_code"] = jobs["noc21_code"].astype(str)
    posting = jobs.groupby("noc21_code").size()
    posting.index = posting.index.astype(str).str.zfill(5)
    cops["posting"] = cops["noc_code"].map(posting).fillna(0)
    cops = cops.dropna(subset=["new_label"]).copy()           # drop unlabeled rows

    # features from OaSIS competency ratings
    oasis = pd.read_csv(CLEAN_DIR / "oasis_descriptors_long.csv", dtype={"noc_code": str})
    cops["avg_rating"] = cops["noc_code"].map(oasis.groupby("noc_code")["rating"].mean())
    cops["core_fraction"] = cops["noc_code"].map(
        oasis.groupby("noc_code")["rating"].apply(lambda r: (r >= 4).mean()))   # share of "core" competencies
    cops["employment_growth"] = cops["employment_growth"].fillna(0)

    # features: salary mean/std per occupation (from Job Bank)
    sal = jobs.groupby("noc21_code")["salary_annual"].agg(["mean", "std"])
    sal.index = sal.index.astype(str).str.zfill(5)
    cops["salary_mean"] = cops["noc_code"].map(sal["mean"]).fillna(
        cops["noc_code"].map(sal["mean"]).median())
    cops["salary_std"] = cops["noc_code"].map(sal["std"]).fillna(0)

    # features: OaSIS split into its 5 categories (the single biggest win in Irai's tuning)
    cat = oasis.groupby(["noc_code", "category"])["rating"].mean().unstack("category")
    cat.columns = ["oa_" + c.replace(" ", "_").lower() for c in cat.columns]
    cops = cops.join(cat, on="noc_code")
    cops[list(cat.columns)] = cops[list(cat.columns)].fillna(cops[list(cat.columns)].median())

    cops["posting_log"] = np.log1p(cops["posting"])           # posting count is heavily skewed
    cops["teer"] = cops["noc_code"].apply(TEER_FROM_NOC)      # kept for the fairness audit
    return cops


def build_salary_model():
    """
    Reproduce Manivannan's salary regressor (mirrors 03). Returns the fitted
    pipeline, the test feature matrix, the ordinal-encoded column list (for SHAP),
    and an audit frame (true, pred, teer, province, popularity) for the fairness step.
    """
    # dtype={"noc21_code": str} is load-bearing: without it, pandas infers this
    # 5-digit code column as int64 and SILENTLY drops leading zeros (verified:
    # ~9% of rows on the real data). TEER_FROM_NOC() re-pads via zfill(5) so it
    # would still resolve correctly either way, but the raw code is also used
    # directly as a categorical feature below, so it must stay a true string.
    df = pd.read_csv(CLEAN_DIR / "jobbank_clean.csv", dtype={"noc21_code": str})
    df = df[df["salary_annual"].notna()]
    df = df[(df["salary_annual"] >= 10_000) & (df["salary_annual"] <= 500_000)]   # drop garbage salaries
    for c in ["education", "experience", "employment_type", "employment_term", "industry"]:
        df[c] = df[c].fillna("Unknown")             # keep rows; ~59% of these fields are blank
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    # dedup reposts BEFORE splitting, or the same job leaks into train and test
    df = df.sort_values("posting_date").drop_duplicates(
        subset=["job_title", "noc21_code", "city", "salary_annual"], keep="first")

    train, test = temporal_split(df)                # train on the past, test on the future
    cf = train["city"].value_counts()               # frequency-encode city (3k+ unique values)
    train, test = train.copy(), test.copy()
    train["city_freq"] = train["city"].map(cf).fillna(0)
    test["city_freq"] = test["city"].map(cf).fillna(0)

    feats = ["noc21_code", "province", "education", "experience",
             "employment_type", "employment_term", "industry", "city_freq"]
    ordinal = [c for c in feats if c != "city_freq"]   # everything except the numeric city_freq
    # ordinal-encode categoricals (trees handle this far better than one-hot here)
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal),
        ("num", "passthrough", ["city_freq"])])
    model = Pipeline([("prep", prep), ("model", RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=SEED, n_jobs=-1))])
    model.fit(train[feats], train["salary_annual"])
    preds = model.predict(test[feats])
    print(f"  salary RF: MAE ${mean_absolute_error(test['salary_annual'], preds):,.0f}  "
          f"R2 {r2_score(test['salary_annual'], preds):.3f}")

    # occupation popularity from TRAINING postings only (never from the test set)
    tc = train["noc21_code"].value_counts()
    pop_map = dict(zip(tc.index, popularity_tier(tc)))
    aud = pd.DataFrame({
        "noc21_code": test["noc21_code"].values,
        "teer": test["noc21_code"].apply(TEER_FROM_NOC).values,
        "province": test["province"].values,
        "true": test["salary_annual"].values, "pred": preds})
    aud["pop_tier"] = aud["noc21_code"].map(pop_map).astype(object).fillna("low")
    return model, test[feats], ordinal, aud


def build_volume_model():
    """
    Reproduce Manivannan's posting-volume regressor (mirrors 03 Part B): predict
    February posting counts per (occupation, province) from Nov/Dec/Jan counts.
    Returns an audit frame for the fairness step.
    """
    df = pd.read_csv(CLEAN_DIR / "jobbank_clean.csv", dtype={"noc21_code": str})
    # count postings per (occupation, province, month), then pivot months into columns
    counts = df.groupby(["noc21_code", "province", "source_month"]).size().reset_index(name="count")
    pivot = counts.pivot_table(index=["noc21_code", "province"], columns="source_month",
                               values="count", fill_value=0).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"november2025": "nov_count", "december2025": "dec_count",
                                  "jan2026": "jan_count", "feb2026": "feb_count"})
    for c in ["nov_count", "dec_count", "jan_count", "feb_count"]:
        if c not in pivot.columns:
            pivot[c] = 0
    feats = ["noc21_code", "province", "nov_count", "dec_count", "jan_count"]
    # random split across combos is fine here: each combo carries its own time history in the lag features
    Xtr, Xte, ytr, yte = train_test_split(pivot[feats], pivot["feb_count"],
                                          test_size=0.2, random_state=SEED)
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         ["noc21_code", "province"]),
        ("num", "passthrough", ["nov_count", "dec_count", "jan_count"])])
    model = Pipeline([("prep", prep), ("model", RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=SEED, n_jobs=-1))])
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    print(f"  volume RF: MAE {mean_absolute_error(yte, preds):.2f}  R2 {r2_score(yte, preds):.3f}")

    tot = pivot.groupby("noc21_code")["jan_count"].sum()
    pop_map = dict(zip(tot.index, popularity_tier(tot)))
    aud = pd.DataFrame({
        "noc21_code": Xte["noc21_code"].values,
        "teer": Xte["noc21_code"].apply(TEER_FROM_NOC).values,
        "province": Xte["province"].values, "true": yte.values, "pred": preds})
    aud["pop_tier"] = aud["noc21_code"].map(pop_map).astype(object).fillna("low")
    return aud


# ══════════════════════════════════════════════════════════════════════════════
# 6. PART A  --  CALIBRATION of the shortage classifier
# ══════════════════════════════════════════════════════════════════════════════

def part_a_calibration(cops):
    banner("PART A -- Calibration of the Labour Shortage Classifier")
    X = cops[SHORTAGE_FEATURES].astype("float64")
    y = cops["new_label"].astype(str).to_numpy()
    y_idx = np.array([SHORTAGE_CLASSES.index(v) for v in y])   # integer labels aligned to proba columns

    def oof_proba(make):
        """
        Out-of-fold probabilities: for each fold, fit on the other 4 and predict
        this one, so every occupation gets a probability from a model that never
        saw it. Columns are realigned to SHORTAGE_CLASSES in case a fold is
        missing a class.
        """
        proba = np.zeros((len(X), len(SHORTAGE_CLASSES)))
        for tr, te in stratified_folds(X, y):
            est = make(); est.fit(X.iloc[tr], y[tr])
            p = est.predict_proba(X.iloc[te])
            col = {c: i for i, c in enumerate(est.classes_)}
            al = np.zeros((len(te), len(SHORTAGE_CLASSES)))
            for j, c in enumerate(SHORTAGE_CLASSES):
                if c in col:
                    al[:, j] = p[:, col[c]]
            proba[te] = al
        return proba

    # the shipped model, plus two standard post-hoc calibrators wrapping it
    base = lambda: RandomForestClassifier(random_state=SEED, class_weight="balanced", **SHORTAGE_PARAMS)
    proba_raw = oof_proba(base)
    ece_raw = expected_calibration_error(y_idx, proba_raw)
    brier_raw = multiclass_brier(y_idx, proba_raw, len(SHORTAGE_CLASSES))

    fixes = {}
    for method in ("sigmoid", "isotonic"):          # Platt scaling vs isotonic regression
        proba = oof_proba(lambda m=method: CalibratedClassifierCV(base(), method=m, cv=3))
        fixes[method] = {"proba": proba,
                         "ece": expected_calibration_error(y_idx, proba),
                         "brier": multiclass_brier(y_idx, proba, len(SHORTAGE_CLASSES))}
    best = min(fixes, key=lambda m: fixes[m]["ece"])   # keep whichever calibrator lowers ECE most

    # ---- report ----
    print(f"  raw model            ECE {ece_raw:.4f}   Brier {brier_raw:.4f}")
    for m, d in fixes.items():
        print(f"  + {m:<8} calib.   ECE {d['ece']:.4f}   Brier {d['brier']:.4f}"
              f"{'  <- chosen' if m == best else ''}")
    print(f"  ECE target {ECE_TARGET}: "
          f"{'MET' if fixes[best]['ece'] < ECE_TARGET else 'NOT met'} after calibration")

    # ---- reliability diagram: raw vs calibrated ----
    cr, ar, _ = reliability_curve(y_idx, proba_raw)
    cc, ac, _ = reliability_curve(y_idx, fixes[best]["proba"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(cr, ar, "o-", color="#c0392b", label=f"raw (ECE {ece_raw:.3f})")
    ax.plot(cc, ac, "s-", color="#27ae60", label=f"{best} calibrated (ECE {fixes[best]['ece']:.3f})")
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Shortage classifier -- reliability diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); save_fig(fig, "trust_reliability_shortage.png")

    # ---- persist as JSON ----
    save_result({"ece_raw": ece_raw, "brier_raw": brier_raw,
                 "ece_calibrated": fixes[best]["ece"], "brier_calibrated": fixes[best]["brier"],
                 "best_method": best, "ece_target": ECE_TARGET,
                 "target_met": bool(fixes[best]["ece"] < ECE_TARGET)},
                "trust_calibration", "shortage_classifier",
                extra={"note": "out-of-fold, 5-fold stratified, SEED=42"})
    print("  saved: figures/trust_reliability_shortage.png, "
          "results/trust_calibration__shortage_classifier.json")
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# 7. PART B  --  SHAP explainability
# ══════════════════════════════════════════════════════════════════════════════

def _shap_bar(mean_abs, names, title, fname, color="#2c3e50"):
    """Horizontal bar chart of mean(|SHAP|) per feature -- global importance."""
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.42 * len(names))))
    ax.barh(np.array(names)[order], np.array(mean_abs)[order], color=color)
    ax.set_xlabel("mean(|SHAP value|)  --  average impact on model output")
    ax.set_title(title); fig.tight_layout(); save_fig(fig, fname)


def part_b_shap_classifier(cops):
    banner("PART B1 -- SHAP explainability: shortage classifier")
    X = cops[SHORTAGE_FEATURES].astype("float64")
    y = cops["new_label"].astype(str).to_numpy()
    model = RandomForestClassifier(random_state=SEED, class_weight="balanced", **SHORTAGE_PARAMS)
    model.fit(X, y)

    # TreeExplainer is exact and fast for forests. Multiclass output can be either
    # a list [n_classes] of (n,feat) arrays or a single (n,feat,n_classes) array;
    # handle both, then average absolute importance across the 3 classes.
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        per_class = [np.abs(s).mean(axis=0) for s in sv]
    else:
        sv = np.asarray(sv)
        per_class = [np.abs(sv[:, :, k]).mean(axis=0) for k in range(sv.shape[2])]
    overall = np.mean(per_class, axis=0)

    _shap_bar(overall, SHORTAGE_FEATURES,
              "Shortage classifier -- global feature importance (SHAP)",
              "trust_shap_shortage.png")
    top = sorted(zip(SHORTAGE_FEATURES, overall), key=lambda kv: -kv[1])[:5]
    print("  top drivers (mean|SHAP|, averaged over classes):")
    for n, v in top:
        print(f"    {n:<24} {v:.4f}")
    save_result({"top_features": [t[0] for t in top]}, "trust_shap", "shortage_classifier",
                extra={"mean_abs_shap_overall": dict(zip(SHORTAGE_FEATURES, [float(v) for v in overall]))})
    print("  saved: figures/trust_shap_shortage.png, results/trust_shap__shortage_classifier.json")
    footer()


def part_b_shap_regressor(pipe, X_test, ordinal_cols, tag, title):
    banner(f"PART B -- SHAP explainability: {tag} regressor")
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    # SHAP on the trees needs the ENCODED matrix; sample up to 2000 rows for speed.
    Xs = X_test.sample(min(2000, len(X_test)), random_state=SEED)
    Xt = prep.transform(Xs)
    # ColumnTransformer emits ordinal (categorical) columns first, then passthrough numeric
    names = ordinal_cols + [c for c in X_test.columns if c not in ordinal_cols]

    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xt))
    mean_abs = np.abs(sv).mean(axis=0)
    _shap_bar(mean_abs, names, title, f"trust_shap_{tag}.png", color="#8e44ad")
    top = sorted(zip(names, mean_abs), key=lambda kv: -kv[1])[:5]
    print("  top drivers (mean|SHAP|):")
    for n, v in top:
        print(f"    {n:<18} {v:,.1f}")
    save_result({"top_features": [t[0] for t in top]}, "trust_shap", tag,
                extra={"mean_abs_shap": dict(zip(names, [float(v) for v in mean_abs]))})
    print(f"  saved: figures/trust_shap_{tag}.png, results/trust_shap__{tag}.json")
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# 8. PART C  --  FAIRNESS / DISPARITY audit
# ══════════════════════════════════════════════════════════════════════════════

def _disparity_bar(audit, vcol, lo, hi, title, xlabel, fname, label_fn=str, higher_is_better=True):
    """Bar chart of a per-group metric with bootstrap error bars; best=green, worst=red."""
    a = audit.reset_index(drop=True).copy()
    a["label"] = a["group"].apply(label_fn)
    elo = (a[vcol] - a[lo]).clip(lower=0); ehi = (a[hi] - a[vcol]).clip(lower=0)
    colors = ["#95a5a6"] * len(a)
    worst_i = int(np.argmax(a[vcol].values)) if higher_is_better else int(np.argmin(a[vcol].values))
    best_i = int(np.argmin(a[vcol].values)) if higher_is_better else int(np.argmax(a[vcol].values))
    # for accuracy (higher better): max is best; for MAE (lower better): min is best
    if higher_is_better:
        best_i, worst_i = int(np.argmax(a[vcol].values)), int(np.argmin(a[vcol].values))
    else:
        best_i, worst_i = int(np.argmin(a[vcol].values)), int(np.argmax(a[vcol].values))
    colors[best_i] = "#27ae60"; colors[worst_i] = "#c0392b"
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.55 * len(a))))
    ax.barh(a["label"], a[vcol], xerr=[elo, ehi], color=colors, capsize=4)

    # Place the "n=" label just past the END of the error-bar whisker, not at
    # the bar's raw value. Putting it at the raw value put the text directly
    # under the CI whisker line, which sliced through the digits (reported as
    # "the candle overlapping the numbers"). Also widen the x-axis so the
    # longest whisker + label isn't clipped by the right edge of the figure.
    whisker_tip = a[vcol] + ehi
    pad = 0.02 * whisker_tip.max() if whisker_tip.max() > 0 else 0.1
    for i, (tip, n) in enumerate(zip(whisker_tip, a["n"])):
        ax.text(tip + pad, i, f"n={n}", va="center", fontsize=8, color="#333")
    ax.set_xlim(0, whisker_tip.max() * 1.18)

    ax.set_xlabel(xlabel); ax.set_title(title)
    fig.tight_layout(); save_fig(fig, fname)


def _print_disparity(d, label_fn=str):
    """
    Print a disparity summary, telling the reader when the % is unreliable.

    `label_fn` translates the raw group key (e.g. TEER digit "0") into the same
    human-readable label already used in the per-group table above it (e.g.
    "TEER 0 (management)") -- without this, best_group/worst_group print as bare
    codes and the two lines under each other look inconsistent.
    """
    if d.get("relative_reliable", True):
        rel = f"{d['disparity_relative']:.1%}"
    else:
        rel = (f"relative % unreliable (base metric near zero); "
               f"trust the absolute gap = {d['absolute_gap']:,.1f}")
    sig = d.get("gap_is_significant")
    print(f"    -> disparity: {rel}  |  worst {label_fn(d['worst_group'])} "
          f"vs best {label_fn(d['best_group'])}  |  significant (intervals disjoint): {sig}")


def audit_classifier_disparity(cops):
    banner("PART C1 -- Fairness audit: shortage classifier")
    X = cops[SHORTAGE_FEATURES].astype("float64")
    y = cops["new_label"].astype(str).to_numpy()

    # out-of-fold predictions: every occupation predicted by a model that never trained on it
    preds = np.empty(len(X), dtype=object)
    for tr, te in stratified_folds(X, y):
        m = RandomForestClassifier(random_state=SEED, class_weight="balanced", **SHORTAGE_PARAMS)
        m.fit(X.iloc[tr], y[tr]); preds[te] = m.predict(X.iloc[te])
    aud = cops[["noc_code", "teer", "posting"]].copy()
    aud["y_true"] = y; aud["y_pred"] = preds
    aud["pop_tier"] = popularity_tier(aud["posting"])

    # axis 1: NOC TEER level (education tier)
    by_teer = grouped_classification_audit(aud, "teer", "y_true", "y_pred")
    d_teer = disparity_summary(by_teer, "accuracy", "acc_lo", "acc_hi", higher_is_better=True)
    print("  accuracy by NOC TEER:")
    for _, r in by_teer.iterrows():
        print(f"    {teer_name(r['group']):<38} acc {r['accuracy']:.3f} "
              f"[{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]  macroF1 {r['macro_f1']:.3f}  n={r['n']}")
    _print_disparity(d_teer, label_fn=teer_name)
    _disparity_bar(by_teer, "accuracy", "acc_lo", "acc_hi",
                   "Shortage classifier -- accuracy by NOC TEER",
                   "accuracy (95% bootstrap CI)",
                   "trust_disparity_shortage_teer.png", label_fn=teer_name)
    save_csv(by_teer, "trust_disparity_shortage_by_teer.csv")

    # axis 2: occupation popularity
    by_pop = grouped_classification_audit(aud, "pop_tier", "y_true", "y_pred")
    d_pop = disparity_summary(by_pop, "accuracy", "acc_lo", "acc_hi", higher_is_better=True)
    print("  accuracy by occupation popularity:")
    for _, r in by_pop.iterrows():
        print(f"    {str(r['group']):<12} acc {r['accuracy']:.3f} "
              f"[{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]  n={r['n']}")
    _print_disparity(d_pop, label_fn=str)
    # (region axis is not applicable: the classifier predicts per-occupation, with no province)

    save_result({"disparity_target": DISPARITY_TARGET}, "trust_fairness", "shortage_classifier",
                extra={"by_teer": d_teer, "by_popularity": d_pop,
                       "note": "region axis N/A: classifier is per-occupation, not per-posting"})
    print("  saved: figures/trust_disparity_shortage_teer.png, "
          "results/trust_fairness__shortage_classifier.json + CSV")
    footer()


def audit_regressor_disparity(aud, tag, prefix, abs_floor=0.0):
    """
    Fairness audit for a regressor across all three axes. `abs_floor` is the
    metric scale below which a relative disparity % is meaningless -- for posting
    volume (MAE of a few postings) we pass a floor so a tiny gap is not reported
    as a giant percentage.
    """
    axes = [("teer", teer_name, "NOC TEER", f"trust_disparity_{tag}_teer.png"),
            ("province", str, "region (province)", f"trust_disparity_{tag}_region.png"),
            ("pop_tier", str, "occupation popularity", f"trust_disparity_{tag}_pop.png")]
    results = {}
    for col, lf, axis_name, fname in axes:
        by = grouped_regression_audit(aud, col, "true", "pred")
        if len(by) < 2:
            continue
        d = disparity_summary(by, "mae", "mae_lo", "mae_hi",
                              higher_is_better=False, abs_floor=abs_floor)
        print(f"  {prefix} MAE by {axis_name} (lower is better):")
        for _, r in by.iterrows():
            print(f"    {lf(r['group'])[:38]:<38} MAE {r['mae']:>9,.0f} "
                  f"[{r['mae_lo']:>8,.0f}, {r['mae_hi']:>8,.0f}]  R2 {r['r2']:.2f}  n={r['n']:,}")
        _print_disparity(d, label_fn=lf)
        _disparity_bar(by, "mae", "mae_lo", "mae_hi",
                       f"{prefix} -- MAE by {axis_name}", "MAE in CAD (95% bootstrap CI)",
                       fname, label_fn=lf, higher_is_better=False)
        save_csv(by, fname.replace(".png", ".csv"))
        results[col] = d
    save_result({"disparity_target": DISPARITY_TARGET}, "trust_fairness", tag, extra=results)
    print(f"  saved: figures/trust_disparity_{tag}_*.png, results/trust_fairness__{tag}.json + CSVs")
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  --  run the three audits in order and print a manifest of everything saved
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_all_seeds(SEED)
    banner("SkillBridge Trust Layer -- calibration, SHAP, fairness")
    ensure_data()
    footer()

    # rebuild the classifier's feature table once, reuse it across all three audits
    cops = build_shortage_frame()
    part_a_calibration(cops)          # A. calibration
    part_b_shap_classifier(cops)      # B. explainability (classifier)
    audit_classifier_disparity(cops)  # C. fairness (classifier)

    # salary regressor: rebuild -> explain -> audit
    sal_model, sal_Xtest, sal_ord, sal_aud = build_salary_model()
    part_b_shap_regressor(sal_model, sal_Xtest, sal_ord, "salary",
                          "Salary regressor -- global feature importance (SHAP)")
    banner("PART C2 -- Fairness audit: salary regressor")
    audit_regressor_disparity(sal_aud, "salary", "Salary")

    # posting-volume regressor: rebuild -> audit (abs_floor guards the near-zero MAE %)
    vol_aud = build_volume_model()
    banner("PART C3 -- Fairness audit: posting-volume regressor")
    audit_regressor_disparity(vol_aud, "posting_volume", "Posting volume", abs_floor=5.0)

    # ---- final manifest: exactly what was persisted, and where ----
    banner("DONE -- everything below is saved to disk (JSON + CSV + PNG)")
    print(f"  results/  ({RESULTS_DIR})")
    for f in sorted(_SAVED["json"]):
        print(f"     [json] {f}")
    for f in sorted(_SAVED["csv"]):
        print(f"     [csv]  {f}")
    print(f"  figures/  ({FIGURES_DIR})")
    for f in sorted(_SAVED["png"]):
        print(f"     [png]  {f}")
    print("=" * _WIDTH)


if __name__ == "__main__":
    main()
