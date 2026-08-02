"""
scripts/07_shortage_query.py
=============================
Ask about an occupation, get its labour-market outlook.
Owner: Irai Kumaran Sivanesan.  CSC 503 Data Mining, Summer 2026.

    python3 scripts/07_shortage_query.py          train quietly, then ask me
    python3 scripts/07_shortage_query.py --eval   also print diagnostics + save plots

Type "nurse" and it tells you whether that occupation is heading for a worker
shortage, a balance, or a surplus. Or it says it does not know, which it does
often, on purpose.


WHY IT REFUSES TO ANSWER SOMETIMES
-----------------------------------
The model is not equally good at its three answers:

    Balance    F1 0.83   good
    Shortage   F1 0.61   usable
    Surplus    F1 0.04   worthless, 1 correct out of 17

Reporting all three as though they were equally solid would be dishonest. So
before answering, this tool works out, from held-out data, how confident it has
to be before each answer is right at least half the time. If it cannot clear
that bar it says so instead of guessing.

The thresholds are measured, not invented. See learn_thresholds().
"""

import sys
import difflib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from skillbridge.config import CLEAN_DIR, FIGURES_DIR, SEED
from skillbridge.splits import stratified_folds
from skillbridge.metrics import classification_report_full

np.random.seed(SEED)

CLASS_ORDER = ["Balance", "Shortage", "Surplus"]
BEST_PARAMS = {"max_depth": 6, "n_estimators": 400, "min_samples_leaf": 1}
TARGET_PRECISION = 0.50     # the promise: when we report X, we are right >= half the time
MIN_SUPPORT = 15            # need this many cases before trusting a precision estimate

BLURB = {
    "Shortage": "SHORTAGE -- projected to need more workers than will be available",
    "Balance":  "BALANCE  -- worker supply and demand projected to roughly match",
    "Surplus":  "SURPLUS  -- projected to have more workers than jobs",
}


# ══════════════════════════════════════════════════════════════
# data  (identical 12-feature build to 02_shortage_classifier.py round 2)
# ══════════════════════════════════════════════════════════════

def build():
    cops = pd.read_csv(CLEAN_DIR / "cops_projections_clean.csv")
    cops = cops[cops["noc_code"].str.len() == 5].copy()
    cops["new_label"] = cops["future_conditions"].map({
        "Strong risk of Shortage": "Shortage", "Moderate risk of Shortage": "Shortage",
        "Balance": "Balance",
        "Moderate risk of Surplus": "Surplus", "Strong risk of Surplus": "Surplus",
    })

    jobs = pd.read_csv(CLEAN_DIR / "jobbank_clean.csv")
    pc = jobs.groupby("noc21_code").size()
    pc.index = pc.index.astype(str).str.zfill(5)
    cops["posting"] = cops["noc_code"].map(pc).fillna(0)
    cops = cops.dropna(subset=["new_label"]).reset_index(drop=True)

    oasis = pd.read_csv(CLEAN_DIR / "oasis_descriptors_long.csv", dtype={"noc_code": str})
    cops["avg_rating"] = cops["noc_code"].map(oasis.groupby("noc_code")["rating"].mean())
    cops["core_fraction"] = cops["noc_code"].map(
        oasis.groupby("noc_code")["rating"].apply(lambda r: (r >= 4).mean()))

    sal = jobs.groupby("noc21_code")["salary_annual"].agg(["mean", "std"])
    sal.index = sal.index.astype(str).str.zfill(5)
    cops["salary_mean"] = cops["noc_code"].map(sal["mean"])
    cops["salary_std"] = cops["noc_code"].map(sal["std"])
    cops["salary_mean"] = cops["salary_mean"].fillna(cops["salary_mean"].median())
    cops["salary_std"] = cops["salary_std"].fillna(0)

    cat = oasis.groupby(["noc_code", "category"])["rating"].mean().unstack("category")
    cat.columns = ["oa_" + c.replace(" ", "_").lower() for c in cat.columns]
    cops = cops.join(cat, on="noc_code")
    catCols = list(cat.columns)
    cops[catCols] = cops[catCols].fillna(cops[catCols].median())

    cops["posting_log"] = np.log1p(cops["posting"])

    feats = (["posting", "posting_log", "avg_rating", "core_fraction",
              "employment_growth", "salary_mean", "salary_std"] + catCols)
    return cops, feats


def rf():
    return RandomForestClassifier(random_state=SEED, class_weight="balanced", **BEST_PARAMS)


# ══════════════════════════════════════════════════════════════
# how far to trust each answer
# ══════════════════════════════════════════════════════════════

def out_of_fold(X, y):
    """Predict every occupation once, always by a model that never saw it."""
    proba = np.zeros((len(X), len(CLASS_ORDER)))
    pred = np.empty(len(X), dtype=object)
    for tr, te in stratified_folds(X, y, n_folds=5, seed=SEED):
        m = rf().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])
        proba[te] = p
        pred[te] = m.classes_[p.argmax(axis=1)]
    return pred, proba


def learn_thresholds(y, pred, proba):
    """Lowest confidence at which each answer reaches TARGET_PRECISION.
    None means the answer never gets there, so we never report it."""
    conf = proba.max(axis=1)
    out = {}
    for c in CLASS_ORDER:
        out[c] = None
        for t in np.arange(0.30, 0.96, 0.01):
            sel = (pred == c) & (conf >= t)
            if sel.sum() < MIN_SUPPORT:
                continue
            prec = float((y[sel] == c).mean())    # when we said c, how often right
            if prec >= TARGET_PRECISION:
                out[c] = (round(float(t), 2), prec, int(sel.sum()))
                break
    return out


# ══════════════════════════════════════════════════════════════
# lookup + answer
# ══════════════════════════════════════════════════════════════

def find(text, cops):
    """5-digit NOC code, exact name, partial name, or near-miss spelling."""
    text = text.strip()
    if text.isdigit():
        return list(cops.index[cops["noc_code"] == text.zfill(5)])
    names = cops["occupation_name"].astype(str)
    exact = cops.index[names.str.lower() == text.lower()]
    if len(exact):
        return list(exact)
    part = cops.index[names.str.lower().str.contains(text.lower(), regex=False)]
    if len(part):
        return list(part)
    close = difflib.get_close_matches(text, names.tolist(), n=5, cutoff=0.5)
    return [int(cops.index[names == c][0]) for c in close]


def answer(row, cops, model, feats, thr):
    p = model.predict_proba(cops.loc[[row], feats])[0]
    top = int(p.argmax())
    cls, conf = CLASS_ORDER[top], float(p[top])
    rule = thr[cls]

    print(f"\n  {cops.loc[row, 'occupation_name']}   (NOC {cops.loc[row, 'noc_code']})")
    print(f"  {'-' * 64}")

    if rule is None:
        print(f"  NOT ENOUGH EVIDENCE")
        print(f"    leans {cls}, but {cls} calls are never reliable enough to report")
    elif conf < rule[0]:
        print(f"  NOT ENOUGH EVIDENCE")
        print(f"    leans {cls} at {conf:.0%}, below the {rule[0]:.0%} needed to report it")
    else:
        print(f"  {BLURB[cls]}")
        print(f"    confidence {conf:.0%}; calls like this are right {rule[1]:.0%} of the time")

    print()
    for i, c in enumerate(CLASS_ORDER):
        bar = "#" * int(round(p[i] * 34))
        print(f"    {c:<9}{p[i]:>6.0%}  {bar}")
    print(f"\n    government's own projection: {cops.loc[row, 'new_label']}")


# ══════════════════════════════════════════════════════════════
# optional diagnostics
# ══════════════════════════════════════════════════════════════

def diagnostics(cops, y, pred, proba, thr, model, feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = classification_report_full(y, pred, labels=CLASS_ORDER)
    print(f"\n  pooled macro-F1 {f1_score(y, pred, average='macro', zero_division=0):.3f}")
    print(f"  per-class F1: {({k: round(v['f1'], 3) for k, v in rep['per_class'].items()})}")
    print(f"\n  confidence needed before each answer is reported")
    print(f"    {'answer':<10}{'threshold':>11}{'precision':>11}{'n':>6}")
    for c in CLASS_ORDER:
        r = thr[c]
        if r:
            print(f"    {c:<10}{r[0]:>11.2f}{r[1]:>11.3f}{r[2]:>6}")
        else:
            print(f"    {c:<10}{'never':>11}{'-':>11}{'-':>6}")

    cov = sum(1 for i in range(len(cops))
              if thr[pred[i]] and proba[i].max() >= thr[pred[i]][0])
    print(f"\n  answers {cov} of {len(cops)} occupations ({cov/len(cops):.0%}), abstains on the rest")

    # graph 1 -- F1 per class
    f1s = [rep["per_class"][c]["f1"] for c in CLASS_ORDER]
    sup = [rep["per_class"][c]["support"] for c in CLASS_ORDER]
    cols = ["#2f855a" if v >= .6 else "#c05621" if v >= .3 else "#c53030" for v in f1s]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for b, v, s in zip(ax.bar(CLASS_ORDER, f1s, color=cols, width=.55), f1s, sup):
        ax.text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}\n(n={s})", ha="center", fontsize=9)
    ax.axhline(.50, ls="--", lw=1, color="grey")
    ax.text(2.42, .515, "target 0.50", fontsize=8, color="grey", ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("F1 score")
    ax.set_title("Shortage classifier: reliability by answer type", fontsize=11)
    ax.text(.5, -.17, "green = usable   orange = weak   red = never reported",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "shortage_f1_by_class.png", dpi=150); plt.close(fig)

    # graph 2 -- what the tool answers for a few real occupations
    rows = []
    for q in ["Registered nurses", "Software", "Cashiers", "Legislators"]:
        hits = find(q, cops)
        if hits and hits[0] not in rows:
            rows.append(hits[0])
    rows = rows[:4]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.6, 1.5 * len(rows)))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        p = model.predict_proba(cops.loc[[r], feats])[0]
        cls = CLASS_ORDER[int(p.argmax())]
        rule = thr[cls]
        abstain = rule is None or p.max() < rule[0]
        bc = ["#cbd5e0"] * 3
        bc[CLASS_ORDER.index(cls)] = "#a0aec0" if abstain else "#2b6cb0"
        ax.barh(CLASS_ORDER, p, color=bc, height=.6)
        if rule:
            ax.axvline(rule[0], ls="--", lw=1, color="#c53030")
        nm = str(cops.loc[r, "occupation_name"])[:52]
        ax.set_title(f"{nm}   ->   {'ABSTAIN' if abstain else cls.upper()}", fontsize=9, loc="left")
        ax.set_xlim(0, 1); ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("model probability   (red line = confidence needed to report)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shortage_query_examples.png", dpi=150); plt.close(fig)
    print(f"  saved 2 figures to {FIGURES_DIR}")


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def ask(prompt, default):
    try:
        v = input(f"  {prompt} [{default}]: ").strip()
    except EOFError:
        v = ""
    return v or default


def main():
    verbose = "--eval" in sys.argv[1:]

    print("  loading data and training, this takes a few seconds...", end="", flush=True)
    cops, feats = build()
    X = cops[feats].astype("float64")
    y = cops["new_label"].astype(str).to_numpy()
    pred, proba = out_of_fold(X, y)
    thr = learn_thresholds(y, pred, proba)
    model = rf().fit(X, y)
    print(" done.")

    if verbose:
        diagnostics(cops, y, pred, proba, thr, model, feats)

    never = [c for c, v in thr.items() if v is None]
    if never:
        print(f"  note: this tool never reports {', '.join(never)}, not reliable enough.")

    print("\n  Type an occupation (e.g. nurse, welder, 31301). Blank line to quit.")
    while True:
        try:
            q = input("\n  occupation> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        hits = find(q, cops)
        if not hits:
            print("    nothing matched. try fewer words, or a 5-digit NOC code.")
            continue
        if len(hits) > 1:
            print(f"    {len(hits)} matches, showing the closest:")
            for h in hits[1:5]:
                print(f"      also: {cops.loc[h, 'occupation_name']} ({cops.loc[h, 'noc_code']})")
        answer(hits[0], cops, model, feats, thr)


if __name__ == "__main__":
    main()
