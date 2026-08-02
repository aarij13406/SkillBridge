"""
scripts/08_export_webapp_data.py
=================================
Bakes the three models' outputs into one JSON file for the web app.
Owner: Irai Kumaran Sivanesan.  CSC 503 Data Mining, Summer 2026.

    python3 scripts/08_export_webapp_data.py     ->  webapp/data.json


WHY PRECOMPUTE INSTEAD OF SERVING A LIVE API
---------------------------------------------
The shortage verdict and the salary estimate are FIXED per occupation. They do
not depend on anything the user types, so computing them on every request would
be the same arithmetic repeated forever. Baking them into a JSON file means the
web app is a static page: no backend, nothing to deploy, nothing to wake up, and
nothing that can time out during a demo.

The genuinely interactive part, subtracting the skills someone already has from
the skills a target occupation needs, is set arithmetic and runs in the browser.


WHAT THIS FILE DOES AND DOES NOT DO
------------------------------------
    shortage    runs MY model (02_shortage_classifier.py round 2 config) and
                applies the abstain rule from 07_shortage_query.py

    salary      READS M. Sundaresan's saved predictions. Does not retrain, does
                not re-implement, does not touch his script.

    volume      READS his saved posting-volume forecast. Same.

    skill gap   competency ratings + leverage, the inputs behind A. Anto's
                dumbbell chart and his "core in ~X% of jobs" figure.

    market      occupation x tool demand built from LinkedIn postings, matching
                the logic in his build_market_matrix().

REQUIRES, all produced by teammates and committed:
    datasets/clean/*.csv
    salary_predictions_for_fairness_audit.csv        (scripts/03)
    posting_volume_predictions_for_fairness_audit.csv (scripts/03)
    results/salary_regional_demand__random_forest.json
    results/posting_volume__random_forest.json
"""

import sys
import re
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from skillbridge.config import CLEAN_DIR, PROJECT_ROOT, RESULTS_DIR, SEED
from skillbridge.splits import stratified_folds
from skillbridge.metrics import classification_report_full

np.random.seed(SEED)

CLASS_ORDER = ["Balance", "Shortage", "Surplus"]
BEST_PARAMS = {"max_depth": 6, "n_estimators": 400, "min_samples_leaf": 1}
TARGET_PRECISION = 0.50
MIN_SUPPORT = 15

OUT_DIR = PROJECT_ROOT / "webapp"
OUT_FILE = OUT_DIR / "data.json"


def hr(t):
    print(f"\n{'=' * 66}\n  {t}\n{'=' * 66}")


# ══════════════════════════════════════════════════════════════
# 1. SHORTAGE  (my model)
# ══════════════════════════════════════════════════════════════

def build_features():
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
    return cops, feats, oasis


def rf():
    return RandomForestClassifier(random_state=SEED, class_weight="balanced", **BEST_PARAMS)


def build_shortage(cops, feats):
    X = cops[feats].astype("float64")
    y = cops["new_label"].astype(str).to_numpy()

    # out-of-fold predictions: every occupation scored by a model that never saw
    # it. these are what the abstain thresholds get measured on. using in-sample
    # predictions would make the model look far more reliable than it is.
    proba = np.zeros((len(X), 3))
    pred = np.empty(len(X), dtype=object)
    for tr, te in stratified_folds(X, y, n_folds=5, seed=SEED):
        m = rf().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])
        proba[te] = p
        pred[te] = m.classes_[p.argmax(axis=1)]

    conf = proba.max(axis=1)
    thresholds = {}
    for c in CLASS_ORDER:
        thresholds[c] = None
        for t in np.arange(0.30, 0.96, 0.01):
            sel = (pred == c) & (conf >= t)
            if sel.sum() < MIN_SUPPORT:
                continue
            prec = float((y[sel] == c).mean())
            if prec >= TARGET_PRECISION:
                thresholds[c] = {"threshold": round(float(t), 2),
                                 "precision": round(prec, 3),
                                 "n": int(sel.sum())}
                break

    report = classification_report_full(y, pred, labels=CLASS_ORDER)

    # final model trained on everything, used for the shipped per-occupation answer
    model = rf().fit(X, y)
    allProba = model.predict_proba(X)

    out = {}
    for i, row in cops.iterrows():
        p = allProba[i]
        top = int(p.argmax())
        cls = CLASS_ORDER[top]
        c = float(p[top])
        rule = thresholds[cls]
        if rule is None:
            state, note = "abstain", f"{cls} predictions are never reliable enough to report"
        elif c < rule["threshold"]:
            state, note = "abstain", f"leans {cls} but below the {rule['threshold']:.0%} confidence needed"
        else:
            state, note = "confident", f"calls like this are right {rule['precision']:.0%} of the time"
        out[row["noc_code"]] = {
            "verdict": cls,
            "state": state,
            "note": note,
            "confidence": round(c, 4),
            "probabilities": {CLASS_ORDER[j]: round(float(p[j]), 4) for j in range(3)},
            "reliability": None if rule is None else rule["precision"],
            "actual": row["new_label"],
        }

    meta = {
        "macro_f1": round(float(report["macro_f1"]), 4),
        "per_class_f1": {k: round(v["f1"], 3) for k, v in report["per_class"].items()},
        "per_class_support": {k: int(v["support"]) for k, v in report["per_class"].items()},
        "thresholds": thresholds,
        "never_reported": [c for c, v in thresholds.items() if v is None],
        "n_occupations": int(len(cops)),
    }
    return out, meta


# ══════════════════════════════════════════════════════════════
# 2. SALARY + VOLUME  (read saved output, do not retrain)
# ══════════════════════════════════════════════════════════════

def read_model_metrics():
    """M. Sundaresan's own reported error, so the UI can state how far off the
    predictions typically are."""
    out = {}
    for key, fn in [("salary", "salary_regional_demand__random_forest.json"),
                    ("volume", "posting_volume__random_forest.json")]:
        p = RESULTS_DIR / fn
        if p.exists():
            m = json.loads(p.read_text()).get("metrics", {})
            out[key] = {"mae": round(float(m.get("mae", 0)), 2),
                        "r2": round(float(m.get("r2", 0)), 3)}
    return out


def build_salary():
    f = PROJECT_ROOT / "salary_predictions_for_fairness_audit.csv"
    if not f.exists():
        print("  ! salary predictions not found, skipping. run scripts/03 first.")
        return {}, {}

    d = pd.read_csv(f)
    d["noc"] = d["noc21_code"].astype(str).str.zfill(5)

    # his file is row-per-posting on the test split. aggregate to the unit the
    # UI asks about: one occupation in one province.
    #
    # low/avg/high come from the PREDICTED column and the spread is real: the
    # same occupation in the same province still varies by city, education and
    # experience, which are all features of his model. so "lowest" is roughly an
    # entry-level posting and "highest" a senior one.
    g = (d.groupby(["noc", "province"])
           .agg(predicted=("predicted_salary", "median"),
                low=("predicted_salary", "min"),
                avg=("predicted_salary", "mean"),
                high=("predicted_salary", "max"),
                actual=("true_salary", "median"),
                n=("predicted_salary", "size"))
           .reset_index())
    g = g[g["n"] >= 3]        # a median from 1-2 postings is not worth showing

    byOcc = {}
    for noc, sub in g.groupby("noc"):
        byOcc[noc] = {
            r["province"]: {"predicted": round(float(r["predicted"])),
                            "low": round(float(r["low"])),
                            "avg": round(float(r["avg"])),
                            "high": round(float(r["high"])),
                            "actual": round(float(r["actual"])),
                            "n": int(r["n"])}
            for _, r in sub.iterrows()
        }

    national = d.groupby("noc")["predicted_salary"].median().round().astype(int).to_dict()
    return byOcc, national


def build_volume():
    f = PROJECT_ROOT / "posting_volume_predictions_for_fairness_audit.csv"
    if not f.exists():
        print("  ! volume predictions not found, skipping.")
        return {}

    d = pd.read_csv(f)
    d["noc"] = d["noc21_code"].astype(str).str.zfill(5)
    out = {}
    for noc, sub in d.groupby("noc"):
        rows = [{
            "province": r["province"],
            "nov": int(r["nov_count"]), "dec": int(r["dec_count"]), "jan": int(r["jan_count"]),
            "actual_feb": int(r["true_posting_count"]),
            "predicted_feb": round(float(r["predicted_posting_count"]), 1),
        } for _, r in sub.iterrows()]
        rows.sort(key=lambda x: -x["predicted_feb"])
        out[noc] = rows[:15]
    return out


# ══════════════════════════════════════════════════════════════
# 3. COMPETENCIES + LEVERAGE
# ══════════════════════════════════════════════════════════════

def build_skillgap(oasis):
    """Ships the full occupation x descriptor ratings matrix so the browser can
    compute a gap between ANY current job and ANY target job:

        gap(descriptor) = rating[target][descriptor] - rating[current][descriptor]

    That is the data behind A. Anto's dumbbell chart in plot_user(): one row per
    competency, a line from "where you are" to "what the job needs".

    Size: ~516 occupations x 181 descriptors, rounded to 1dp, roughly 350 KB.
    """
    wide = (oasis.groupby(["noc_code", "descriptor_name"])["rating"]
                 .mean().unstack("descriptor_name"))
    descriptors = [c.strip() for c in wide.columns]
    ratings = {noc: [round(float(v), 1) for v in row]
               for noc, row in zip(wide.index, wide.to_numpy())}

    # which OaSIS family each descriptor belongs to
    catOf = (oasis.drop_duplicates("descriptor_name")
                  .set_index(oasis.drop_duplicates("descriptor_name")["descriptor_name"].str.strip())
                  ["category"].to_dict())
    categories = [catOf.get(d, "Other") for d in descriptors]

    # LEVERAGE, the market-relevance figure A. Anto prints as
    #   "it's core in ~X% of jobs"
    # from build_leverage() in 13_skillgap.py: count occupations rating the
    # descriptor >= 4, i.e. core. His version divides by the max to normalise it
    # for use as a ranking bonus; here we keep the raw FRACTION OF OCCUPATIONS,
    # because "core in 34% of all occupations" is a statement a user can read,
    # whereas the normalised number is only meaningful relative to the top skill.
    arr = wide.to_numpy()
    leverage = [round(float((arr[:, j] >= 4).mean()), 3) for j in range(arr.shape[1])]
    return descriptors, categories, ratings, leverage


# ══════════════════════════════════════════════════════════════
# 4. MARKET TOOLS  (second half of A. Anto's output)
# ══════════════════════════════════════════════════════════════

def build_market():
    """Occupation x market-skill demand, from real LinkedIn postings.

    This is the data behind his plot_market_gap(): concrete tools and
    technologies (Python, Excel, SQL) that the OaSIS taxonomy does not contain,
    because it only describes ability-level competencies.

    Matching logic copied from his build_market_matrix() so the two agree:
    singularise the occupation name, take the phrase before the first separator,
    and find LinkedIn job titles containing it. Keyed by noc_code here rather
    than his occupation_id, because that is what the rest of this file joins on.

        prevalence = fraction of that occupation's postings mentioning the skill
    """
    postsP = CLEAN_DIR / "linkedin_postings_canada.csv"
    skillsP = CLEAN_DIR / "linkedin_skills_canada.csv"
    occP = CLEAN_DIR / "occupation_lookup.csv"
    if not (postsP.exists() and skillsP.exists() and occP.exists()):
        print("  ! LinkedIn files missing, skipping market tools")
        return {}, []

    MIN_POSTS, MIN_DF, TOP_VOCAB = 15, 30, 400

    occ = pd.read_csv(occP, dtype={"noc_code": str})[["noc_code", "occupation_name"]].drop_duplicates()
    posts = pd.read_csv(postsP, usecols=["job_link", "job_title"]).dropna()
    skills = pd.read_csv(skillsP).dropna()

    def sing(t):
        return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w
                        for w in re.findall(r"[a-z0-9+#.]+", str(t).lower()))

    SEP = re.compile(r",| and | including | except |/| - ")
    posts["ts"] = posts["job_title"].map(sing)

    links = {}
    for noc, name in zip(occ["noc_code"], occ["occupation_name"]):
        ph = sing(SEP.split(str(name))[0]).strip()
        if len(ph) < 4:
            continue
        m = posts["ts"].str.contains(ph, regex=False, na=False)
        if m.sum() >= MIN_POSTS:
            links.setdefault(str(noc).zfill(5), set()).update(posts.loc[m, "job_link"])

    if not links:
        print("  ! no occupation matched enough postings")
        return {}, []

    kept = set().union(*links.values())
    sk = skills[skills["job_link"].isin(kept)]
    vocab = [s for s, c in sk["skill_clean"].value_counts().items() if c >= MIN_DF][:TOP_VOCAB]
    sid = {s: j for j, s in enumerate(vocab)}
    byLink = (sk[sk["skill_clean"].isin(sid)]
              .groupby("job_link")["skill_clean"].apply(list).to_dict())

    out = {}
    for noc, lk in links.items():
        cnt = np.zeros(len(vocab))
        for l in lk:
            for s in byLink.get(l, []):
                cnt[sid[s]] += 1
        prev = cnt / len(lk)
        out[noc] = {str(j): round(float(p), 3) for j, p in enumerate(prev) if p >= 0.02}
    return out, vocab


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    hr("1/5  shortage classifier")
    cops, feats, oasis = build_features()
    shortage, shortageMeta = build_shortage(cops, feats)
    print(f"  {len(shortage)} occupations scored, macro-F1 {shortageMeta['macro_f1']}")
    print(f"  per-class F1: {shortageMeta['per_class_f1']}")
    if shortageMeta["never_reported"]:
        print(f"  never reported: {', '.join(shortageMeta['never_reported'])}")
    nAbstain = sum(1 for v in shortage.values() if v["state"] == "abstain")
    print(f"  abstains on {nAbstain} of {len(shortage)} ({nAbstain/len(shortage):.0%})")

    hr("2/5  salary  (reading saved predictions)")
    salary, salaryNational = build_salary()
    print(f"  {len(salary)} occupations with province-level salary")

    hr("3/5  posting volume  (saved forecast)")
    volume = build_volume()
    print(f"  {len(volume)} occupations with volume forecast")

    hr("4/5  competencies and leverage")
    descriptors, categories, ratings, leverage = build_skillgap(oasis)
    print(f"  {len(ratings)} occupations x {len(descriptors)} descriptors")
    print(f"  categories: {sorted(set(categories))}")
    print(f"  leverage (share of occupations where a competency is core): "
          f"min {min(leverage):.2f}, max {max(leverage):.2f}")

    hr("5/5  market tools from LinkedIn  (takes ~30s)")
    market, marketVocab = build_market()
    print(f"  {len(market)} occupations matched, vocabulary of {len(marketVocab)} tools")

    # the occupation list that drives the search box
    occupations = sorted(
        [{"noc": r["noc_code"], "name": str(r["occupation_name"]),
          "has_salary": r["noc_code"] in salary, "has_volume": r["noc_code"] in volume}
         for _, r in cops.iterrows()],
        key=lambda x: x["name"])

    payload = {
        "meta": {
            "generated_by": "scripts/08_export_webapp_data.py",
            "shortage_model": f"RandomForest class_weight=balanced {BEST_PARAMS}",
            "shortage": shortageMeta,
            "regression": read_model_metrics(),
            "core_threshold": 4,
        },
        "occupations": occupations,
        "shortage": shortage,
        "salary": salary,
        "salary_national": salaryNational,
        "volume": volume,
        "descriptors": descriptors,
        "categories": categories,
        "leverage": leverage,
        "ratings": ratings,
        "market_vocab": marketVocab,
        "market": market,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, separators=(",", ":")))
    hr("done")
    print(f"  wrote {OUT_FILE}  ({OUT_FILE.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(occupations)} occupations searchable")
    print(f"\n  serve it:  cd webapp && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
