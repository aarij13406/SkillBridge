"""
scripts/10_recommender.py
=========================
SkillBridge, Component 1: Occupation Recommender.  THE MODEL LADDER.
Owner: Muhammad Aarij (V01096775).  CSC 503 Data Mining, Summer 2026.

Models live in skillbridge/models.py because three entry points need them
(10_recommender, 11_coldstart, 12_query). This file is the driver: split, fit the
ladder, evaluate on both axes, sweep sparsity, and persist the production model.

THE TASK
--------
The OaSIS matrix is COMPLETE: 900 profiles x 181 descriptors, 162,899 of 162,900
cells observed. "Does this edge exist?" is degenerate (the answer is always yes), so
the proposal's link-prediction framing cannot distinguish a good model from a stupid
one. Reformulated as MATRIX COMPLETION: hide 10% of CELLS, predict the ordinal 0-5
rating. Every promised metric survives, and becomes meaningful: the binary question
"is this descriptor CORE (>= 4)?" has a 14.1% positive rate.

TWO AXES, TWO TASKS, TWO BASELINES
-----------------------------------
PRIMARY    fix a DESCRIPTOR, rank OCCUPATIONS.   ~90 candidates.  <- the proposal's task
           baseline that VARIES here: occ_popularity
SECONDARY  fix an OCCUPATION, rank DESCRIPTORS.  ~18 candidates.  (Skill-Gap direction)
           baseline that VARIES here: desc_popularity

A baseline that does not vary along the axis being ranked is ranking nothing. That is
why desc_popularity scores BELOW random on the primary axis: it hands all 90 candidate
occupations an identical score.

USAGE
    python scripts/10_recommender.py --sweep --demo
    python scripts/10_recommender.py --fast
    python scripts/10_recommender.py --only mf bpr_d node2vec
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from skillbridge.config import SEED, set_all_seeds, CORE_THRESHOLD, NEGATIVE_THRESHOLD
from skillbridge.metrics import (
    precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k,
    regression_report, save_result,
)
from skillbridge.models import (
    OCC, DSC, RAT, RATING_MIN, RATING_MAX,
    load_raw, build_matrix, make_ladder, MFRecommender, score_all_occupations,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COMPONENT = "recommender"


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def _calibrate(model, mx):
    """Best MONOTONE map from a model's score onto the 0-5 scale, fitted on TRAIN.

    BPR and node2vec emit RANKING scores, not ratings, so a raw RMSE against them would
    be meaningless and any comparison with MF would be a strawman. Isotonic regression
    is ORDER-PRESERVING: it leaves every ranking metric untouched while handing each
    model the most favourable rating-scale reading that exists for it.
    """
    s = model.score(mx.train[OCC].to_numpy(), mx.train[DSC].to_numpy())
    iso = IsotonicRegression(y_min=RATING_MIN, y_max=RATING_MAX, out_of_bounds="clip")
    iso.fit(s, mx.train[RAT].to_numpy())
    return iso


def rank_axis(mx, scores, group_col, ks=(5, 10)):
    """Group by `group_col`, rank the OTHER axis inside each group.

    Relevance is BINARY (rating >= 4) for P@k / R@k / MRR, GRADED (raw 0-5) for NDCG.
    A descriptor rated 5 should outrank one rated 4; only NDCG rewards that.
    """
    te = mx.test.assign(score=scores)
    acc = {f"p@{k}": [] for k in ks}
    acc.update({f"r@{k}": [] for k in ks})
    acc.update({f"ndcg@{k}": [] for k in ks})
    acc["mrr"] = []

    for _, g in te.groupby(group_col, sort=False):
        y, s = g[RAT].to_numpy(), g["score"].to_numpy()
        rel = (y >= CORE_THRESHOLD).astype(int)
        if rel.sum() == 0:
            continue
        for k in ks:
            acc[f"p@{k}"].append(precision_at_k(rel, s, k))
            acc[f"r@{k}"].append(recall_at_k(rel, s, k))
            acc[f"ndcg@{k}"].append(ndcg_at_k(y, s, k))
        acc["mrr"].append(reciprocal_rank(rel, s))

    out = {m: (float(np.nanmean(v)) if v else float("nan")) for m, v in acc.items()}
    out["n_queries"] = len(acc["mrr"])
    out["avg_candidates"] = float(te.groupby(group_col).size().mean())
    return out


def evaluate(model, mx):
    u, i, y = mx.test[OCC].to_numpy(), mx.test[DSC].to_numpy(), mx.test[RAT].to_numpy()
    s = model.score(u, i)
    m = {"model": model.name, "target_axis": model.target_axis}

    # BINARY: core (>=4) vs irrelevant (<=1). The 43.2% of cells rated 2-3 are EXCLUDED:
    # they are neither class, and folding them in either direction would make AUC a
    # measurement of an arbitrary labelling decision rather than of the model.
    #
    # NOTE: this AUC is POOLED GLOBALLY over test cells, NOT computed per query. A model
    # can therefore post a high AUC while ranking badly on the primary axis (bpr_o does
    # exactly that). NDCG@10 and P@5 are the primary-axis metrics; AUC is supporting.
    pos, neg = y >= CORE_THRESHOLD, y <= NEGATIVE_THRESHOLD
    keep = pos | neg
    m["roc_auc"] = float(roc_auc_score(pos[keep].astype(int), s[keep]))

    m.update(rank_axis(mx, s, DSC))                                     # PRIMARY
    m.update({f"sg_{k}": v for k, v in rank_axis(mx, s, OCC).items()})  # SECONDARY

    iso = _calibrate(model, mx)
    m.update({f"cal_{k}": v for k, v in regression_report(y, iso.predict(s)).items()})

    if hasattr(model, "stopped_at"):
        m["stopped_at_epoch"] = int(model.stopped_at)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _table(rows, cols, title, note=""):
    print("\n" + "=" * 96)
    print(f"  {title}")
    print("=" * 96)
    head = " ".join(f"{c:>{w}}" for c, w, _ in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        print(" ".join(
            (f"{str(r.get(c, '')):>{w}}" if f == "s" else f"{r.get(c, float('nan')):>{w}{f}}")
            for c, w, f in cols
        ))
    if note:
        print(f"\n{note}")


def print_tables(rows):
    _table(rows,
           [("model", 16, "s"), ("target_axis", 12, "s"), ("roc_auc", 8, ".4f"),
            ("p@5", 7, ".4f"), ("p@10", 7, ".4f"), ("mrr", 7, ".4f"),
            ("ndcg@10", 8, ".4f"), ("cal_rmse", 9, ".4f")],
           "PRIMARY AXIS: fix a competency, rank OCCUPATIONS   (the proposal's task)",
           f"  {rows[0].get('avg_candidates', 0):.0f} candidate occupations per query, "
           f"{rows[0].get('n_queries', 0)} queries.\n"
           f"  Compare against occ_popularity: the baseline that VARIES on this axis.\n"
           f"  desc_popularity is CONSTANT here and ranks nothing.\n"
           f"  AUC is POOLED, not per-query. NDCG@10 and P@5 are the primary-axis metrics.")

    _table(rows,
           [("model", 16, "s"), ("target_axis", 12, "s"), ("sg_p@5", 8, ".4f"),
            ("sg_p@10", 9, ".4f"), ("sg_mrr", 8, ".4f"), ("sg_ndcg@10", 11, ".4f")],
           "SECONDARY AXIS: fix an occupation, rank COMPETENCIES  (Skill-Gap direction)",
           f"  {rows[0].get('sg_avg_candidates', 0):.0f} candidates per query: a small pool that "
           f"flatters every model.\n  Do not lead with these numbers.")

    ob = next((r for r in rows if r["model"] == "occ_popularity"), None)
    db = next((r for r in rows if r["model"] == "desc_popularity"), None)
    if ob and db:
        print("\n" + "=" * 96)
        print("  GAIN OVER THE AXIS-APPROPRIATE BASELINE")
        print("=" * 96)
        print(f"  {'model':>16}   {'PRIMARY (vs occ_pop)':>28}   {'SECONDARY (vs desc_pop)':>28}")
        print(f"  {'-' * 80}")
        for r in rows:
            if r["model"] in ("occ_popularity", "desc_popularity", "random"):
                continue
            p = f"NDCG {r['ndcg@10'] - ob['ndcg@10']:+.4f}  P@5 {r['p@5'] - ob['p@5']:+.4f}"
            s = f"NDCG {r['sg_ndcg@10'] - db['sg_ndcg@10']:+.4f}  P@5 {r['sg_p@5'] - db['sg_p@5']:+.4f}"
            print(f"  {r['model']:>16}   {p:>28}   {s:>28}")

    bd = next((r for r in rows if r["model"] == "bpr_d"), None)
    bo = next((r for r in rows if r["model"] == "bpr_o"), None)
    mf = next((r for r in rows if r["model"] == "mf"), None)
    nb = next((r for r in rows if r["model"] == "mf_no_bi"), None)

    if bd and bo:
        print("\n" + "=" * 96)
        print("  EXPERIMENT 1: must the pairwise contrast axis match the evaluation axis?")
        print("  (identical architecture, optimiser, k, epochs. ONLY the axis differs.)")
        print("=" * 96)
        print(f"  {'':>10} {'PRIMARY':>10} {'SECONDARY':>12}   carries     trained for")
        print(f"  {'-' * 64}")
        print(f"  {'BPR-D':>10} {bd['ndcg@10']:>10.4f} {bd['sg_ndcg@10']:>12.4f}   b_u         primary")
        print(f"  {'BPR-O':>10} {bo['ndcg@10']:>10.4f} {bo['sg_ndcg@10']:>12.4f}   b_i         secondary")
        if mf:
            print(f"  {'MF':>10} {mf['ndcg@10']:>10.4f} {mf['sg_ndcg@10']:>12.4f}   b_u + b_i   both")
        dp = bd["ndcg@10"] - bo["ndcg@10"]
        ds = bo["sg_ndcg@10"] - bd["sg_ndcg@10"]
        print(f"\n  BPR-D beats BPR-O on PRIMARY   by {dp:+.4f}")
        print(f"  BPR-O beats BPR-D on SECONDARY by {ds:+.4f}")
        print("  -> " + ("CONFIRMED. Each variant wins on the axis it was trained for. Which bias\n"
                         "     term is identifiable (b_u vs b_i) is fixed by the ranking axis, exactly\n"
                         "     as the difference operator predicts."
                         if dp > 0 and ds > 0 else
                         "NOT fully confirmed. Report exactly what the numbers show."))

    if mf and nb and bd:
        print("\n" + "=" * 96)
        print("  EXPERIMENT 2 (ABLATION): is the MF-over-BPR gap the LOSS, or the MISSING BIAS?")
        print("=" * 96)
        print(f"  {'model':>10} {'loss':>11} {'biases':>12} {'PRIMARY NDCG@10':>18}")
        print(f"  {'-' * 55}")
        print(f"  {'MF':>10} {'pointwise':>11} {'b_u + b_i':>12} {mf['ndcg@10']:>18.4f}")
        print(f"  {'MF-no-bi':>10} {'pointwise':>11} {'b_u only':>12} {nb['ndcg@10']:>18.4f}")
        print(f"  {'BPR-D':>10} {'pairwise':>11} {'b_u only':>12} {bd['ndcg@10']:>18.4f}")
        print()
        print(f"  MF -> MF-no-bi   (remove b_i, KEEP the loss)  : {nb['ndcg@10'] - mf['ndcg@10']:+.4f}")
        print(f"  MF-no-bi -> BPR-D (keep biases, CHANGE loss)  : {bd['ndcg@10'] - nb['ndcg@10']:+.4f}")
        print()
        if abs(nb["ndcg@10"] - mf["ndcg@10"]) > abs(bd["ndcg@10"] - nb["ndcg@10"]):
            print("  -> The BIAS TERM explains the gap, not the loss function. On the primary axis")
            print("     b_i CANCELS out of the ranking, so removing it 'should' change nothing --")
            print("     and yet it does. Without b_i, the latent vector q_i must encode the")
            print("     descriptor's general LEVEL as well as its INTERACTION pattern. The bias")
            print("     does not affect the ranking directly; it FREES THE GEOMETRY.")
        else:
            print("  -> The LOSS FUNCTION explains more of the gap than the missing bias term.")
            print("     Report exactly that.")


def save_figure(rows, sweep=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 3 if sweep is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(6.4 * n, 5.2))
    names = [r["model"] for r in rows]
    ob = next((r for r in rows if r["model"] == "occ_popularity"), None)

    ax[0].barh(names, [r["ndcg@10"] for r in rows], color="#4C72B0")
    if ob:
        ax[0].axvline(ob["ndcg@10"], ls="--", c="crimson", lw=1.4, label="occ_popularity (the bar)")
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel("NDCG@10")
    ax[0].set_title("PRIMARY: rank occupations for a competency")
    ax[0].invert_yaxis()

    w, y = 0.38, np.arange(len(names))
    ax[1].barh(y - w / 2, [r["ndcg@10"] for r in rows], w, label="primary", color="#4C72B0")
    ax[1].barh(y + w / 2, [r["sg_ndcg@10"] for r in rows], w, label="secondary", color="#DD8452")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(names)
    ax[1].set_xlabel("NDCG@10")
    ax[1].set_title("Axis matters: same scores, two rankings")
    ax[1].legend(fontsize=8)
    ax[1].invert_yaxis()

    if sweep is not None:
        for mname, g in sweep.groupby("model"):
            g = g.sort_values("test_frac")
            ax[2].plot(g["seen_per_occ"], g["ndcg@10"], marker="o", label=mname)
        if ob:
            ax[2].axhline(ob["ndcg@10"], ls="--", c="crimson", lw=1.0, alpha=0.6)
        ax[2].set_xlabel("skills known per occupation   (<- colder start)")
        ax[2].set_ylabel("NDCG@10 (primary)")
        ax[2].set_title("Cold start: how much profile do we need?")
        ax[2].invert_xaxis()
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "recommender_ladder.png"
    fig.savefig(out, dpi=150)
    print(f"\n  figure -> {out}")


# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(df, n_occ, n_desc, occ_names, desc_names, noc_of, args,
              fracs=(0.10, 0.30, 0.50, 0.70, 0.90),
              models=("occ_popularity", "jaccard_cf", "node2vec", "mf", "bpr_d")):
    """At 10% holdout the model has already seen ~163 of an occupation's 181 ratings
    before guessing the last 18. That is interpolation inside a near-complete row, and
    a regime that NEVER occurs in production: a real job seeker arrives with 10 or 15
    skills, not 163. The sweep reports the whole difficulty curve instead of one
    flattering point, and answers the product question directly."""
    print("\n" + "=" * 96)
    print("  SPARSITY SWEEP   (cold start: how much profile do we actually need?)")
    print("=" * 96)
    out = []
    for f in fracs:
        mx = build_matrix(df, n_occ, n_desc, occ_names, desc_names, noc_of,
                          test_frac=f, seed=args.seed, verbose=False)
        seen = mx.M.sum(axis=1).mean()
        line = f"  {100*f:>3.0f}% held out (~{seen:>5.1f}/{n_desc} seen) : "
        for m in make_ladder(args.k, args.epochs):
            if m.name not in models:
                continue
            m.fit(mx, np.random.default_rng(args.seed))
            r = rank_axis(mx, m.score(mx.test[OCC].to_numpy(), mx.test[DSC].to_numpy()), DSC)
            out.append({"model": m.name, "test_frac": f, "seen_per_occ": seen,
                        "ndcg@10": r["ndcg@10"], "p@5": r["p@5"], "mrr": r["mrr"]})
            line += f"{m.name} {r['ndcg@10']:.3f}  "
        print(line)

    sweep = pd.DataFrame(out)
    sweep.to_csv(RESULTS_DIR / "recommender_sparsity_sweep.csv", index=False)
    print(f"\n  sweep -> {RESULTS_DIR / 'recommender_sparsity_sweep.csv'}")
    return sweep


def demo(model, mx, n=3, rng=None):
    """Profile -> ranked occupations, showing LEVEL (naive) beside FIT (centred)."""
    rng = rng or np.random.default_rng(SEED)
    print("\n" + "=" * 96)
    print(f"  DEMO: skill profile -> ranked occupations   [{model.name}]")
    print("=" * 96)

    core = mx.train[mx.train[RAT] >= CORE_THRESHOLD]
    for u_true in rng.choice(core[OCC].unique(), size=n, replace=False):
        profile = core.loc[core[OCC] == u_true, DSC].unique()
        if len(profile) < 3:
            continue
        raw, cen = score_all_occupations(model, profile, mx.n_occ, centre=True)
        name = mx.occ_names.get(int(u_true), str(u_true))
        r_raw = int(np.where(np.argsort(-raw) == u_true)[0][0]) + 1
        r_cen = int(np.where(np.argsort(-cen) == u_true)[0][0]) + 1

        print(f"\n  PROFILE: {name[:60]}   ({len(profile)} core competencies)")
        print(f"      {'NAIVE (measures LEVEL)':<44}{'CENTRED (measures FIT)':<44}")
        print(f"      {'-' * 42}  {'-' * 42}")
        for k, (a, b) in enumerate(zip(np.argsort(-raw)[:5], np.argsort(-cen)[:5]), 1):
            an = mx.occ_names.get(int(a), str(a))[:36] + (" *" if a == u_true else "")
            bn = mx.occ_names.get(int(b), str(b))[:36] + (" *" if b == u_true else "")
            print(f"  {k}.  {an:<44}{bn:<44}")
        print(f"      {'true rank #' + str(r_raw) + '/900':<44}"
              f"{'true rank #' + str(r_cen) + '/900':<44}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    if args.fast:
        args.epochs = 8

    set_all_seeds(args.seed)

    print("=" * 96)
    print("  SkillBridge / Component 1 / Occupation Recommender")
    print("  Matrix completion over the OaSIS occupation x descriptor rating matrix")
    print("=" * 96)

    df, n_occ, n_desc, occ_names, desc_names, noc_of = load_raw(PROCESSED_DIR)
    cells = n_occ * n_desc
    print(f"  matrix   : {n_occ} occupations x {n_desc} descriptors = {cells:,} cells")
    print(f"  observed : {len(df):,}  ({len(df)/cells:.1%} coverage -- COMPLETE)")
    print(f"  core >=4 : {(df[RAT] >= CORE_THRESHOLD).mean():.1%}   "
          f"irrelevant <=1 : {(df[RAT] <= NEGATIVE_THRESHOLD).mean():.1%}")

    mx = build_matrix(df, n_occ, n_desc, occ_names, desc_names, noc_of,
                      args.test_frac, args.seed)

    ladder = make_ladder(args.k, args.epochs)
    if args.only:
        ladder = [m for m in ladder if m.name in args.only]

    print(f"\n  fitting {len(ladder)} models "
          f"(seed={args.seed}, k={args.k}, max_epochs={args.epochs})\n")

    rows, fitted = [], {}
    for model in ladder:
        # FRESH generator per model. Sharing one across the ladder makes `--only mf` and
        # a full run disagree for the same seed, defeating the purpose of a fixed seed.
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        model.fit(mx, rng)
        res = evaluate(model, mx)
        res.update({"fit_seconds": round(time.time() - t0, 2), "seed": args.seed,
                    "k": args.k, "test_frac": args.test_frac})
        rows.append(res)
        fitted[model.name] = model
        stop = f"  stop@{res['stopped_at_epoch']}" if "stopped_at_epoch" in res else ""
        print(f"  [{res['fit_seconds']:>7.2f}s] {model.name:<16} AUC {res['roc_auc']:.4f}   "
              f"NDCG-prim {res['ndcg@10']:.4f}   NDCG-sec {res['sg_ndcg@10']:.4f}{stop}")
        save_result(res, COMPONENT, model.name, RESULTS_DIR)

    print_tables(rows)

    sweep = run_sweep(df, n_occ, n_desc, occ_names, desc_names, noc_of, args) if args.sweep else None
    save_figure(rows, sweep)

    if args.demo and "mf" in fitted:
        demo(fitted["mf"], mx, n=3)

    # ---- PRODUCTION model: train on (almost) all cells and persist --------------
    # scripts/12_query.py loads this. A production model should not be handicapped by a
    # test holdout it will never be evaluated against; we keep a 2% slice only so early
    # stopping still has a validation signal.
    if (not args.only) or ("mf" in (args.only or [])):
        print("\n" + "=" * 96)
        print("  Training the PRODUCTION model on the full matrix (for 12_query.py)")
        print("=" * 96)
        full = build_matrix(df, n_occ, n_desc, occ_names, desc_names, noc_of,
                            test_frac=0.02, seed=args.seed, verbose=False)
        prod = MFRecommender(k=args.k, epochs=args.epochs)
        prod.fit(full, np.random.default_rng(args.seed))
        prod.save(MODELS_DIR / "mf_production.npz")
        print(f"  saved -> {MODELS_DIR / 'mf_production.npz'}   "
              f"(val RMSE {prod.val_rmse:.4f}, stopped at epoch {prod.stopped_at})")
        print("\n  Now try the product:")
        print("     python scripts/12_query.py --interactive")

    print(f"\n  results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
