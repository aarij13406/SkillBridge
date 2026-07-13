"""
scripts/12_query.py
===================
SkillBridge, Component 1: THE PRODUCT.
Owner: Muhammad Aarij (V01096775).  CSC 503 Data Mining, Summer 2026.

Competencies in. Ranked Canadian occupations out.


WHICH MECHANISM SHIPS, AND WHY
-------------------------------
Three ways to turn a competency profile into a ranking over 900 occupations. Only one
of them is supported by evidence, and scripts/11_coldstart.py is the evidence.

  mf_impute        Fold the person into the latent space, RECONSTRUCT all 181 of their
    [PRIMARY]      competency ratings, then Pearson-correlate against each occupation's
                   true 181-vector.
                   The 181 - |S| unrevealed dimensions are IMPUTED by the model.
                   This is the mechanism the cold-start experiment selects.

  raw_correlation  Pearson r on ONLY the competencies the person revealed.
    [BASELINE]     NO LEARNING AT ALL. This is what a competent analyst does without a
                   model, and it is the bar mf_impute has to clear. Shown alongside so
                   the reader can see the model earning (or failing to earn) its place.

  value            "Which occupations VALUE the competencies you named?"
    [DIAGNOSTIC]   IGNORES YOUR RATINGS BY CONSTRUCTION.
                   This was a BUG in an early version: the tool folded the person in,
                   printed ||p_you||, and then never used it. Rating Time Management
                   2/5 versus 5/5 produced identical output. The defect was caught not
                   by a test but by a domain-intuition check: a query declaring weak
                   time management returned senior-management roles, which is
                   implausible. Kept and LABELLED, so the difference between "which
                   occupations value X" and "which occupations fit me" is visible
                   rather than silently conflated.


THE FOLD-IN
-----------
You are not in the training set. You have no latent vector. We infer one: hold Q and
b_i FIXED (learned from 900 occupations) and solve by ridge over your revealed skills S:

    min_{p, b}  SUM_{i in S} ( r_i - mu - b_i - b - p . q_i )^2 + lambda(||p||^2 + b^2)

    X = [ Q_S | 1 ]      y = r_S - mu - b_S
    [ p ; b ] = ( X^T X + lambda I )^{-1} X^T y        <-- CLOSED FORM

One linear solve. lambda is NOT inherited from training: training saw ~163 observations
per occupation, a cold user gives 5-20, and the regulariser must reflect that. It is
tuned in 11_coldstart.py by simulating cold start on occupations whose true latent
vectors we already know.


HOW MANY SKILLS DO YOU NEED?
-----------------------------
From the cold-start evaluation, on 90 occupations the model has NEVER seen. Numbers are
filled in by that script; see results/coldstart_summary.csv for the current run.

The honest framing is NOT "we beat random". Random is the floor. The bar is
raw_correlation -- comparing your revealed ratings directly to each occupation's, with
no model at all. The model is only worth having where it beats THAT.


THE CLOSED VOCABULARY  (a real limitation; say it before a reviewer does)
-------------------------------------------------------------------------
This model knows exactly 181 competencies, and they are OaSIS's, not the world's:
"Reading Comprehension", "Deductive Reasoning", "Manual Dexterity", "Negotiating".

It does NOT know "Python", "Excel", "SQL", "React", "patient care". A real person types
those. This is a SELF-ASSESSMENT INSTRUMENT over a fixed taxonomy, which is exactly how
ESDC's own career tools work: a legitimate product, but a bounded one.

Open-vocabulary input needs a semantic bridge from arbitrary skill strings onto the
OaSIS taxonomy. That bridge is Component 5 (NLP Skill Extraction, Nalluraj). Wiring it
to this recommender is the primary integration task, and it is the integration lead's
job -- mine.


USAGE
    python scripts/12_query.py --interactive
    python scripts/12_query.py --skills "Instructing:5, Social Perceptiveness:5, Active Listening:4"
    python scripts/12_query.py --like "Registered nurses"     # borrow a real profile (sanity check)
    python scripts/12_query.py --list --search comm           # browse the 181

PREREQUISITE
    python scripts/10_recommender.py     # trains and saves models/mf_production.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from skillbridge.config import CORE_THRESHOLD
from skillbridge.models import OCC, DSC, RAT, MFRecommender, fold_in, load_raw

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "mf_production.npz"

DEFAULT_RATING = 4.0     # naming a skill without a rating implies you are good at it
FOLDIN_REG = 1.0         # overridden by results/coldstart_summary.csv if present


# ═══════════════════════════════════════════════════════════════════════════

def match_descriptor(query: str, desc_names: dict):
    q = query.strip().lower()
    if not q:
        return None
    items = [(int(i), str(n)) for i, n in desc_names.items()]
    for i, n in items:
        if n.lower() == q:
            return i, n
    for pool in ([(i, n) for i, n in items if n.lower().startswith(q)],
                 [(i, n) for i, n in items if q in n.lower()]):
        if len(pool) == 1:
            return pool[0]
        if len(pool) > 1:
            print(f"  ! '{query}' is ambiguous:")
            for i, n in pool[:8]:
                print(f"       {n}")
            return None
    print(f"  ! '{query}' matched nothing.")
    print("    This model knows only OaSIS's 181 competencies, not open vocabulary.")
    print(f"    Browse:  python scripts/12_query.py --list --search {q}")
    return None


def parse_skills(text: str, desc_names: dict):
    ids, ratings, shown = [], [], []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, val = part.rsplit(":", 1)
            try:
                r = float(val)
            except ValueError:
                name, r = part, DEFAULT_RATING
        else:
            name, r = part, DEFAULT_RATING
        if m := match_descriptor(name, desc_names):
            i, n = m
            ids.append(i)
            ratings.append(float(np.clip(r, 0, 5)))
            shown.append(f"{n:<46} {ratings[-1]:.0f}/5")
    return np.array(ids, dtype=int), np.array(ratings, dtype=float), shown


def pearson_rows(A, v):
    """Pearson r between every row of A and the vector v.

    The centring is not cosmetic. Without it, occupations that rate EVERYTHING highly
    (surgeons, senior managers) correlate with every possible profile. That popularity
    bias is what made the naive scorer return ophthalmologists for an early childhood
    educator's profile.
    """
    Ac = A - A.mean(axis=1, keepdims=True)
    vc = v - v.mean()
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(vc)
    out = np.zeros(A.shape[0])
    ok = denom > 1e-9
    out[ok] = (Ac[ok] @ vc) / denom[ok]
    return out


def show(title, scores, occ_names, noc_of, top=10, note=None, mark=None):
    order = np.argsort(-scores)[:top]
    print(f"\n  {title}")
    if note:
        print(f"  {note}")
    print("  " + "-" * 78)
    for r, u in enumerate(order, 1):
        star = " *" if mark is not None and u == mark else ""
        print(f"  {r:>2}. {str(occ_names.get(int(u), u))[:52]:<54} "
              f"{scores[u]:>6.3f}   NOC {noc_of.get(int(u), '')}{star}")
    spread = float(scores[order[0]] - scores[order[-1]])
    tag = "   <-- NARROW. These are near-ties, not a confident ranking." if spread < 0.05 else ""
    print(f"      spread across the top {top}: {spread:.3f}{tag}")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", type=str, default=None)
    ap.add_argument("--like", type=str, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--search", type=str, default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--reg", type=float, default=FOLDIN_REG)
    args = ap.parse_args()

    df, n_occ, n_desc, occ_names, desc_names, noc_of = load_raw(PROCESSED_DIR)

    if args.list:
        lut = pd.read_csv(PROCESSED_DIR / "descriptor_lookup.csv")
        if args.search:
            lut = lut[lut["descriptor_name"].str.contains(args.search, case=False, na=False)]
        print(f"\n  {len(lut)} of 181 competencies\n")
        for cat, g in lut.groupby("category"):
            print(f"  {str(cat).upper()}")
            for n in sorted(g["descriptor_name"]):
                print(f"     {n}")
            print()
        return

    if not MODEL_PATH.exists():
        raise SystemExit(f"\n  Model not found: {MODEL_PATH}\n"
                         f"  Train it:  python scripts/10_recommender.py\n")
    mf = MFRecommender.load(MODEL_PATH)
    k = mf.P.shape[1]

    R_full = np.zeros((n_occ, n_desc))
    R_full[df[OCC].to_numpy(), df[DSC].to_numpy()] = df[RAT].to_numpy()

    print("=" * 84)
    print("  SkillBridge  |  competencies in -> Canadian occupations out")
    print("=" * 84)
    print(f"  model : MF, k={k}, over 900 occupations x {n_desc} OaSIS competencies")
    print(f"          (35,674 parameters reconstructing 162,900 ratings: a 4.6x compression)")

    # ---- profile ------------------------------------------------------------
    u0 = None
    if args.like:
        hit = [(int(u), str(n)) for u, n in occ_names.items()
               if args.like.lower() in str(n).lower()]
        if not hit:
            raise SystemExit(f"  No occupation matching '{args.like}'")
        u0, name0 = hit[0]
        core = df[(df[OCC] == u0) & (df[RAT] >= CORE_THRESHOLD)]
        desc_ids, ratings = core[DSC].to_numpy(), core[RAT].to_numpy()
        shown = [f"{desc_names[int(i)]:<46} {r:.0f}/5" for i, r in zip(desc_ids, ratings)]
        print(f"\n  Borrowing the core profile of: {name0}")
        print("  (a sanity check: does the model put this occupation back near the top?)")
    elif args.interactive:
        print("\n  Name your competencies, comma separated. Optional 0-5 rating after a colon.")
        print("  Example:  Instructing:5, Social Perceptiveness:5, Active Listening:4")
        print("  Browse them:  python scripts/12_query.py --list\n")
        text = input("  > ").strip()
        if not text:
            raise SystemExit("  nothing entered.")
        desc_ids, ratings, shown = parse_skills(text, desc_names)
    elif args.skills:
        desc_ids, ratings, shown = parse_skills(args.skills, desc_names)
    else:
        raise SystemExit("  Use --skills, --like, --interactive, or --list.")

    if len(desc_ids) < 2:
        raise SystemExit("\n  Need at least 2 recognised competencies.")

    n_S = len(desc_ids)
    print(f"\n  YOUR PROFILE  ({n_S} of {n_desc} competencies)")
    print("  " + "-" * 78)
    for s in shown[:25]:
        print(f"     {s}")
    if len(shown) > 25:
        print(f"     ... and {len(shown) - 25} more")

    # ---- fold in ------------------------------------------------------------
    p_you, b_you = fold_in(mf, desc_ids, ratings, reg=args.reg)
    print(f"\n  FOLDED IN   ||p_you|| = {np.linalg.norm(p_you):.3f}   b_you = {b_you:+.3f}")
    print(f"  {n_S} observations -> {k + 1} unknowns, solved in closed form by ridge "
          f"(lambda = {args.reg}).")
    if n_S < k + 1:
        print(f"  Underdetermined ({n_S} < {k+1}), so p_you is recovered only within the SPAN")
        print(f"  of your revealed competencies; ridge shrinks the rest toward zero. That is a")
        print(f"  real limitation, not a fatal one -- see results/coldstart_summary.csv for how")
        print(f"  placement quality actually degrades with |S|.")

    # ---- the three rankings -------------------------------------------------
    d_all = np.arange(n_desc)
    r_hat = np.clip(mf.mu + b_you + mf.bi[d_all] + mf.Q[d_all] @ p_you, 0, 5)

    s_impute = pearson_rows(R_full, r_hat)                              # PRIMARY
    s_rawcorr = pearson_rows(R_full[:, desc_ids], ratings)              # BASELINE

    all_u = np.arange(n_occ)
    named = np.zeros(n_occ)
    for d in desc_ids:
        named += mf.score(all_u, np.full(n_occ, int(d)))
    named /= n_S
    own = np.zeros(n_occ)
    for d in range(n_desc):
        own += mf.score(all_u, np.full(n_occ, d))
    own /= n_desc
    s_value = named - own                                               # DIAGNOSTIC

    show("[PRIMARY]  mf_impute -- your reconstructed 181-competency profile vs each occupation's",
         s_impute, occ_names, noc_of, args.top, mark=u0,
         note=f"   USES your ratings. {n_desc - n_S} of the 181 compared dimensions are IMPUTED\n"
              f"   by the model. This is the mechanism the cold-start experiment selects.")

    show("[BASELINE] raw_correlation -- your revealed ratings only. NO MODEL AT ALL.",
         s_rawcorr, occ_names, noc_of, args.top, mark=u0,
         note=f"   Compares only the {n_S} competencies you named. This is what an analyst does\n"
              f"   WITHOUT machine learning. If PRIMARY does not beat it, the model is not\n"
              f"   earning its place. See results/coldstart_summary.csv for where it does.")

    show("[DIAGNOSTIC] value -- which occupations VALUE these competencies?",
         s_value, occ_names, noc_of, 5, mark=u0,
         note="   IGNORES YOUR RATINGS by construction. A DIFFERENT question, and the source\n"
              "   of an early bug. Shown so the two questions are not silently conflated.")

    # ---- what the model inferred about you ----------------------------------
    told = set(desc_ids.tolist())
    print("\n  WHAT THE MODEL INFERS ABOUT YOU")
    print("  " + "-" * 78)
    print("  Top competencies it predicts you are strong in. Anything WITHOUT '(told)' was")
    print("  never revealed: the model is inferring it purely from latent structure.")
    print()
    for d in np.argsort(-r_hat)[:10]:
        tag = "  (told)" if int(d) in told else ""
        print(f"     {str(desc_names.get(int(d), d))[:52]:<54} {r_hat[d]:.2f}/5{tag}")

    # ---- sanity check -------------------------------------------------------
    if u0 is not None:
        print(f"\n  SANITY CHECK: where did '{str(occ_names[u0])[:44]}' land?")
        print("  " + "-" * 78)
        for label, s in (("mf_impute      ", s_impute),
                         ("raw_correlation", s_rawcorr),
                         ("value          ", s_value)):
            r = int(np.where(np.argsort(-s) == u0)[0][0]) + 1
            print(f"     {label}  #{r}/900")
        print("\n  The profile came FROM that occupation, so a working mechanism must rank it")
        print("  near the top. If it does not, something is wrong with the mechanism, not")
        print("  with the occupation.")

    print()


if __name__ == "__main__":
    main()
