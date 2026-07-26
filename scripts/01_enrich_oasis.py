"""
SkillBridge — 01: Build the Occupation-Descriptor Rating Matrix
================================================================
CSC 503 Data Mining, Summer 2026

WHAT THIS DOES
--------------
The OaSIS framework rates 900 Canadian occupational profiles across 181
competency descriptors, split across five files by category. We merge them
into a single occupation x descriptor RATING MATRIX.

THE KEY STRUCTURAL FACT (established empirically, see Step 3)
-------------------------------------------------------------
The OaSIS matrix is COMPLETE. Every occupation is rated on every descriptor:

        900 profiles  x  181 descriptors  =  162,900 cells
                          162,899 ratings observed

There is no missingness. Therefore this is NOT a link-prediction problem
("does this edge exist?" -- the answer is always yes, which is degenerate).

It is a MATRIX COMPLETION problem: hold out 10% of the observed CELLS and
predict their RATING. This is the classic collaborative-filtering setup
(MovieLens, with occupations in place of users).

The signal lives in the ratings, not the edges:

        rating 0  "not applicable"        26.7%
        rating 1                          16.1%
        rating 2                          21.9%
        rating 3                          21.2%
        rating 4                          10.3%   \\
        rating 5                           3.8%   /   CORE = 14.1%

The interesting binary question is: which 14% of descriptors are CORE to
this occupation? Predicting "always core" yields 14% precision, so this is
a genuinely imbalanced, genuinely hard task -- and ROC-AUC on it is
meaningful, unlike AUC on a saturated edge set.

THE BUG THIS SCRIPT FIXES
--------------------------
OaSIS codes carry a PROFILE SUFFIX:  12100.00, 12100.01, 12100.02 ...
Casting them to int collapses all three into NOC 12100, merging distinct
occupational profiles and creating duplicate (occupation, descriptor) cells.
The previous run produced 69,503 duplicate cells and a nonsensical 127.9%
"density".

We therefore keep BOTH:
    oasis_code  "12100.01"   <- the occupation node in the graph  (900)
    noc_code    "12100"      <- the join key to COPS / Job Bank   (516)

CATEGORIES INCLUDED (all on a consistent 1-5 scale, 0 = not applicable)
-----------------------------------------------------------------------
    skills               proficiency required
    abilities            proficiency required
    knowledge            level required
    personal-attributes  importance
    work-activities      complexity

EXCLUDED
--------
    work-context   Each descriptor has its OWN scale (frequency / duration /
                   yes-no). Merging it would silently corrupt every rating.
                   Saved separately as side features.
    interests      RIASEC categorical codes, not ratings.

OUTPUT
------
    data/processed/oasis_descriptors_long.csv
    data/processed/descriptor_lookup.csv
    data/processed/occupation_lookup.csv
    data/processed/oasis_work_context.csv

RUN
---
    python scripts/01_enrich_oasis.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skillbridge.config import set_all_seeds, SEED

set_all_seeds(SEED)

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════

from skillbridge.config import DATASETS_DIR, CLEAN_DIR

ROOT      = Path(__file__).resolve().parents[1]
RAW_OASIS = DATASETS_DIR / "raw" / "oasis"   # was hardcoded "data/raw/oasis", didn't match the shared config
PROCESSED = CLEAN_DIR                        # was hardcoded "data/processed", now uses the shared CLEAN_DIR
PROCESSED.mkdir(parents=True, exist_ok=True)

# Glob, never hardcode. The government ships these with inconsistent naming:
#   abilities_oasis_2025_v1.1.csv        (underscore, v1.1)
#   work-activities_oasis_2025_v1.1.csv  (hyphen)
#   knowledge_oasis_2025_v.1.1.csv       (v.1.1 -- extra dot)
RATING_CATEGORIES = {
    "Skills":              "skills*oasis*.csv",
    "Abilities":           "abilities*oasis*.csv",
    "Knowledge":           "knowledge*oasis*.csv",
    "Personal Attributes": "personal*attributes*oasis*.csv",
    "Work Activities":     "work*activities*oasis*.csv",
}
WORK_CONTEXT_PATTERN = "work*context*oasis*.csv"


def hr(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def find_one(pattern: str) -> Path | None:
    hits = sorted(RAW_OASIS.glob(pattern))
    if "context" not in pattern:
        hits = [h for h in hits if "context" not in h.stem.lower()]
    if not hits:
        return None
    if len(hits) > 1:
        print(f"    ! {len(hits)} files matched '{pattern}', using {hits[0].name}")
    return hits[0]


def read_oasis(path: Path) -> pd.DataFrame:
    """
    Read EVERYTHING as string.

    Critical: if pandas infers the code column as float, "12100.01" becomes
    12100.01 and "12100.00" becomes 12100.0 -- the profile suffix is mangled
    or lost entirely. Read as str; convert ratings to numeric afterwards.
    """
    for sep, enc in [(";", "utf-8-sig"), (",", "utf-8-sig"),
                     (";", "latin-1"),   (",", "latin-1")]:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise IOError(f"Could not parse {path.name}")


def split_id_and_rating_cols(df: pd.DataFrame):
    """Detect the code column and the name column; everything else is a rating."""
    code_col = name_col = None
    for c in df.columns:
        lc = str(c).lower()
        if code_col is None and "code" in lc:
            code_col = c
        elif name_col is None and any(k in lc for k in ("label", "title", "name")):
            name_col = c
    if code_col is None:
        code_col = df.columns[0]
    id_cols = [c for c in (code_col, name_col) if c is not None]
    rating_cols = [c for c in df.columns if c not in id_cols]
    return code_col, name_col, rating_cols


# ══════════════════════════════════════════════════════════════
# STEP 1: Load and melt each rating category
# ══════════════════════════════════════════════════════════════

hr("STEP 1: Loading OaSIS rating categories")

frames: list[pd.DataFrame] = []
name_lookup: dict[str, str] = {}
per_category: dict[str, int] = {}

for category, pattern in RATING_CATEGORIES.items():
    path = find_one(pattern)
    if path is None:
        print(f"  [ MISS ] {category:<20} nothing matched '{pattern}'")
        continue

    df = read_oasis(path)
    code_col, name_col, rating_cols = split_id_and_rating_cols(df)

    # ---- THE FIX ----------------------------------------------------
    # Keep the full profile code as a STRING. Do not cast to int.
    df["oasis_code"] = df[code_col].astype(str).str.strip()
    df = df[df["oasis_code"].str.len() > 0]
    df = df[~df["oasis_code"].str.lower().isin(["nan", "none", ""])]

    # The 5-digit NOC is the part before the decimal -- our join key to
    # COPS and Job Bank. The suffix identifies the specific profile.
    df["noc_code"] = (
        df["oasis_code"].str.split(".").str[0].str.strip().str.zfill(5)
    )
    # -----------------------------------------------------------------

    if name_col is not None:
        name_lookup.update(dict(zip(df["oasis_code"], df[name_col].astype(str))))

    long = df.melt(
        id_vars=["oasis_code", "noc_code"],
        value_vars=rating_cols,
        var_name="descriptor_name",
        value_name="rating",
    )
    long["rating"] = pd.to_numeric(long["rating"], errors="coerce")
    long = long.dropna(subset=["rating"])
    long["category"] = category

    frames.append(long[["oasis_code", "noc_code", "descriptor_name", "category", "rating"]])
    per_category[category] = long["descriptor_name"].nunique()

    n_prof = df["oasis_code"].nunique()
    n_noc  = df["noc_code"].nunique()
    print(f"  [  OK  ] {category:<20} {path.name}")
    print(f"           {n_prof:>4} profiles ({n_noc} distinct NOC) "
          f"x {len(rating_cols):>3} descriptors  ->  {len(long):>7,} ratings")

if not frames:
    raise SystemExit(f"\nNo OaSIS rating files found in {RAW_OASIS}")

edges = pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════
# STEP 2: Integrity check -- no duplicate cells
# ══════════════════════════════════════════════════════════════

hr("STEP 2: Integrity check")

dups = int(edges.duplicated(subset=["oasis_code", "descriptor_name"]).sum())
if dups:
    raise AssertionError(
        f"{dups:,} duplicate (occupation, descriptor) cells.\n"
        "The occupation key is not unique. Check the profile-suffix handling:\n"
        "  OaSIS codes look like 12100.00 / 12100.01 -- casting to int merges them."
    )
print("  ✓ No duplicate (occupation, descriptor) cells.")

n_prof = edges["oasis_code"].nunique()
n_noc  = edges["noc_code"].nunique()
n_desc = edges["descriptor_name"].nunique()
print(f"  Occupational profiles : {n_prof:>7,}   <- graph nodes")
print(f"  Distinct NOC codes    : {n_noc:>7,}   <- join key to COPS / Job Bank")
print(f"  Descriptors           : {n_desc:>7,}")
print(f"  Observed ratings      : {len(edges):>7,}")


# ══════════════════════════════════════════════════════════════
# STEP 3: Matrix completeness -- the structural fact
# ══════════════════════════════════════════════════════════════

hr("STEP 3: Matrix completeness")

cells    = n_prof * n_desc
observed = len(edges)
coverage = observed / cells

print(f"  Full matrix : {n_prof} x {n_desc} = {cells:>9,} cells")
print(f"  Observed    : {'':>{len(str(n_prof)) + len(str(n_desc)) + 3}}   {observed:>9,} ratings")
print(f"  Coverage    : {'':>{len(str(n_prof)) + len(str(n_desc)) + 3}}   {coverage:>9.1%}")
print()

if coverage > 0.99:
    print("  => The matrix is COMPLETE. Every occupation is rated on every descriptor.")
    print()
    print("     There is no missingness, so 'does this edge exist?' is NOT a")
    print("     question -- the answer is always yes. Binary LINK PREDICTION is")
    print("     degenerate by construction, at any density.")
    print()
    print("     TASK = MATRIX COMPLETION. Hold out 10% of CELLS, predict the")
    print("            RATING. Classic collaborative filtering (MovieLens setup).")
    print()
    print("     The signal is in the ratings, not the edges.")
else:
    print(f"  => Matrix is {coverage:.1%} observed. Both completion and link")
    print("     prediction are viable framings.")


# ══════════════════════════════════════════════════════════════
# STEP 4: Rating distribution -- the real class balance
# ══════════════════════════════════════════════════════════════

hr("STEP 4: Rating distribution")

dist  = edges["rating"].value_counts().sort_index()
total = len(edges)
labels = {0: "not applicable", 5: "critical"}

print(f"  {'Rating':<10} {'Count':>10} {'Share':>8}   {'':<18}")
print(f"  {'-' * 52}")
for r, c in dist.items():
    print(f"  {int(r):<10} {c:>10,} {c/total:>7.1%}   {labels.get(int(r), '')}")

core = int((edges["rating"] >= 4).sum())
neg  = int((edges["rating"] <= 1).sum())
amb  = total - core - neg

print(f"\n  CORE       (rating >= 4) : {core:>9,}  ({core/total:>5.1%})  <- positive class")
print(f"  IRRELEVANT (rating <= 1) : {neg:>9,}  ({neg/total:>5.1%})  <- negative class")
print(f"  AMBIGUOUS  (rating 2-3)  : {amb:>9,}  ({amb/total:>5.1%})")
print()
print("  => BINARY task: 'is this descriptor CORE to this occupation?'")
print(f"     Positive rate {core/total:.1%}. Predicting 'always core' yields only")
print(f"     {core/total:.1%} precision -- so ROC-AUC here is MEANINGFUL.")
print()
print("  => GRADED task: predict the 0-5 rating. Metrics: RMSE, NDCG@k.")
print(f"     Uses ALL {total:,} ratings, ambiguous included.")

print("\n  Descriptors per category:")
for cat, k in sorted(per_category.items(), key=lambda x: -x[1]):
    print(f"    {cat:<22} {k:>3}")


# ══════════════════════════════════════════════════════════════
# STEP 5: Integer IDs
# ══════════════════════════════════════════════════════════════

hr("STEP 5: Assigning integer IDs")

occupations = sorted(edges["oasis_code"].unique())
descriptors = sorted(edges["descriptor_name"].unique())

occ_to_id  = {o: i for i, o in enumerate(occupations)}
desc_to_id = {d: i for i, d in enumerate(descriptors)}

edges["occupation_id"]   = edges["oasis_code"].map(occ_to_id)
edges["descriptor_id"]   = edges["descriptor_name"].map(desc_to_id)
edges["occupation_name"] = edges["oasis_code"].map(name_lookup).fillna("")

edges = edges[[
    "occupation_id", "oasis_code", "noc_code", "occupation_name",
    "descriptor_id", "descriptor_name", "category", "rating",
]].sort_values(["occupation_id", "descriptor_id"]).reset_index(drop=True)

print(f"  occupation_id : 0 .. {len(occupations) - 1}")
print(f"  descriptor_id : 0 .. {len(descriptors) - 1}")


# ══════════════════════════════════════════════════════════════
# STEP 6: Work Context -> side features
# ══════════════════════════════════════════════════════════════

hr("STEP 6: Work Context (side features, NOT in the rating matrix)")

wc_path = find_one(WORK_CONTEXT_PATTERN)
if wc_path is None:
    print("  (not found -- optional, skipping)")
else:
    wc = read_oasis(wc_path)
    code_col, name_col, wc_cols = split_id_and_rating_cols(wc)
    wc["oasis_code"] = wc[code_col].astype(str).str.strip()
    wc["noc_code"]   = wc["oasis_code"].str.split(".").str[0].str.zfill(5)
    wc[["oasis_code", "noc_code"] + wc_cols].to_csv(
        PROCESSED / "oasis_work_context.csv", index=False
    )
    print(f"  Saved oasis_work_context.csv  "
          f"({wc['oasis_code'].nunique()} profiles x {len(wc_cols)} descriptors)")
    print("  Excluded from the rating matrix: each descriptor uses its own scale")
    print("  (frequency / duration / yes-no). Merging would corrupt the 1-5 ratings.")
    print("  -> Use as side features for the shortage classifier.")


# ══════════════════════════════════════════════════════════════
# STEP 7: Save
# ══════════════════════════════════════════════════════════════

hr("STEP 7: Saving")

edges.to_csv(PROCESSED / "oasis_descriptors_long.csv", index=False)
print(f"  oasis_descriptors_long.csv   {len(edges):>9,} rows")

pd.DataFrame({
    "descriptor_id":   [desc_to_id[d] for d in descriptors],
    "descriptor_name": descriptors,
}).merge(
    edges[["descriptor_name", "category"]].drop_duplicates(),
    on="descriptor_name", how="left",
).to_csv(PROCESSED / "descriptor_lookup.csv", index=False)
print(f"  descriptor_lookup.csv        {n_desc:>9,} rows")

edges[["occupation_id", "oasis_code", "noc_code", "occupation_name"]] \
    .drop_duplicates() \
    .to_csv(PROCESSED / "occupation_lookup.csv", index=False)
print(f"  occupation_lookup.csv        {n_prof:>9,} rows")


hr("DONE")
print(f"""
  Occupation x descriptor rating matrix built.

      {n_prof} occupational profiles  x  {n_desc} descriptors
      {observed:,} ratings   ({coverage:.1%} coverage -- COMPLETE matrix)
      {core:,} core ({core/total:.1%})  |  {neg:,} irrelevant  |  {amb:,} ambiguous

  TASK: matrix completion. Hold out 10% of cells, predict the rating.
        Binary view : is rating >= 4?   ({core/total:.1%} positive)
        Graded view : predict 0-5.      (RMSE, NDCG@k)

  Next:  python scripts/10_recommender.py
""")
