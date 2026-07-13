"""
SkillBridge — Shared Configuration
===================================
CSC 503 Data Mining, Summer 2026

EVERY component imports from here. Do not hardcode paths or seeds
anywhere else. If a path is wrong, fix it HERE, once.

Usage:
    from skillbridge.config import SEED, CLEAN_DIR, OUTPUT_DIR
"""

from pathlib import Path
import os
import random
import numpy as np

# ═══════════════════════════════════════════════════════════════
# REPRODUCIBILITY — non-negotiable
# ═══════════════════════════════════════════════════════════════
# Every model, every split, every shuffle uses this seed.
# If two people report different numbers on the same model,
# the seed is the first thing we check.

SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Call this at the top of EVERY script. No exceptions."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════
# PATHS — edit ONLY this block for your machine
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(r"C:\Users\AARIJ\Downloads\uvic\CSC-503-DataMining\SkillBridge")

DATASETS_DIR = PROJECT_ROOT / "datasets"
CLEAN_DIR    = DATASETS_DIR / "clean"
RAW_OASIS_DIR = DATASETS_DIR          # where the OaSIS CSVs live
OUTPUT_DIR   = PROJECT_ROOT / "output" / "csc503"
RESULTS_DIR  = OUTPUT_DIR / "results"   # metric JSONs, one per experiment
FIGURES_DIR  = OUTPUT_DIR / "figures"

for _d in (OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CLEAN DATA FILES (from the CSC501 ETL — already built)
# ═══════════════════════════════════════════════════════════════

F_OASIS_SKILLS      = CLEAN_DIR / "oasis_skills_long.csv"
F_OASIS_DESCRIPTORS = CLEAN_DIR / "oasis_descriptors_long.csv"   # NEW: enriched
F_OASIS_DESC_TEXT   = CLEAN_DIR / "oasis_skill_descriptions.csv"
F_NOC_LOOKUP        = CLEAN_DIR / "noc_lookup.csv"
F_COPS              = CLEAN_DIR / "cops_projections_clean.csv"
F_JOBBANK           = CLEAN_DIR / "jobbank_clean.csv"
F_LINKEDIN_POSTS    = CLEAN_DIR / "linkedin_postings_canada.csv"
F_LINKEDIN_SKILLS   = CLEAN_DIR / "linkedin_skills_canada.csv"


# ═══════════════════════════════════════════════════════════════
# TASK CONSTANTS
# ═══════════════════════════════════════════════════════════════

# --- Occupation Recommender (Aarij) ---
# OaSIS ratings are 1-5. We binarize for AUC / Precision@k:
CORE_THRESHOLD    = 4    # rating >= 4  -> "core descriptor" (positive)
NEGATIVE_THRESHOLD = 1   # rating <= 1  -> "irrelevant"      (negative)
# Ratings of 2-3 are "ambiguous" and excluded from the binary
# evaluation, but ARE used for NDCG (graded relevance) and RMSE.

HELD_OUT_FRACTION = 0.10   # 10% of edges held out, per the proposal
TOP_K_VALUES      = (5, 10)

# --- Labour Shortage Classifier (Irai) ---
# COPS has 5 raw classes. Collapse to 3 per the proposal.
SHORTAGE_CLASS_MAP = {
    "Strong risk of Shortage":  "Shortage",
    "Moderate risk of Shortage": "Shortage",
    "Balance":                   "Balance",
    "Moderate risk of Surplus":  "Surplus",
    "Strong risk of Surplus":    "Surplus",
}
SHORTAGE_CLASSES = ["Shortage", "Balance", "Surplus"]
# WARNING: Surplus has ~16 occupations total. In stratified 5-fold that
# is ~3 test samples per fold. Macro-F1 will be BOUND by this class.
# Report it honestly; do not hide it.

N_FOLDS = 5

# --- Salary & Regional Demand (Manivannan) ---
# Split by posting date, NOT randomly. Random split leaks.
TEMPORAL_SPLIT_DATE = "2026-01-01"   # train: before, test: on/after

# --- Fairness audit (Dharnesh) ---
# NOC TEER level = 2nd digit of the 5-digit NOC code.
# 0=management, 1=university degree, ... 5=no formal education.
TEER_FROM_NOC = lambda noc: str(noc).zfill(5)[1]


# ═══════════════════════════════════════════════════════════════
# SANITY CHECK
# ═══════════════════════════════════════════════════════════════

def check_data_present(require_enriched: bool = False) -> None:
    """Fail loudly and early if a file is missing."""
    required = [
        F_NOC_LOOKUP, F_COPS, F_JOBBANK,
        F_LINKEDIN_POSTS, F_OASIS_SKILLS,
    ]
    if require_enriched:
        required.append(F_OASIS_DESCRIPTORS)

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files:\n  " +
            "\n  ".join(str(p) for p in missing) +
            "\n\nRun the ETL (01_clean_all_data.py) first, and for the "
            "enriched descriptor graph run 06_enrich_oasis.py."
        )
    print(f"  [config] All required data present in {CLEAN_DIR}")


if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"CLEAN_DIR    = {CLEAN_DIR}")
    print(f"OUTPUT_DIR   = {OUTPUT_DIR}")
    print(f"SEED         = {SEED}")
    check_data_present()
