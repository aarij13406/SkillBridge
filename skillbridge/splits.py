"""
SkillBridge — Split Protocols
==============================
Every evaluation protocol in the proposal, implemented once, with
leakage guards baked in.

THE RULE:
    Nothing computed from the test set may ever touch the model.
    That includes graph statistics. If you compute "skill popularity"
    over the FULL edge list and then predict held-out edges, you have
    leaked — the popularity count already saw the answer.

    These functions return (train, test). Fit ONLY on train. Always.
"""

from __future__ import annotations
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# ═══════════════════════════════════════════════════════════════
# 1. HELD-OUT EDGE SPLIT  (Occupation Recommender — Aarij)
# ═══════════════════════════════════════════════════════════════

def split_edges(
    edges: pd.DataFrame,
    test_frac: float = 0.10,
    seed: int = 42,
    stratify_by: str | None = "occupation_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out `test_frac` of the occupation-descriptor edges.

    edges : DataFrame with at minimum
              occupation_id | descriptor_id | rating   (rating in 1..5)

    Stratified by occupation so that EVERY occupation contributes some
    held-out edges. Without this, a random split could leave some
    occupations with zero test edges and others with all of them,
    making per-query metrics (P@k, MRR) meaningless.

    Returns (train_edges, test_edges).

    ⚠  LEAKAGE GUARD: compute popularity, Jaccard, embeddings, EVERYTHING
       from `train_edges` only. `test_edges` is for scoring, nothing else.
    """
    strat = edges[stratify_by] if stratify_by else None

    # Occupations with only 1 edge cannot be stratified; keep them in train.
    if strat is not None:
        counts = edges[stratify_by].value_counts()
        singletons = counts[counts < 2].index
        mask_single = edges[stratify_by].isin(singletons)
        edges_strat = edges[~mask_single]
        edges_single = edges[mask_single]
        strat = edges_strat[stratify_by]
    else:
        edges_strat, edges_single = edges, edges.iloc[0:0]

    train, test = train_test_split(
        edges_strat, test_size=test_frac, random_state=seed, stratify=strat
    )
    train = pd.concat([train, edges_single], ignore_index=True)

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_query_sets(
    train_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    all_descriptor_ids: np.ndarray,
    core_threshold: int = 4,
    negative_threshold: int = 1,
):
    """
    Turn held-out edges into per-query (occupation) evaluation arrays.

    For each occupation that has at least one held-out edge:
      candidates = all descriptors NOT seen in train for that occupation
                   (i.e. the model must rank among things it hasn't been told)
      y_true     = 1 if the held-out rating >= core_threshold
                   0 if the held-out rating <= negative_threshold
                   (2..3 excluded from the BINARY task — ambiguous)
      y_graded   = the raw 1..5 rating, 0 for unseen  -> used by NDCG

    Yields dicts, one per occupation. Consumed by metrics.ranking_report().
    """
    train_pairs = set(zip(train_edges["occupation_id"], train_edges["descriptor_id"]))
    test_by_occ = test_edges.groupby("occupation_id")

    queries = []
    for occ, grp in test_by_occ:
        # Candidate pool: descriptors this occupation was NOT trained on
        cands = np.array([
            d for d in all_descriptor_ids if (occ, d) not in train_pairs
        ])
        if len(cands) == 0:
            continue

        rating_map = dict(zip(grp["descriptor_id"], grp["rating"]))

        y_bin, y_grad, keep = [], [], []
        for d in cands:
            r = rating_map.get(d)
            if r is None:
                # Unrated in test: treat as a sampled negative
                y_bin.append(0)
                y_grad.append(0.0)
                keep.append(True)
            elif r >= core_threshold:
                y_bin.append(1)
                y_grad.append(float(r))
                keep.append(True)
            elif r <= negative_threshold:
                y_bin.append(0)
                y_grad.append(float(r))
                keep.append(True)
            else:
                # Ambiguous (2-3): keep for NDCG, drop from binary
                y_bin.append(0)
                y_grad.append(float(r))
                keep.append(False)

        queries.append({
            "occupation_id": occ,
            "candidates":    cands,
            "y_true":        np.array(y_bin,  dtype=int),
            "y_graded":      np.array(y_grad, dtype=float),
            "binary_mask":   np.array(keep,   dtype=bool),
        })

    return queries


# ═══════════════════════════════════════════════════════════════
# 2. LEAVE-SKILL-OUT  (Skill-Gap Recommender — Anand)
# ═══════════════════════════════════════════════════════════════

def leave_one_skill_out(
    edges: pd.DataFrame,
    min_rating: int = 4,
    seed: int = 42,
) -> Iterator[Tuple[pd.DataFrame, int, int]]:
    """
    For each occupation, hide exactly ONE of its core descriptors and ask:
    can the model recover it?

    Yields (train_edges_for_this_query, occupation_id, held_out_descriptor_id).

    This directly answers the proposal's question: "given a current skill
    set and a target occupation, recommend the top-k skills to acquire."
    The person's "current skills" = the occupation's remaining descriptors.
    """
    rng = np.random.default_rng(seed)
    core = edges[edges["rating"] >= min_rating]

    for occ, grp in core.groupby("occupation_id"):
        if len(grp) < 2:
            continue   # need at least one to hide and one to keep
        held_idx = rng.choice(grp.index)
        held_desc = int(edges.loc[held_idx, "descriptor_id"])
        train = edges.drop(index=held_idx)
        yield train, occ, held_desc


# ═══════════════════════════════════════════════════════════════
# 3. STRATIFIED K-FOLD  (Labour Shortage Classifier — Irai)
# ═══════════════════════════════════════════════════════════════

def stratified_folds(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified 5-fold. With ~500 NOCs and a Surplus class of ~16, each
    test fold will contain roughly 3 Surplus examples.

    Irai: this is expected. Do NOT drop the class to make numbers look
    better. Report per-class F1 and explain the sample size. A
    well-characterized failure is a result.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    counts = pd.Series(y).value_counts()
    if counts.min() < n_folds:
        print(
            f"  [splits] WARNING: rarest class has {counts.min()} samples "
            f"but n_folds={n_folds}. Some folds will have <1 test sample "
            f"for that class. Class: {counts.idxmin()}"
        )
    yield from skf.split(X, y)


# ═══════════════════════════════════════════════════════════════
# 4. TEMPORAL SPLIT  (Salary & Regional Demand — Manivannan)
# ═══════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    date_col: str = "posting_date",
    cutoff: str = "2026-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train on postings BEFORE the cutoff, test on postings AFTER.

    ⚠  A random split here LEAKS. Salaries for the same job at the same
       employer appear in multiple months; a random split puts near-
       duplicates in both train and test and inflates R² badly.

    Job Bank spans Nov 2025 - Feb 2026, so the default cutoff of
    2026-01-01 gives roughly: train = Nov+Dec, test = Jan+Feb.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    cut = pd.Timestamp(cutoff)
    train = d[d[date_col] < cut]
    test = d[d[date_col] >= cut]

    print(
        f"  [splits] Temporal split at {cutoff}: "
        f"train={len(train):,} ({train[date_col].min().date()} to {train[date_col].max().date()}), "
        f"test={len(test):,} ({test[date_col].min().date()} to {test[date_col].max().date()})"
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cross_province_split(
    df: pd.DataFrame,
    held_out_provinces: list[str],
    province_col: str = "province",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Secondary generalization check from the proposal: train on some
    provinces, test on others. Answers "does the salary model transfer
    to a region it has never seen?"
    """
    test = df[df[province_col].isin(held_out_provinces)]
    train = df[~df[province_col].isin(held_out_provinces)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# LEAKAGE SELF-TEST — run this before trusting any number
# ═══════════════════════════════════════════════════════════════

def assert_no_edge_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Hard fail if any (occupation, descriptor) pair appears in both."""
    tr = set(zip(train["occupation_id"], train["descriptor_id"]))
    te = set(zip(test["occupation_id"], test["descriptor_id"]))
    overlap = tr & te
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} (occupation, descriptor) pairs appear "
            f"in BOTH train and test. Example: {list(overlap)[:3]}"
        )
    print(f"  [splits] ✓ No edge leakage. train={len(tr):,}  test={len(te):,}")


def assert_no_temporal_leakage(
    train: pd.DataFrame, test: pd.DataFrame, date_col: str = "posting_date"
) -> None:
    """Hard fail if any training posting is dated on/after the earliest test posting."""
    tr_max = pd.to_datetime(train[date_col]).max()
    te_min = pd.to_datetime(test[date_col]).min()
    if tr_max >= te_min:
        raise AssertionError(
            f"LEAKAGE: latest train date ({tr_max.date()}) is not before "
            f"earliest test date ({te_min.date()})."
        )
    print(f"  [splits] ✓ No temporal leakage. train ends {tr_max.date()}, test starts {te_min.date()}")
