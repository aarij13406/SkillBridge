"""
SkillBridge — Baselines
========================
The proposal promises: "models are compared against named baselines on
held-out data rather than claimed without evidence."

These are those baselines. Every model must beat them or we say so.

A recommender that beats random but loses to popularity has learned
nothing. That is the bar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class BaseRecommender:
    """Common interface. Every model — baseline or not — implements this."""
    name: str = "base"

    def fit(self, train_edges: pd.DataFrame, n_occupations: int, n_descriptors: int):
        raise NotImplementedError

    def score(self, occupation_id: int, descriptor_ids: np.ndarray) -> np.ndarray:
        """Return a score per candidate descriptor. Higher = better fit."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# BASELINE 1: Random
# ═══════════════════════════════════════════════════════════════

class RandomRecommender(BaseRecommender):
    """
    The floor. If a model cannot beat this, it is broken.
    Expected ROC-AUC = 0.50 by construction.
    """
    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fit(self, train_edges, n_occupations, n_descriptors):
        return self

    def score(self, occupation_id, descriptor_ids):
        return self.rng.random(len(descriptor_ids))


# ═══════════════════════════════════════════════════════════════
# BASELINE 2: Descriptor Popularity
# ═══════════════════════════════════════════════════════════════

class PopularityRecommender(BaseRecommender):
    """
    Recommend whatever is most commonly required across all occupations.
    Ignores the query entirely — same ranking for everyone.

    This is the baseline that MATTERS. In a dense graph, popularity is
    devastatingly strong, because "Communication is important" is true
    for almost every occupation. If matrix factorization cannot beat
    popularity, it has not learned anything occupation-specific.
    """
    name = "Popularity"

    def fit(self, train_edges, n_occupations, n_descriptors):
        # Mean rating per descriptor, computed on TRAIN ONLY.
        pop = train_edges.groupby("descriptor_id")["rating"].mean()
        self.scores_ = np.zeros(n_descriptors)
        self.scores_[pop.index.values] = pop.values
        return self

    def score(self, occupation_id, descriptor_ids):
        return self.scores_[descriptor_ids]


# ═══════════════════════════════════════════════════════════════
# BASELINE 3: Jaccard / item-based collaborative filtering
# ═══════════════════════════════════════════════════════════════

class JaccardRecommender(BaseRecommender):
    """
    Occupation-occupation similarity via shared descriptors, then
    propagate: "occupations similar to yours also need X."

        sim(o1, o2) = |D(o1) ∩ D(o2)| / |D(o1) ∪ D(o2)|
        score(o, d)  = Σ_{o' ≠ o}  sim(o, o') · rating(o', d)
                       ─────────────────────────────────────
                              Σ_{o' ≠ o}  sim(o, o')

    This is classic memory-based CF. It is a genuinely competitive
    baseline and the one most likely to embarrass a poorly-tuned
    matrix factorization.
    """
    name = "Jaccard-CF"

    def __init__(self, core_threshold: int = 4, top_n_neighbours: int = 50):
        self.core_threshold = core_threshold
        self.top_n = top_n_neighbours

    def fit(self, train_edges, n_occupations, n_descriptors):
        self.n_occ = n_occupations
        self.n_desc = n_descriptors

        # Binary occupation x descriptor matrix (core descriptors only)
        core = train_edges[train_edges["rating"] >= self.core_threshold]
        B = np.zeros((n_occupations, n_descriptors), dtype=np.float32)
        B[core["occupation_id"].values, core["descriptor_id"].values] = 1.0

        # Full rating matrix (for the weighted propagation step)
        R = np.zeros((n_occupations, n_descriptors), dtype=np.float32)
        R[train_edges["occupation_id"].values,
          train_edges["descriptor_id"].values] = train_edges["rating"].values

        # Jaccard: intersection / union, vectorised
        inter = B @ B.T                                   # (n_occ, n_occ)
        sizes = B.sum(axis=1, keepdims=True)              # (n_occ, 1)
        union = sizes + sizes.T - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(sim, 0.0)                        # never your own neighbour

        # Keep only top-N neighbours per occupation (denoises + speeds up)
        if self.top_n < n_occupations:
            cutoff = np.partition(sim, -self.top_n, axis=1)[:, -self.top_n][:, None]
            sim = np.where(sim >= cutoff, sim, 0.0)

        self.sim_ = sim
        self.R_ = R
        return self

    def score(self, occupation_id, descriptor_ids):
        w = self.sim_[occupation_id]                      # (n_occ,)
        denom = w.sum()
        if denom == 0:
            return np.zeros(len(descriptor_ids))
        numer = w @ self.R_[:, descriptor_ids]            # (n_cands,)
        return numer / denom


# ═══════════════════════════════════════════════════════════════
# BASELINE 4 (Skill-Gap): Most-common-missing
# ═══════════════════════════════════════════════════════════════

class MostCommonMissingRecommender(BaseRecommender):
    """
    The proposal's named baseline for the Skill-Gap component (Anand).
    Given what you already have, recommend the globally most common
    descriptor you are missing.
    """
    name = "MostCommonMissing"

    def fit(self, train_edges, n_occupations, n_descriptors):
        core = train_edges[train_edges["rating"] >= 4]
        freq = core["descriptor_id"].value_counts()
        self.freq_ = np.zeros(n_descriptors)
        self.freq_[freq.index.values] = freq.values
        return self

    def score(self, occupation_id, descriptor_ids):
        return self.freq_[descriptor_ids]


# ═══════════════════════════════════════════════════════════════
# Registry — so every script can loop over the same set
# ═══════════════════════════════════════════════════════════════

def get_baselines(seed: int = 42) -> "list[BaseRecommender]":
    return [
        RandomRecommender(seed=seed),
        PopularityRecommender(),
        JaccardRecommender(),
    ]
