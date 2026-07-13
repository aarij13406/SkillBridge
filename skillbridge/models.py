"""
skillbridge/models.py
=====================
Every model in the Occupation Recommender ladder, plus the data structures and the
cold-start fold-in solver.

Lives in the package (not in a script) because THREE entry points need it:

    scripts/10_recommender.py   the ladder + evaluation
    scripts/11_coldstart.py     fold-in evaluation on unseen occupations
    scripts/12_query.py         the product: skills in, ranked occupations out


THE SCORE MODEL, AND THE ALGEBRA THAT DRIVES THE WHOLE COMPONENT
----------------------------------------------------------------
        x(u, i) = mu + b_u + b_i + p_u . q_i

    u = occupation      p_u = its k-dim latent vector    b_u = its general "level"
    i = descriptor      q_i = its k-dim latent vector    b_i = its general "level"

Rank OCCUPATIONS for a fixed descriptor i  (PRIMARY, the proposal's task):

        x(u,i) - x(v,i) = (b_u - b_v) + (p_u - p_v) . q_i
                          ^^^^^^^^^^^
        b_i CANCELS. Constant across every candidate in the query.

Rank DESCRIPTORS for a fixed occupation u  (SECONDARY, the Skill-Gap direction):

        x(u,i) - x(u,j) = (b_i - b_j) + p_u . (q_i - q_j)
                          ^^^^^^^^^^^
        b_u CANCELS.

  ==> WHICH BIAS TERM IS IDENTIFIABLE IS DETERMINED BY THE RANKING AXIS.

A PAIRWISE loss only ever sees differences, so it can only learn the bias that
survives the subtraction. A POINTWISE loss takes no difference and therefore
cancels nothing: it learns both.

That single fact explains every row of the results table:

    MF        pointwise    nothing cancels    learns b_u AND b_i    wins both axes
    BPR-D     pairwise     b_i cancels        learns b_u only       wins PRIMARY
    BPR-O     pairwise     b_u cancels        learns b_i only       wins SECONDARY

MF does not beat BPR because pointwise losses are better. MF beats BPR because it
is the only model in the ladder with a COMPLETE PARAMETERISATION. The MFNoBi
ablation below isolates exactly that: strip b_i from MF and its primary-axis
performance should collapse toward BPR-D's, proving the gap is the missing bias
term and not the loss function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from skillbridge.config import CORE_THRESHOLD, NEGATIVE_THRESHOLD

OCC, DSC, RAT = "occupation_id", "descriptor_id", "rating"
RATING_MIN, RATING_MAX = 0.0, 5.0


def sigmoid(x):
    """Overflow-safe logistic."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Matrix:
    n_occ: int
    n_desc: int
    train: pd.DataFrame
    test: pd.DataFrame
    R: np.ndarray              # n_occ x n_desc, TRAIN ratings, 0 where unobserved
    M: np.ndarray              # n_occ x n_desc, 1.0 where observed in TRAIN
    occ_names: dict
    desc_names: dict
    noc_of: dict               # occupation_id -> noc_code (for the NOC-group ground truth)


def load_raw(processed_dir: Path):
    """Load the rating matrix. dtype=str on the codes is NOT optional: casting
    oasis_code to a number collapses 12100.00 / 12100.01 / 12100.02 into 12100,
    merging 900 profiles into 516 and manufacturing 69,503 duplicate cells."""
    path = processed_dir / "oasis_descriptors_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}\nRun: python scripts/01_enrich_oasis.py")

    df = pd.read_csv(path, dtype={"oasis_code": str, "noc_code": str})
    if missing := {OCC, DSC, RAT} - set(df.columns):
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df[OCC] = df[OCC].astype(np.int32)
    df[DSC] = df[DSC].astype(np.int32)
    df[RAT] = df[RAT].astype(np.float64)

    if dups := int(df.duplicated(subset=[OCC, DSC]).sum()):
        raise ValueError(f"{dups:,} duplicate cells. Re-run 01_enrich_oasis.py.")

    n_occ = int(df[OCC].max()) + 1
    n_desc = int(df[DSC].max()) + 1
    occ_names = dict(zip(df[OCC], df["occupation_name"])) if "occupation_name" in df else {}
    desc_names = dict(zip(df[DSC], df["descriptor_name"])) if "descriptor_name" in df else {}
    noc_of = dict(zip(df[OCC], df["noc_code"])) if "noc_code" in df else {}
    return df, n_occ, n_desc, occ_names, desc_names, noc_of


def build_matrix(df, n_occ, n_desc, occ_names, desc_names, noc_of,
                 test_frac, seed, verbose=True):
    from skillbridge.splits import split_edges, assert_no_edge_leakage

    # Stratify the CELL split by rating so the held-out set carries the same ordinal
    # distribution as train. Without this the 3.8% of cells rated 5 land unevenly
    # across occupations and NDCG becomes noisy.
    train, test = split_edges(df, test_frac=test_frac, seed=seed, stratify_by=RAT)
    assert_no_edge_leakage(train, test)

    R = np.zeros((n_occ, n_desc))
    M = np.zeros((n_occ, n_desc))
    R[train[OCC], train[DSC]] = train[RAT]
    M[train[OCC], train[DSC]] = 1.0

    if verbose:
        print(f"  split    : {len(train):,} train / {len(test):,} test   "
              f"(~{M.sum(axis=1).mean():.0f} of {n_desc} ratings seen per occupation)")
    return Matrix(n_occ, n_desc, train, test, R, M, occ_names, desc_names, noc_of)


# ═══════════════════════════════════════════════════════════════════════════
# BASE
# ═══════════════════════════════════════════════════════════════════════════

class Recommender:
    name = "base"
    on_rating_scale = False
    target_axis = "-"

    def fit(self, mx: Matrix, rng):
        raise NotImplementedError

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════

class RandomRecommender(Recommender):
    """The floor. Its only job is to prove the metric code is honest: Random MUST
    land at AUC ~= 0.50. If it does not, the bug is in metrics.py, not here."""
    name = "random"
    on_rating_scale = True

    def fit(self, mx, rng):
        self._rng = rng
        return self

    def score(self, u, i):
        return self._rng.uniform(RATING_MIN, RATING_MAX, size=len(u))


class DescriptorPopularity(Recommender):
    """score(u,i) = mean TRAIN rating of descriptor i.

    CONSTANT across occupations, so it ranks NOTHING on the PRIMARY axis: all ~90
    candidates receive an identical score and the metrics are tie-break noise.
    (Exactly why it scored BELOW Random on primary NDCG. A baseline that loses to
    random is not a baseline, it is the wrong baseline.)

    It is the CORRECT baseline for the SECONDARY axis.
    """
    name = "desc_popularity"
    on_rating_scale = True
    target_axis = "secondary"

    def fit(self, mx, rng):
        n = mx.M.sum(axis=0)
        s = (mx.R * mx.M).sum(axis=0)
        g = float(s.sum() / max(n.sum(), 1.0))
        self.mean = np.where(n > 0, s / np.maximum(n, 1.0), g)
        return self

    def score(self, u, i):
        return self.mean[i]


class OccupationPopularity(Recommender):
    """score(u,i) = mean TRAIN rating GIVEN BY occupation u.

    Captures the LEVEL effect: surgeons and senior managers rate nearly everything
    highly; labourers rate most things low. CONSTANT across descriptors, so it
    cannot rank SECONDARY -- but it is the right baseline for PRIMARY, and a strong
    one. Beating it requires learning something genuinely descriptor-specific about
    an occupation, not merely its overall level.
    """
    name = "occ_popularity"
    on_rating_scale = True
    target_axis = "primary"

    def fit(self, mx, rng):
        n = mx.M.sum(axis=1)
        s = (mx.R * mx.M).sum(axis=1)
        g = float(s.sum() / max(n.sum(), 1.0))
        self.mean = np.where(n > 0, s / np.maximum(n, 1.0), g)
        return self

    def score(self, u, i):
        return self.mean[u]


class JaccardRecommender(Recommender):
    """Occupation-occupation neighbourhood CF. The proposal's 'content-based
    similarity baseline'.

    Similarity is Jaccard over TRAIN CORE SETS (rating >= 4), not over full rating
    vectors. Correlating full vectors would be dominated by the 26.7% of cells rated
    0, which every occupation shares, making all 900 look alike.

    KNOWN FAILURE MODE (see the sparsity sweep): core rate is 14.1%, so at 90%
    holdout each occupation has ~18 observed ratings => ~2.5 core descriptors.
    Jaccard between two sets of size 2.5 is noise, and the model drops BELOW the
    occupation-mean baseline. Memory-based CF does not degrade in cold start; it
    fails outright.
    """
    name = "jaccard_cf"
    on_rating_scale = True
    target_axis = "both"

    def __init__(self, topk=30):
        self.topk = topk

    def fit(self, mx, rng):
        B = ((mx.R >= CORE_THRESHOLD) & (mx.M > 0)).astype(np.float64)
        inter = B @ B.T
        sizes = B.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - inter
        S = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        np.fill_diagonal(S, 0.0)

        if self.topk < S.shape[0] - 1:
            cut = np.partition(S, -self.topk, axis=1)[:, -self.topk][:, None]
            S = np.where(S >= cut, S, 0.0)

        num = S @ (mx.R * mx.M)
        den = S @ mx.M
        base = np.broadcast_to(DescriptorPopularity().fit(mx, rng).mean, mx.R.shape)
        self.P = np.where(den > 1e-9, num / np.maximum(den, 1e-9), base)
        return self

    def score(self, u, i):
        return self.P[u, i]


# ═══════════════════════════════════════════════════════════════════════════
# MATRIX FACTORIZATION
# ═══════════════════════════════════════════════════════════════════════════

class MFRecommender(Recommender):
    """Pointwise matrix factorization, from scratch.

        r_hat(u,i) = mu + b_u + b_i + p_u . q_i
        L(Theta)   = SUM_{(u,i) in train} ( r_ui - r_hat(u,i) )^2 + lambda ||Theta||^2

    Minibatch SGD. Note precisely what is minimised: SQUARED ERROR ON RATINGS -- a
    REGRESSION objective -- while P@k / MRR / NDCG are RANKING objectives.

    A pointwise loss takes no difference, so NOTHING cancels: MF is the only model
    in the ladder that learns BOTH biases. That is why it wins both axes, and the
    MFNoBi ablation below proves it.

    use_bi=False gives the ABLATION: strip b_i, keep everything else identical.
    On the PRIMARY axis b_i provably cancels out of the ranking, so a naive reading
    says removing it should change nothing. It does not: without b_i, the latent
    vector q_i must encode the descriptor's general LEVEL as well as its INTERACTION
    pattern, and the geometry is worse for it. The bias does not affect the ranking
    directly; it frees q_i to model interaction alone.

    Early-stops on validation RMSE: its own objective. (BPR early-stops on validation
    AUC, which is its own objective. Symmetric -- neither is handed the other's metric.)
    """
    name = "mf"
    on_rating_scale = True
    target_axis = "both"

    def __init__(self, k=32, epochs=200, lr=0.01, reg=0.05, batch=2048,
                 patience=6, use_bi=True, name=None):
        self.k, self.epochs, self.lr, self.reg = k, epochs, lr, reg
        self.batch, self.patience, self.use_bi = batch, patience, use_bi
        if name:
            self.name = name

    def fit(self, mx, rng):
        tr = mx.train
        ua, ia, ra = tr[OCC].to_numpy(), tr[DSC].to_numpy(), tr[RAT].to_numpy()

        # Inner validation carved out of TRAIN. Test cells are never touched here.
        perm = rng.permutation(len(tr))
        nv = int(0.1 * len(tr))
        va, fi = perm[:nv], perm[nv:]
        uv, iv, rv = ua[va], ia[va], ra[va]
        ut, it, rt = ua[fi], ia[fi], ra[fi]

        self.mu = float(rt.mean())
        bu = np.zeros(mx.n_occ)
        bi = np.zeros(mx.n_desc)
        P = 0.1 * rng.standard_normal((mx.n_occ, self.k))
        Q = 0.1 * rng.standard_normal((mx.n_desc, self.k))

        best, best_state, bad = np.inf, None, 0
        self.history = []

        for ep in range(self.epochs):
            order = rng.permutation(len(ut))
            for s in range(0, len(order), self.batch):
                b = order[s:s + self.batch]
                ub, ib, rb = ut[b], it[b], rt[b]

                pred = self.mu + bu[ub] + np.einsum("bk,bk->b", P[ub], Q[ib])
                if self.use_bi:
                    pred = pred + bi[ib]
                e = pred - rb

                np.add.at(P, ub, -self.lr * (e[:, None] * Q[ib] + self.reg * P[ub]))
                np.add.at(Q, ib, -self.lr * (e[:, None] * P[ub] + self.reg * Q[ib]))
                np.add.at(bu, ub, -self.lr * (e + self.reg * bu[ub]))
                if self.use_bi:
                    # np.add.at, NOT fancy indexing: a descriptor appears many times
                    # in one batch and its gradients must ACCUMULATE, not overwrite.
                    np.add.at(bi, ib, -self.lr * (e + self.reg * bi[ib]))

            vp = self.mu + bu[uv] + np.einsum("bk,bk->b", P[uv], Q[iv])
            if self.use_bi:
                vp = vp + bi[iv]
            vr = float(np.sqrt(np.mean((np.clip(vp, RATING_MIN, RATING_MAX) - rv) ** 2)))
            self.history.append({"epoch": ep, "val_rmse": vr})

            if vr < best - 1e-4:
                best, bad, best_state = vr, 0, (bu.copy(), bi.copy(), P.copy(), Q.copy())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.bu, self.bi, self.P, self.Q = best_state
        if not self.use_bi:
            self.bi = np.zeros(mx.n_desc)
        self.val_rmse = best
        self.stopped_at = len(self.history)
        return self

    def score(self, u, i):
        p = self.mu + self.bu[u] + self.bi[i] + np.einsum("bk,bk->b", self.P[u], self.Q[i])
        return np.clip(p, RATING_MIN, RATING_MAX)

    # ---- persistence, so scripts/12_query.py can load a trained model -----------
    def save(self, path: Path):
        np.savez(path, mu=self.mu, bu=self.bu, bi=self.bi, P=self.P, Q=self.Q,
                 k=self.k, reg=self.reg)

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        m = cls(k=int(z["k"]), reg=float(z["reg"]))
        m.mu = float(z["mu"])
        m.bu, m.bi, m.P, m.Q = z["bu"], z["bi"], z["P"], z["Q"]
        return m


class MFNoBiRecommender(MFRecommender):
    """THE ABLATION. Identical to MF, with b_i removed.

    On the PRIMARY axis b_i cancels out of the ranking, so a naive argument says
    this should change nothing. Watch what actually happens: primary NDCG should
    collapse toward BPR-D's, isolating the cause of the MF-over-BPR gap as the
    MISSING BIAS TERM rather than the loss function.
    """
    name = "mf_no_bi"

    def __init__(self, k=32, epochs=200, lr=0.01, reg=0.05, batch=2048, patience=6):
        super().__init__(k=k, epochs=epochs, lr=lr, reg=reg, batch=batch,
                         patience=patience, use_bi=False, name="mf_no_bi")


# ═══════════════════════════════════════════════════════════════════════════
# BPR
# ═══════════════════════════════════════════════════════════════════════════

class _BPRBase(Recommender):
    """Bayesian Personalized Ranking (Rendle et al., 2009), from scratch.

        ln p(Theta | >) = SUM_{pairs} ln sigma( x_pos - x_neg ) - lambda ||Theta||^2

    With z = sigma(-(x_pos - x_neg)):   d/dtheta [ ln sigma(dx) ] = z * d(dx)/dtheta

    Classic BPR contrasts an observed item against an UNOBSERVED one. The OaSIS
    matrix has NO unobserved cells, so we generalise to the graded ordinal case:
    both members of a pair are observed, and the pair is (higher-rated, lower-rated).
    The objective then approximates the full ordinal ranking, which is exactly what
    NDCG measures. This extension is what makes a pairwise loss applicable to a
    complete matrix at all.

    Subclasses differ ONLY in the axis pairs are contrasted along, and hence -- forced
    by the algebra -- in which bias term is identifiable.
    """
    on_rating_scale = False   # a ranking score, not a rating; isotonic-calibrated for RMSE

    def __init__(self, k=32, epochs=200, lr=0.05, reg=0.01,
                 pairs_per_epoch=None, batch=4096, patience=6):
        self.k, self.epochs, self.lr, self.reg = k, epochs, lr, reg
        self.pairs_per_epoch, self.batch, self.patience = pairs_per_epoch, batch, patience

    @staticmethod
    def _blocks(df, group_col, n_groups):
        """Sort into contiguous per-group blocks; for every row count how many rows in
        the SAME group rank strictly below it. Ratings are sorted within a block, so
        that count is a searchsorted, giving O(1) negative sampling."""
        d = df.sort_values([group_col, RAT], kind="mergesort")
        g, r = d[group_col].to_numpy(), d[RAT].to_numpy()
        ptr = np.concatenate([[0], np.cumsum(np.bincount(g, minlength=n_groups))])
        n_lower = np.empty(len(d), dtype=np.int64)
        for k in range(n_groups):
            a, b = ptr[k], ptr[k + 1]
            if b > a:
                blk = r[a:b]
                n_lower[a:b] = np.searchsorted(blk, blk, side="left")
        return d, ptr, n_lower, np.flatnonzero(n_lower > 0)

    @staticmethod
    def _val_arrays(val):
        uv, iv, yv = val[OCC].to_numpy(), val[DSC].to_numpy(), val[RAT].to_numpy()
        keep = (yv >= CORE_THRESHOLD) | (yv <= NEGATIVE_THRESHOLD)
        return uv[keep], iv[keep], (yv[keep] >= CORE_THRESHOLD).astype(int)

    def _inner_split(self, mx, rng):
        tr = mx.train
        perm = rng.permutation(len(tr))
        nv = int(0.1 * len(tr))
        return tr.iloc[perm[:nv]], tr.iloc[perm[nv:]].reset_index(drop=True)


class BPRDescriptorAnchored(_BPRBase):
    """BPR-D. Fix a DESCRIPTOR, contrast two OCCUPATIONS.  Targets the PRIMARY axis.

        x(u,i) - x(v,i) = (b_u - b_v) + (p_u - p_v) . q_i

    b_i cancels, so only b_u is identifiable: this model carries b_u and NOT b_i.

    Gradients (z = sigma(-dx)):
        d(dx)/db_u = +1   d(dx)/db_v = -1
        d(dx)/dp_u = +q_i  d(dx)/dp_v = -q_i   d(dx)/dq_i = p_u - p_v
    """
    name = "bpr_d"
    target_axis = "primary"

    def fit(self, mx, rng):
        from sklearn.metrics import roc_auc_score
        val, fit_df = self._inner_split(mx, rng)
        d, ptr, n_lower, pool = self._blocks(fit_df, DSC, mx.n_desc)
        s_o, s_d = d[OCC].to_numpy(), d[DSC].to_numpy()
        if len(pool) == 0:
            raise ValueError("BPR-D: no comparable pairs (every descriptor is flat).")

        n_pairs = self.pairs_per_epoch or len(fit_df)
        uv, iv, yv = self._val_arrays(val)

        P = 0.1 * rng.standard_normal((mx.n_occ, self.k))
        Q = 0.1 * rng.standard_normal((mx.n_desc, self.k))
        bu = np.zeros(mx.n_occ)                      # the identifiable bias

        best, best_state, bad = -np.inf, None, 0
        self.history = []

        for ep in range(self.epochs):
            t = pool[rng.integers(0, len(pool), size=n_pairs)]
            i_, u_ = s_d[t], s_o[t]                              # fixed descriptor, higher occ
            off = (rng.random(n_pairs) * n_lower[t]).astype(np.int64)
            v_ = s_o[ptr[i_] + off]                              # lower-rated occupation

            for s in range(0, n_pairs, self.batch):
                ub, vb, ib = u_[s:s + self.batch], v_[s:s + self.batch], i_[s:s + self.batch]
                pu, pv, qi = P[ub], P[vb], Q[ib]
                dx = (bu[ub] - bu[vb]) + np.einsum("bk,bk->b", pu - pv, qi)
                z = sigmoid(-dx)

                np.add.at(P, ub, -self.lr * (-z[:, None] * qi + self.reg * pu))
                np.add.at(P, vb, -self.lr * (z[:, None] * qi + self.reg * pv))
                np.add.at(Q, ib, -self.lr * (-z[:, None] * (pu - pv) + self.reg * qi))
                np.add.at(bu, ub, -self.lr * (-z + self.reg * bu[ub]))
                np.add.at(bu, vb, -self.lr * (z + self.reg * bu[vb]))

            sv = bu[uv] + np.einsum("bk,bk->b", P[uv], Q[iv])
            vauc = float(roc_auc_score(yv, sv))
            self.history.append({"epoch": ep, "val_auc": vauc})
            if vauc > best + 1e-4:
                best, bad, best_state = vauc, 0, (P.copy(), Q.copy(), bu.copy())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.P, self.Q, self.bu = best_state
        self.val_auc = best
        self.stopped_at = len(self.history)
        return self

    def score(self, u, i):
        return self.bu[u] + np.einsum("bk,bk->b", self.P[u], self.Q[i])


class BPROccupationAnchored(_BPRBase):
    """BPR-O. Fix an OCCUPATION, contrast two DESCRIPTORS.  Targets the SECONDARY axis.

        x(u,i) - x(u,j) = (b_i - b_j) + p_u . (q_i - q_j)

    b_u cancels, so only b_i is identifiable: this model carries b_i and NOT b_u.

    Gradients (z = sigma(-dx)):
        d(dx)/db_i = +1   d(dx)/db_j = -1
        d(dx)/dp_u = q_i - q_j   d(dx)/dq_i = +p_u   d(dx)/dq_j = -p_u
    """
    name = "bpr_o"
    target_axis = "secondary"

    def fit(self, mx, rng):
        from sklearn.metrics import roc_auc_score
        val, fit_df = self._inner_split(mx, rng)
        d, ptr, n_lower, pool = self._blocks(fit_df, OCC, mx.n_occ)
        s_o, s_d = d[OCC].to_numpy(), d[DSC].to_numpy()
        if len(pool) == 0:
            raise ValueError("BPR-O: no comparable pairs (every occupation is flat).")

        n_pairs = self.pairs_per_epoch or len(fit_df)
        uv, iv, yv = self._val_arrays(val)

        P = 0.1 * rng.standard_normal((mx.n_occ, self.k))
        Q = 0.1 * rng.standard_normal((mx.n_desc, self.k))
        bi = np.zeros(mx.n_desc)                     # the identifiable bias

        best, best_state, bad = -np.inf, None, 0
        self.history = []

        for ep in range(self.epochs):
            t = pool[rng.integers(0, len(pool), size=n_pairs)]
            u_, i_ = s_o[t], s_d[t]
            off = (rng.random(n_pairs) * n_lower[t]).astype(np.int64)
            j_ = s_d[ptr[u_] + off]

            for s in range(0, n_pairs, self.batch):
                ub, ib, jb = u_[s:s + self.batch], i_[s:s + self.batch], j_[s:s + self.batch]
                pu, qi, qj = P[ub], Q[ib], Q[jb]
                dx = (bi[ib] - bi[jb]) + np.einsum("bk,bk->b", pu, qi - qj)
                z = sigmoid(-dx)

                np.add.at(P, ub, -self.lr * (-z[:, None] * (qi - qj) + self.reg * pu))
                np.add.at(Q, ib, -self.lr * (-z[:, None] * pu + self.reg * qi))
                np.add.at(Q, jb, -self.lr * (z[:, None] * pu + self.reg * qj))
                np.add.at(bi, ib, -self.lr * (-z + self.reg * bi[ib]))
                np.add.at(bi, jb, -self.lr * (z + self.reg * bi[jb]))

            sv = bi[iv] + np.einsum("bk,bk->b", P[uv], Q[iv])
            vauc = float(roc_auc_score(yv, sv))
            self.history.append({"epoch": ep, "val_auc": vauc})
            if vauc > best + 1e-4:
                best, bad, best_state = vauc, 0, (P.copy(), Q.copy(), bi.copy())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.P, self.Q, self.bi = best_state
        self.val_auc = best
        self.stopped_at = len(self.history)
        return self

    def score(self, u, i):
        return self.bi[i] + np.einsum("bk,bk->b", self.P[u], self.Q[i])


# ═══════════════════════════════════════════════════════════════════════════
# NODE2VEC  (the proposal's third named approach)
# ═══════════════════════════════════════════════════════════════════════════

class Node2VecRecommender(Recommender):
    """node2vec (Grover & Leskovec, KDD 2016), from scratch.

    WHY IT NEEDS A THRESHOLDED GRAPH
    ---------------------------------
    node2vec learns embeddings from random walks. On the FULL OaSIS graph every
    occupation is connected to every descriptor (100% coverage), so a random walk
    visits nodes essentially uniformly and the embeddings collapse to noise. It is
    the same degeneracy that made link prediction meaningless.

    So we threshold: keep only CORE edges (rating >= 4). That yields a 14.1%-dense
    bipartite graph in which walk structure actually carries signal -- an occupation
    is reachable from a descriptor only if that descriptor genuinely matters to it.

    This is a modelling decision that must be stated, not hidden: node2vec sees a
    BINARISED view of the data, while MF and BPR see the full ordinal ratings. Any
    performance gap is therefore partly an information gap, and it would be dishonest
    to attribute it entirely to the algorithm.

    ARCHITECTURE
    ------------
    Bipartite node set:  0 .. n_occ-1              = occupations
                         n_occ .. n_occ+n_desc-1   = descriptors

    Second-order biased walks with return parameter p and in-out parameter q:

        w(prev -> cur -> next) =  1/p  if next == prev          (backtrack)
                                  1    if next adjacent to prev (stay local, BFS-like)
                                  1/q  otherwise                (explore, DFS-like)

    Then skip-gram with negative sampling (SGNS) over the walk corpus, trained by
    SGD -- the same gradient machinery as BPR:

        L = -ln sigma(w_c . c_o) - SUM_{neg} ln sigma(-w_c . c_neg)

    Score(u, i) = dot(emb[u], emb[n_occ + i]).  A ranking score, not a rating, so it
    is isotonic-calibrated before any RMSE is computed.
    """
    name = "node2vec"
    on_rating_scale = False
    target_axis = "both"

    def __init__(self, k=32, walks_per_node=10, walk_len=40, window=5,
                 p=1.0, q=0.5, neg=5, epochs=3, lr=0.025):
        self.k, self.walks_per_node, self.walk_len = k, walks_per_node, walk_len
        self.window, self.p, self.q, self.neg = window, p, q, neg
        self.epochs, self.lr = epochs, lr

    def _build_graph(self, mx):
        core = mx.train[mx.train[RAT] >= CORE_THRESHOLD]
        n_nodes = mx.n_occ + mx.n_desc
        adj = [[] for _ in range(n_nodes)]
        for u, i in zip(core[OCC].to_numpy(), core[DSC].to_numpy()):
            d = mx.n_occ + int(i)
            adj[int(u)].append(d)
            adj[d].append(int(u))
        self.adj = [np.array(a, dtype=np.int32) for a in adj]
        self.adj_set = [set(a.tolist()) for a in self.adj]
        self.n_nodes = n_nodes
        return int(core.shape[0])

    def _walk(self, start, rng):
        walk = [start]
        if len(self.adj[start]) == 0:
            return walk
        walk.append(int(rng.choice(self.adj[start])))

        while len(walk) < self.walk_len:
            cur = walk[-1]
            nbrs = self.adj[cur]
            if len(nbrs) == 0:
                break
            prev = walk[-2]
            prev_set = self.adj_set[prev]

            # second-order node2vec transition weights
            w = np.where(nbrs == prev, 1.0 / self.p,
                         np.where([int(x) in prev_set for x in nbrs], 1.0, 1.0 / self.q))
            w = w / w.sum()
            walk.append(int(rng.choice(nbrs, p=w)))
        return walk

    def fit(self, mx, rng):
        n_core = self._build_graph(mx)

        # 1. generate the walk corpus
        walks = []
        nodes = np.arange(self.n_nodes)
        for _ in range(self.walks_per_node):
            for s in rng.permutation(nodes):
                if len(self.adj[s]):
                    walks.append(self._walk(int(s), rng))

        # 2. skip-gram pairs (centre, context) within the window
        centres, contexts = [], []
        for w in walks:
            L = len(w)
            for pos in range(L):
                lo, hi = max(0, pos - self.window), min(L, pos + self.window + 1)
                for j in range(lo, hi):
                    if j != pos:
                        centres.append(w[pos])
                        contexts.append(w[j])
        centres = np.asarray(centres, dtype=np.int32)
        contexts = np.asarray(contexts, dtype=np.int32)

        # 3. negative-sampling distribution: unigram ^ 0.75 (Mikolov et al.)
        counts = np.bincount(np.concatenate([centres, contexts]), minlength=self.n_nodes)
        probs = np.power(counts, 0.75)
        probs = probs / probs.sum()

        # 4. SGNS by SGD
        W = (rng.random((self.n_nodes, self.k)) - 0.5) / self.k   # input  (the output embedding)
        C = np.zeros((self.n_nodes, self.k))                      # output (context)

        n_pairs = len(centres)
        batch = 8192
        for ep in range(self.epochs):
            order = rng.permutation(n_pairs)
            for s in range(0, n_pairs, batch):
                b = order[s:s + batch]
                cb, ob = centres[b], contexts[b]
                nb = rng.choice(self.n_nodes, size=(len(b), self.neg), p=probs)

                wc = W[cb]                                        # (B, k)

                # positive:  maximise ln sigma(wc . co)
                co = C[ob]
                gp = (sigmoid(np.einsum("bk,bk->b", wc, co)) - 1.0)   # (B,)
                gW = gp[:, None] * co
                np.add.at(C, ob, -self.lr * gp[:, None] * wc)

                # negatives: maximise ln sigma(-wc . cn)
                cn = C[nb]                                        # (B, neg, k)
                gn = sigmoid(np.einsum("bk,bnk->bn", wc, cn))     # (B, neg)
                gW += np.einsum("bn,bnk->bk", gn, cn)
                np.add.at(C, nb, -self.lr * gn[..., None] * wc[:, None, :])

                np.add.at(W, cb, -self.lr * gW)

        self.W = W
        self.n_occ = mx.n_occ
        self.n_core_edges = n_core
        self.n_walks = len(walks)
        self.n_pairs = n_pairs
        return self

    def score(self, u, i):
        return np.einsum("bk,bk->b", self.W[u], self.W[self.n_occ + i])


# ═══════════════════════════════════════════════════════════════════════════
# COLD-START FOLD-IN  (the theory bonus, finally load-bearing)
# ═══════════════════════════════════════════════════════════════════════════

def fold_in(mf: MFRecommender, desc_ids: np.ndarray, ratings: np.ndarray,
            reg: float | None = None):
    """Solve for a NEW occupation's latent vector from a handful of revealed ratings.

    A person who walks in off the street has no p_u: the model has never seen them.
    We must infer one. Hold Q and b_i FIXED (they were learned from 810 occupations)
    and solve for p_u and b_u by least squares over the revealed subset S:

        min_{p_u, b_u}  SUM_{i in S} ( r_i - mu - b_i - b_u - p_u . q_i )^2
                        + lambda ( ||p_u||^2 + b_u^2 )

    Augment the design matrix with a column of ones so b_u is estimated jointly:

        X = [ Q_S | 1 ]         theta = [ p_u ; b_u ]
        y = r_S - mu - b_S

        theta = ( X^T X + lambda I )^{-1} X^T y            <-- ridge, CLOSED FORM

    That is the whole derivation. No iteration, no gradient descent: one linear solve.
    It is what makes the recommender usable at inference time by someone who was never
    in the training set, and it is the mechanism behind scripts/12_query.py.

    Returns (p_u, b_u).
    """
    reg = mf.reg if reg is None else reg
    Q_S = mf.Q[desc_ids]                              # (|S|, k)
    b_S = mf.bi[desc_ids]                             # (|S|,)
    y = np.asarray(ratings, dtype=float) - mf.mu - b_S

    X = np.hstack([Q_S, np.ones((len(Q_S), 1))])      # (|S|, k+1)
    A = X.T @ X + reg * np.eye(X.shape[1])
    theta = np.linalg.solve(A, X.T @ y)
    return theta[:-1], float(theta[-1])


def score_all_occupations(mf: MFRecommender, desc_ids: np.ndarray, n_occ: int,
                          centre: bool = True):
    """Given a skill profile, score all n_occ occupations.

        raw(u)     = mean_{d in profile} x(u, d)
        centred(u) = raw(u) - mean_{d in ALL} x(u, d)

    The RAW score measures LEVEL: surgeons rate everything highly, so they win every
    query regardless of what was asked. It ranked "Early childhood educators" ->
    ophthalmologists, general surgeons. Textbook popularity bias.

    CENTRING asks the right question: relative to how this occupation normally rates
    things, does it rate YOUR skills unusually highly? It measures FIT. Empirically it
    moves the true occupation from rank #92 to rank #10.

    Returns (raw, centred), each of shape (n_occ,).
    """
    all_u = np.arange(n_occ)

    raw = np.zeros(n_occ)
    for d in desc_ids:
        raw += mf.score(all_u, np.full(n_occ, int(d)))
    raw /= max(len(desc_ids), 1)

    if not centre:
        return raw, raw

    level = np.zeros(n_occ)
    n_desc = mf.Q.shape[0]
    for d in range(n_desc):
        level += mf.score(all_u, np.full(n_occ, d))
    level /= n_desc

    return raw, raw - level


# ═══════════════════════════════════════════════════════════════════════════

def make_ladder(k=32, epochs=200):
    return [
        RandomRecommender(),
        DescriptorPopularity(),
        OccupationPopularity(),
        JaccardRecommender(topk=30),
        Node2VecRecommender(k=k),
        MFNoBiRecommender(k=k, epochs=epochs),
        MFRecommender(k=k, epochs=epochs),
        BPRDescriptorAnchored(k=k, epochs=epochs),
        BPROccupationAnchored(k=k, epochs=epochs),
    ]
