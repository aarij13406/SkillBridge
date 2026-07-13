"""
SkillBridge — Evaluation Metrics
=================================
ONE implementation of each metric. Everyone imports from here.

If Aarij computes Precision@5 one way and Anand computes it another,
the numbers in the report are not comparable and the whole evaluation
is worthless. So: one file, one definition, no exceptions.

Usage:
    from skillbridge.metrics import ranking_report, classification_report_full
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    brier_score_loss,
)


# ═══════════════════════════════════════════════════════════════
# RANKING METRICS  (Occupation Recommender, Skill-Gap Recommender)
# ═══════════════════════════════════════════════════════════════

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Of the top-k items we recommended, what fraction were truly relevant?

        P@k = |{relevant} ∩ {top-k retrieved}| / k

    y_true  : binary relevance, shape (n_items,)
    y_score : predicted score,  shape (n_items,)
    """
    if len(y_true) == 0:
        return np.nan
    k = min(k, len(y_true))
    top_k = np.argsort(-y_score)[:k]
    return float(np.sum(y_true[top_k]) / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Of ALL truly relevant items, what fraction did we surface in the top-k?

        R@k = |{relevant} ∩ {top-k retrieved}| / |{relevant}|
    """
    n_rel = float(np.sum(y_true))
    if n_rel == 0:
        return np.nan          # undefined; caller should skip this query
    k = min(k, len(y_true))
    top_k = np.argsort(-y_score)[:k]
    return float(np.sum(y_true[top_k]) / n_rel)


def reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    1 / (rank of the first relevant item). Averaged over queries -> MRR.
    Returns 0.0 if nothing relevant was retrieved at all.
    """
    order = np.argsort(-y_score)
    hits = np.where(y_true[order] == 1)[0]
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def ndcg_at_k(y_true_graded: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain with GRADED relevance.

        DCG@k  = Σ_{i=1..k}  (2^rel_i - 1) / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    This is where the OaSIS 1-5 importance rating earns its keep: a
    descriptor rated 5 SHOULD outrank one rated 3, and NDCG rewards that.
    Binary metrics throw that information away.
    """
    if len(y_true_graded) == 0 or np.sum(y_true_graded) == 0:
        return np.nan
    k = min(k, len(y_true_graded))

    def _dcg(rels: np.ndarray) -> float:
        discounts = np.log2(np.arange(2, len(rels) + 2))
        return float(np.sum((np.power(2, rels) - 1) / discounts))

    order = np.argsort(-y_score)[:k]
    dcg = _dcg(y_true_graded[order])
    idcg = _dcg(np.sort(y_true_graded)[::-1][:k])
    return float(dcg / idcg) if idcg > 0 else np.nan


def ranking_report(
    y_true_per_query: Sequence[np.ndarray],
    y_score_per_query: Sequence[np.ndarray],
    y_graded_per_query: Sequence[np.ndarray] | None = None,
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """
    Full ranking evaluation, averaged over queries.

    Each element of the sequences is ONE query (e.g. one occupation, with
    scores over all candidate descriptors).

    Returns every metric the proposal promised:
        Precision@k, Recall@k, MRR, ROC-AUC, and (bonus) NDCG@k.
    """
    out: Dict[str, float] = {}

    for k in k_values:
        p = [precision_at_k(t, s, k) for t, s in zip(y_true_per_query, y_score_per_query)]
        r = [recall_at_k(t, s, k)    for t, s in zip(y_true_per_query, y_score_per_query)]
        out[f"precision@{k}"] = float(np.nanmean(p))
        out[f"recall@{k}"]    = float(np.nanmean(r))

    rr = [reciprocal_rank(t, s) for t, s in zip(y_true_per_query, y_score_per_query)]
    out["mrr"] = float(np.nanmean(rr))

    # ROC-AUC is computed GLOBALLY (pool all queries), which is what the
    # proposal's "ROC-AUC on held-out edges" means.
    all_true  = np.concatenate(list(y_true_per_query))
    all_score = np.concatenate(list(y_score_per_query))
    if len(np.unique(all_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(all_true, all_score))
        out["pr_auc"]  = float(average_precision_score(all_true, all_score))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"]  = np.nan

    if y_graded_per_query is not None:
        for k in k_values:
            nd = [ndcg_at_k(g, s, k)
                  for g, s in zip(y_graded_per_query, y_score_per_query)]
            out[f"ndcg@{k}"] = float(np.nanmean(nd))

    return out


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS  (Labour Shortage Classifier)
# ═══════════════════════════════════════════════════════════════

def classification_report_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
) -> Dict:
    """
    Macro-F1 is the PRIMARY metric — it weights the rare Surplus class
    equally with the dominant Balance class. Accuracy would be a lie here
    (predict "Balance" always -> 75% accuracy, 0 useful information).
    """
    out: Dict = {
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy":    float(np.mean(y_true == y_pred)),
    }

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out["per_class"] = {
        str(lab): {
            "precision": float(pi), "recall": float(ri),
            "f1": float(fi), "support": int(si),
        }
        for lab, pi, ri, fi, si in zip(labels or np.unique(y_true), p, r, f, sup)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    if y_proba is not None and labels is not None and len(labels) > 2:
        try:
            out["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr",
                              average="macro", labels=labels)
            )
        except ValueError:
            out["roc_auc_ovr"] = np.nan

    return out


# ═══════════════════════════════════════════════════════════════
# REGRESSION METRICS  (Salary & Regional Demand)
# ═══════════════════════════════════════════════════════════════

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE in CAD is the headline. R² tells us if we beat the mean predictor."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mape": np.nan}

    nz = yt != 0
    mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.any() else np.nan

    return {
        "mae":  float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2":   float(r2_score(yt, yp)),
        "mape": mape,
        "n":    int(len(yt)),
    }


# ═══════════════════════════════════════════════════════════════
# MULTI-LABEL METRICS  (NLP Skill Extraction)
# ═══════════════════════════════════════════════════════════════

def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred : binary indicator matrices, shape (n_docs, n_skills).
    Micro-F1 is the headline for extraction (it weights by skill frequency).
    """
    return {
        "micro_precision": float(precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0)[0]),
        "micro_recall": float(precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0)[1]),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }


# ═══════════════════════════════════════════════════════════════
# CALIBRATION METRICS  (Dharnesh's trust layer)
# ═══════════════════════════════════════════════════════════════

def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """
    ECE: |confidence - accuracy|, averaged over confidence bins,
    weighted by bin size.

        ECE = Σ_b  (|B_b| / n) · |acc(B_b) - conf(B_b)|

    A model with ECE ~0 means: when it says "80% sure", it is right 80%
    of the time. Target for this project: ECE < 0.05.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        ece += (in_bin.sum() / n) * abs(
            accuracies[in_bin].mean() - confidences[in_bin].mean()
        )
    return float(ece)


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """Mean squared error between predicted probability vector and one-hot truth."""
    onehot = np.zeros((len(y_true), n_classes))
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10):
    """Returns (mean_confidence_per_bin, accuracy_per_bin, count_per_bin) for plotting."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.sum() == 0:
            conf.append(np.nan); acc.append(np.nan); cnt.append(0)
        else:
            conf.append(confidences[m].mean())
            acc.append(accuracies[m].mean())
            cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


# ═══════════════════════════════════════════════════════════════
# RESULT PERSISTENCE — so nobody loses a number
# ═══════════════════════════════════════════════════════════════

def save_result(
    results: Dict,
    component: str,
    model_name: str,
    results_dir: Path,
    extra: Dict | None = None,
) -> Path:
    """
    Write one JSON per (component, model). The report tables get built
    by globbing this directory — no copy-pasting numbers from terminals.
    """
    payload = {
        "component": component,
        "model": model_name,
        "metrics": results,
    }
    if extra:
        payload["extra"] = extra

    results_dir.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(" ", "_").replace("/", "-")
    path = results_dir / f"{component}__{safe}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_all_results(results_dir: Path) -> "list[dict]":
    """Read every saved result back — used to build the report tables."""
    return [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
