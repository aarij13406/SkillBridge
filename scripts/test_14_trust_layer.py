"""
Unit tests for scripts/14_trust_layer.py  (Dharnesh Somasundaram)
==================================================================

These are the checks that were previously done by hand, now automated so they
can be rerun with one command instead of manually re-verifying after every
change. They test the MATH (calibration metrics, bootstrap CIs, disparity
logic) against hand-computed ground truth and known edge cases -- they do
NOT re-run the full audit against real data (that's a slow integration test,
covered separately by just running the main script).

HOW TO RUN
----------
    pip install pytest
    pytest scripts/test_14_trust_layer.py -v

All tests should pass in a few seconds. A failure here means either the
statistics are wrong or someone changed behavior without updating the tests
-- either way, don't trust the audit's numbers until this is green again.
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Import 14_trust_layer.py as a module. It can't be `import 14_trust_layer`
#    (module names can't start with a digit), so load it by file path. This
#    executes the module's top level (creates results/ and figures/ dirs if
#    missing -- harmless and idempotent) but NOT main(), which is guarded by
#    `if __name__ == "__main__"`. ──
_THIS_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("trust_layer", _THIS_DIR / "14_trust_layer.py")
tl = importlib.util.module_from_spec(_SPEC)
sys.modules["trust_layer"] = tl
_SPEC.loader.exec_module(tl)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_ece_matches_hand_computation():
    """
    5 predictions, 2 classes, chosen so the bins and the hand math are simple:
    bin [0.5,0.6]: 3 preds, all correct, mean conf 0.55 -> |1.0-0.55|=0.45
    bin [0.9,1.0]: 2 preds, 1 correct,  mean conf 0.965 -> |0.5-0.965|=0.465
    """
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([
        [0.55, 0.45], [0.58, 0.42], [0.52, 0.48],  # all correct, ~0.55 conf
        [0.95, 0.05],                               # WRONG (pred 0, true 1)
        [0.02, 0.98],                               # correct
    ])
    ece = tl.expected_calibration_error(y_true, proba, n_bins=10)
    manual = (3 / 5) * abs(1.0 - 0.55) + (2 / 5) * abs(0.5 - 0.965)
    assert ece == pytest.approx(manual, abs=1e-9)


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    """A model that is always exactly as confident as it is correct scores ECE 0."""
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])   # 100% confident, always right
    assert tl.expected_calibration_error(y_true, proba) == pytest.approx(0.0, abs=1e-9)


def test_multiclass_brier_matches_hand_computation():
    y_true = np.array([0, 1, 2])
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
    onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    manual = np.mean(np.sum((proba - onehot) ** 2, axis=1))
    assert tl.multiclass_brier(y_true, proba, n_classes=3) == pytest.approx(manual, abs=1e-9)


def test_reliability_curve_shapes_and_empty_bins():
    """Bins with no predictions in them should report NaN, not crash or fabricate a value."""
    y_true = np.array([0, 0, 1])
    proba = np.array([[0.95, 0.05], [0.96, 0.04], [0.05, 0.95]])   # nothing lands in low-confidence bins
    conf, acc, count = tl.reliability_curve(y_true, proba, n_bins=10)
    assert len(conf) == len(acc) == len(count) == 10
    assert np.isnan(conf[0])          # the [0.0, 0.1] bin is empty
    assert count[0] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ══════════════════════════════════════════════════════════════════════════════

def test_bootstrap_ci_covers_the_true_mean():
    """95% CI on a large normal sample should contain the true generating mean."""
    rng = np.random.default_rng(0)
    sample = rng.normal(100.0, 15, 500)
    point, lo, hi = tl.bootstrap_ci(sample, np.mean, n_boot=3000, seed=1)
    assert lo <= 100.0 <= hi
    assert lo < point < hi


def test_bootstrap_ci_width_matches_theoretical_sem():
    """The bootstrap interval width should track the textbook 1.96*SEM formula."""
    rng = np.random.default_rng(0)
    sigma, n = 15, 500
    sample = rng.normal(100.0, sigma, n)
    _, lo, hi = tl.bootstrap_ci(sample, np.mean, n_boot=3000, seed=1)
    theoretical_half_width = 1.96 * sigma / np.sqrt(n)
    observed_half_width = (hi - lo) / 2
    assert observed_half_width == pytest.approx(theoretical_half_width, rel=0.15)


def test_bootstrap_ci_empty_input_returns_nan():
    point, lo, hi = tl.bootstrap_ci(np.array([]))
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


# ══════════════════════════════════════════════════════════════════════════════
# Disparity logic  --  best/worst direction, significance, the abs_floor guard
# ══════════════════════════════════════════════════════════════════════════════

def test_disparity_higher_is_better_picks_max_as_best():
    """For accuracy (higher = better), the highest-value group must be 'best'."""
    audit = pd.DataFrame({
        "group": ["0", "1", "2"],
        "accuracy": [0.80, 0.90, 0.85],
        "acc_lo": [0.70, 0.85, 0.80],
        "acc_hi": [0.90, 0.95, 0.90],
    })
    d = tl.disparity_summary(audit, "accuracy", "acc_lo", "acc_hi", higher_is_better=True)
    assert d["best_group"] == "1" and d["best_value"] == 0.90
    assert d["worst_group"] == "0" and d["worst_value"] == 0.80
    assert d["disparity_relative"] == pytest.approx((0.90 - 0.80) / 0.90)


def test_disparity_lower_is_better_picks_min_as_best():
    """For MAE (lower = better), the lowest-value group must be 'best'."""
    audit = pd.DataFrame({
        "group": ["low_err", "high_err"],
        "mae": [10.0, 50.0],
        "mae_lo": [9.0, 45.0],
        "mae_hi": [11.0, 55.0],
    })
    d = tl.disparity_summary(audit, "mae", "mae_lo", "mae_hi", higher_is_better=False)
    assert d["best_group"] == "low_err"
    assert d["worst_group"] == "high_err"
    assert d["gap_is_significant"] is True     # 11 < 45 -> intervals disjoint


def test_disparity_overlapping_intervals_are_not_significant():
    audit = pd.DataFrame({
        "group": ["a", "b"],
        "mae": [10.0, 12.0],
        "mae_lo": [8.0, 9.0],
        "mae_hi": [13.0, 15.0],       # a's hi (13) > b's lo (9) -> overlap
    })
    d = tl.disparity_summary(audit, "mae", "mae_lo", "mae_hi", higher_is_better=False)
    assert d["intervals_overlap"] is True
    assert d["gap_is_significant"] is False


def test_disparity_disjoint_intervals_are_significant():
    audit = pd.DataFrame({
        "group": ["a", "b"],
        "mae": [10.0, 50.0],
        "mae_lo": [8.0, 45.0],
        "mae_hi": [12.0, 55.0],       # a's hi (12) < b's lo (45) -> disjoint
    })
    d = tl.disparity_summary(audit, "mae", "mae_lo", "mae_hi", higher_is_better=False)
    assert d["intervals_overlap"] is False
    assert d["gap_is_significant"] is True


def test_disparity_abs_floor_flags_near_zero_metric_as_unreliable():
    """
    Regression test for the posting-volume bug: when the metric itself is tiny
    (MAE ~1), a relative percentage is misleading and must be flagged as such.
    """
    audit = pd.DataFrame({
        "group": ["x", "y"],
        "mae": [1.0, 5.0],
        "mae_lo": [0.8, 4.0],
        "mae_hi": [1.2, 6.0],
    })
    unreliable = tl.disparity_summary(audit, "mae", "mae_lo", "mae_hi",
                                      higher_is_better=False, abs_floor=5.0)
    assert unreliable["relative_reliable"] is False

    reliable = tl.disparity_summary(audit, "mae", "mae_lo", "mae_hi",
                                    higher_is_better=False, abs_floor=0.5)
    assert reliable["relative_reliable"] is True


def test_disparity_fewer_than_two_groups_returns_nan_gracefully():
    audit = pd.DataFrame({"group": ["only_one"], "mae": [10.0]})
    d = tl.disparity_summary(audit, "mae")
    assert np.isnan(d["disparity_relative"])


# ══════════════════════════════════════════════════════════════════════════════
# TEER extraction  --  including the "does zfill self-heal a stripped code" case
# ══════════════════════════════════════════════════════════════════════════════

def test_teer_from_noc_on_proper_5digit_codes():
    assert tl.TEER_FROM_NOC("12345") == "2"
    assert tl.TEER_FROM_NOC("00000") == "0"
    assert tl.TEER_FROM_NOC("54321") == "4"


def test_teer_from_noc_self_heals_a_leading_zero_stripped_code():
    """
    Regression test for the noc21_code dtype bug: pandas reading a code column
    without dtype=str can turn "01067" into the plain string "1067". zfill(5)
    inside TEER_FROM_NOC must still recover the correct TEER digit either way.
    """
    assert tl.TEER_FROM_NOC("01067") == tl.TEER_FROM_NOC("1067")
    assert tl.TEER_FROM_NOC("01067") == "1"


def test_teer_name_labels_all_six_tiers_and_falls_back_gracefully():
    for digit in "012345":
        label = tl.teer_name(digit)
        assert label.startswith(f"TEER {digit}")
    assert tl.teer_name("9") == "TEER 9"      # unknown digit -> generic fallback, not a crash


# ══════════════════════════════════════════════════════════════════════════════
# Popularity tiering
# ══════════════════════════════════════════════════════════════════════════════

def test_popularity_tier_splits_into_three_roughly_equal_groups():
    counts = pd.Series(range(1, 31))   # 30 distinct values, easy tertiles
    tiers = tl.popularity_tier(counts)
    sizes = tiers.value_counts()
    assert set(sizes.index) == {"low", "medium", "high"}
    assert sizes.min() >= 9 and sizes.max() <= 11   # roughly even thirds of 30


def test_popularity_tier_handles_too_few_distinct_values():
    """qcut fails when there aren't enough distinct values for 3 bins; must not crash."""
    counts = pd.Series([5, 5, 5, 5])
    tiers = tl.popularity_tier(counts)     # should fall back to pd.cut, not raise
    assert len(tiers) == 4


# ══════════════════════════════════════════════════════════════════════════════
# Grouped audits  --  min_n filtering and R^2 edge cases
# ══════════════════════════════════════════════════════════════════════════════

def test_grouped_regression_audit_drops_groups_below_min_n():
    df = pd.DataFrame({
        "grp": ["a"] * 25 + ["b"] * 3,    # b has only 3 rows, below default min_n=20
        "true": np.random.default_rng(0).normal(100, 10, 28),
        "pred": np.random.default_rng(1).normal(100, 10, 28),
    })
    out = tl.grouped_regression_audit(df, "grp", "true", "pred")
    assert set(out["group"]) == {"a"}      # group "b" correctly excluded


def test_grouped_regression_audit_mae_is_correct():
    df = pd.DataFrame({
        "grp": ["a"] * 25,
        "true": [100.0] * 25,
        "pred": [110.0] * 25,      # constant 10-unit error
    })
    out = tl.grouped_regression_audit(df, "grp", "true", "pred", min_n=1)
    assert out.loc[0, "mae"] == pytest.approx(10.0)
    assert out.loc[0, "mae_lo"] == pytest.approx(10.0)  # zero variance -> a degenerate but valid CI
    assert out.loc[0, "mae_hi"] == pytest.approx(10.0)


def test_grouped_classification_audit_accuracy_is_correct():
    df = pd.DataFrame({
        "grp": ["a"] * 10,
        "y_true": ["X"] * 8 + ["Y"] * 2,
        "y_pred": ["X"] * 8 + ["Y"] * 2,   # perfect predictions
    })
    out = tl.grouped_classification_audit(df, "grp", "y_true", "y_pred", min_n=1)
    assert out.loc[0, "accuracy"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Result persistence  --  JSON actually lands on disk with the right shape
# ══════════════════════════════════════════════════════════════════════════════

def test_save_result_writes_valid_json(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(tl, "RESULTS_DIR", tmp_path)
    path = tl.save_result({"ece": 0.02}, "unit_test_component", "unit_test_model",
                          extra={"note": "test"})
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["component"] == "unit_test_component"
    assert payload["metrics"]["ece"] == 0.02
    assert payload["extra"]["note"] == "test"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
