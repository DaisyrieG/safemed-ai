"""Unit tests for evaluation metrics and Wilcoxon signed-rank inferential statistical tests."""

import pytest
import numpy as np
from src.stage5_evaluation.metrics import (
    hallucination_exposure_rate,
    ground_truth_retention_rate,
    filter_metrics,
    fact_level_metrics,
)
from src.stage5_evaluation.statistical_tests import (
    wilcoxon_paired,
    wilcoxon_one_sample,
    hodges_lehmann,
    rank_biserial,
    bootstrap_ci,
    holm_adjust,
    StatisticalAnalyzer,
)


def test_hallucination_exposure_rate():
    docs = [
        {"title": "Doc 1", "true_label": "harmful"},
        {"title": "Doc 2", "true_label": "ground_truth"},
        {"title": "Doc 3", "true_label": "harmful"},
        {"title": "Doc 4", "true_label": "mediocre"},
        {"title": "Doc 5", "true_label": "ground_truth"},
    ]
    assert hallucination_exposure_rate(docs) == pytest.approx(0.4)

    clean_docs = [{"title": f"Doc {i}", "true_label": "ground_truth"} for i in range(5)]
    assert hallucination_exposure_rate(clean_docs) == 0.0


def test_ground_truth_retention_rate():
    docs = [
        {"title": "Doc 1", "true_label": "ground_truth"},
        {"title": "Doc 2", "true_label": "ground_truth"},
        {"title": "Doc 3", "true_label": "ground_truth"},
        {"title": "Doc 4", "true_label": "mediocre"},
        {"title": "Doc 5", "true_label": "harmful"},
    ]
    assert ground_truth_retention_rate(docs) == pytest.approx(0.6)


def test_filter_metrics():
    y_true = ["harmful", "harmful", "ground_truth", "mediocre", "harmful"]
    y_pred = ["harmful", "mediocre", "ground_truth", "mediocre", "harmful"]

    result = filter_metrics(y_true, y_pred)

    assert result["TPR_Recall"] == pytest.approx(2 / 3)
    assert result["FPR"] == 0.0
    assert result["Precision"] == 1.0
    assert result["F1_score"] == pytest.approx(0.8)


def test_fact_level_metrics():
    claims = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
    flags = [True, True, True, False]
    res = fact_level_metrics(claims, flags)
    assert res["precision"] == 0.75
    assert res["hallucination_rate"] == 0.25


def test_wilcoxon_paired():
    control = [0.4, 0.6, 0.2, 0.4, 0.8, 0.6]
    treatment = [0.0, 0.2, 0.0, 0.0, 0.2, 0.0]

    res = wilcoxon_paired(control, treatment)
    assert res["test_type"] == "Wilcoxon Signed-Rank Test (Paired)"
    assert "statistic" in res
    assert "p_value" in res
    assert res["mean_diff"] < 0
    assert res["median_diff"] < 0
    assert res["hodges_lehmann"] < 0
    assert res["rank_biserial"] <= -0.8
    assert res["significant"] is True
    assert len(res["ci_95"]) == 2


def test_wilcoxon_one_sample():
    gtrr_scores = [0.8, 0.6, 1.0, 0.8, 0.6, 1.0]

    res = wilcoxon_one_sample(gtrr_scores, reference=0.0)
    assert res["test_type"] == "Wilcoxon Signed-Rank Test (One-Sample)"
    assert res["p_value"] < 0.05
    assert res["hodges_lehmann"] > 0
    assert res["rank_biserial"] == 1.0
    assert res["significant"] is True


def test_hodges_lehmann():
    diffs = np.array([-0.4, -0.2, -0.2, 0.0])
    hl = hodges_lehmann(diffs)
    assert hl < 0


def test_rank_biserial():
    diffs_pos = np.array([0.1, 0.2, 0.3])
    assert rank_biserial(diffs_pos) == 1.0

    diffs_neg = np.array([-0.1, -0.2, -0.3])
    assert rank_biserial(diffs_neg) == -1.0


def test_bootstrap_ci():
    diffs = np.array([-0.2, -0.4, -0.2, -0.6, -0.4])
    ci_lower, ci_upper = bootstrap_ci(diffs, n_resamples=1000)
    assert ci_lower <= ci_upper
    assert ci_upper <= 0.0


def test_holm_adjust():
    p_values = [0.01, 0.04, 0.03, 0.20]
    adj = holm_adjust(p_values)
    assert len(adj) == len(p_values)
    assert adj[0] <= adj[2] <= adj[1] <= adj[3]


def test_statistical_analyzer_class():
    analyzer = StatisticalAnalyzer(alpha=0.05, n_bootstrap=1000)
    c = [0.4, 0.6, 0.2, 0.4, 0.8, 0.6]
    t = [0.0, 0.2, 0.0, 0.0, 0.2, 0.0]
    res = analyzer.wilcoxon_paired(c, t)
    assert res["significant"] is True
    assert res["hodges_lehmann"] < 0
