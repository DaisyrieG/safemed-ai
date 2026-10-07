"""Wilcoxon signed-rank tests (Pratt), effect sizes, bootstrap CIs and Holm adjustment."""

from typing import List, Dict, Any, Tuple, Sequence
import numpy as np
from scipy import stats


def hodges_lehmann(d: np.ndarray) -> float:
    """Hodges-Lehmann estimate of the median paired difference."""
    diff = np.asarray(d, dtype=float)
    n = len(diff)
    if n == 0:
        return 0.0
    i_idx, j_idx = np.triu_indices(n)
    walsh_averages = (diff[i_idx] + diff[j_idx]) / 2.0
    return float(np.median(walsh_averages))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size."""
    diff = np.asarray(d, dtype=float)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    r_pos = float(np.sum(ranks[nonzero > 0]))
    r_neg = float(np.sum(ranks[nonzero < 0]))
    total_ranks = r_pos + r_neg
    if total_ranks == 0:
        return 0.0
    return float((r_pos - r_neg) / total_ranks)


def bootstrap_ci(
    d: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for mean difference."""
    diff = np.asarray(d, dtype=float)
    n = len(diff)
    if n == 0:
        return (0.0, 0.0)
    boot_means = []
    for _ in range(n_resamples):
        sample = np.random.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))
    sorted_means = np.sort(boot_means)
    lower = float(sorted_means[int((alpha / 2) * n_resamples)])
    upper = float(sorted_means[int((1 - alpha / 2) * n_resamples)])
    return (round(lower, 4), round(upper, 4))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down method for family-wise error rate (FWER) control."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, (m - rank) * p)
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = round(cum_max, 4)
    return adjusted


def wilcoxon_paired(
    x1: Sequence[float],
    x2: Sequence[float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """Paired Wilcoxon Signed-Rank Test for comparing two matched continuous/proportional samples (e.g., H1, H3)."""
    arr1 = np.asarray(x1, dtype=float)
    arr2 = np.asarray(x2, dtype=float)
    diff = arr2 - arr1
    n = len(diff)

    if n == 0 or np.all(diff == 0):
        return {
            "test_type": "Wilcoxon Signed-Rank Test (Paired)",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_diff": round(float(np.mean(diff)), 4) if n > 0 else 0.0,
            "median_diff": round(float(np.median(diff)), 4) if n > 0 else 0.0,
            "hodges_lehmann": 0.0,
            "rank_biserial": 0.0,
            "ci_95": (0.0, 0.0),
            "n_pairs": n,
        }

    res: Any = stats.wilcoxon(arr2, arr1, zero_method="pratt", correction=True, alternative=alternative)
    stat = float(res.statistic)
    pval = float(res.pvalue)

    hl = hodges_lehmann(diff)
    rb = rank_biserial(diff)
    ci = bootstrap_ci(diff, n_resamples=n_bootstrap, alpha=alpha)

    return {
        "test_type": "Wilcoxon Signed-Rank Test (Paired)",
        "statistic": round(stat, 4),
        "p_value": round(pval, 6),
        "significant": bool(pval < alpha),
        "mean_diff": round(float(np.mean(diff)), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "hodges_lehmann": round(hl, 4),
        "rank_biserial": round(rb, 4),
        "ci_95": ci,
        "n_pairs": n,
    }


def wilcoxon_one_sample(
    x: Sequence[float],
    reference: float = 0.0,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """One-Sample Wilcoxon Signed-Rank Test against a fixed reference value (e.g., H2, H4)."""
    arr = np.asarray(x, dtype=float)
    diff = arr - reference
    n = len(diff)

    if n == 0 or np.all(diff == 0):
        return {
            "test_type": "Wilcoxon Signed-Rank Test (One-Sample)",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_diff": round(float(np.mean(diff)), 4) if n > 0 else 0.0,
            "median_diff": round(float(np.median(diff)), 4) if n > 0 else 0.0,
            "hodges_lehmann": 0.0,
            "rank_biserial": 0.0,
            "ci_95": (0.0, 0.0),
            "n_samples": n,
        }

    res: Any = stats.wilcoxon(diff, zero_method="pratt", correction=True, alternative=alternative)
    stat = float(res.statistic)
    pval = float(res.pvalue)

    hl = hodges_lehmann(diff)
    rb = rank_biserial(diff)
    ci = bootstrap_ci(diff, n_resamples=n_bootstrap, alpha=alpha)

    return {
        "test_type": "Wilcoxon Signed-Rank Test (One-Sample)",
        "statistic": round(stat, 4),
        "p_value": round(pval, 6),
        "significant": bool(pval < alpha),
        "mean_diff": round(float(np.mean(diff)), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "hodges_lehmann": round(hl, 4),
        "rank_biserial": round(rb, 4),
        "ci_95": ci,
        "n_samples": n,
    }


class StatisticalAnalyzer:
    """Wrapper around the paired and one-sample Wilcoxon tests."""
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def wilcoxon_paired(
        self,
        x1: Sequence[float],
        x2: Sequence[float],
        alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        return wilcoxon_paired(x1, x2, alternative=alternative, alpha=self.alpha, n_bootstrap=self.n_bootstrap)

    def wilcoxon_one_sample(
        self,
        x: Sequence[float],
        reference: float = 0.0,
        alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        return wilcoxon_one_sample(x, reference=reference, alternative=alternative, alpha=self.alpha, n_bootstrap=self.n_bootstrap)

    def rank_biserial(self, d: np.ndarray) -> float:
        return rank_biserial(d)

    def hodges_lehmann(self, d: np.ndarray) -> float:
        return hodges_lehmann(d)

    def bootstrap_ci(self, d: np.ndarray) -> Tuple[float, float]:
        return bootstrap_ci(d, n_resamples=self.n_bootstrap, alpha=self.alpha)

    def holm_adjust(self, p_values: Sequence[float]) -> List[float]:
        return holm_adjust(p_values)


def mcnemar_test(control: Sequence[int], treatment: Sequence[int], alpha: float = 0.05) -> Dict[str, Any]:
    """McNemar's test for paired binary outcomes (Chapter 3, Table 8b): Edwards' continuity correction when
    b + c >= 25, otherwise the exact binomial test. b = control 1 / treatment 0, c = control 0 / treatment 1."""
    x1 = np.asarray(control, dtype=int)
    x2 = np.asarray(treatment, dtype=int)
    b = int(((x1 == 1) & (x2 == 0)).sum())
    c = int(((x1 == 0) & (x2 == 1)).sum())
    if b + c == 0:
        stat, pval, method = 0.0, 1.0, "no discordant pairs"
    elif b + c >= 25:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        pval, method = float(stats.chi2.sf(stat, 1)), "McNemar with Edwards' correction"
    else:
        stat = float(min(b, c))
        pval, method = float(stats.binomtest(b, b + c, 0.5).pvalue), "exact binomial (b + c < 25)"
    return {
        "test_type": "McNemar (paired binary)",
        "method": method,
        "n": int(len(x1)),
        "b_control_only": b,
        "c_treatment_only": c,
        "statistic": round(float(stat), 4),
        "p_value": round(pval, 6),
        "significant": bool(pval < alpha),
        "rate_control": round(float(x1.mean()), 4) if len(x1) else None,
        "rate_treatment": round(float(x2.mean()), 4) if len(x2) else None,
        "difference": round(float(x2.mean() - x1.mean()), 4) if len(x1) else None,
    }


def paired_continuous_test(control: Sequence[float], treatment: Sequence[float], alpha: float = 0.05,
                           n_permutations: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Chapter 3 rule for a paired continuous outcome: paired t-test when Shapiro-Wilk does not reject normality
    of the differences (p >= 0.05), otherwise a paired sign-flip permutation test on the mean difference."""
    x1 = np.asarray(control, dtype=float)
    x2 = np.asarray(treatment, dtype=float)
    d = x2 - x1
    n = len(d)
    out = {"test_type": "Paired continuous", "n": int(n), "mean_control": round(float(x1.mean()), 4) if n else None,
           "mean_treatment": round(float(x2.mean()), 4) if n else None,
           "mean_diff": round(float(d.mean()), 4) if n else None}
    if n < 3 or np.allclose(d, d[0] if n else 0):
        return {**out, "method": "constant differences", "shapiro_p": None, "statistic": 0.0,
                "p_value": 1.0 if n == 0 or np.allclose(d, 0) else 0.0, "significant": bool(n and not np.allclose(d, 0))}
    shapiro_p = float(stats.shapiro(d).pvalue)
    if shapiro_p >= 0.05:
        res = stats.ttest_rel(x2, x1)
        stat, pval, method = float(res.statistic), float(res.pvalue), "paired t-test"
    else:
        rng = np.random.default_rng(seed)
        observed = abs(d.mean())
        flips = rng.choice([-1.0, 1.0], size=(n_permutations, n))
        perm = np.abs((flips * d).mean(axis=1))
        stat = float(d.mean())
        pval = float((1 + (perm >= observed - 1e-12).sum()) / (n_permutations + 1))
        method = "paired sign-flip permutation (Shapiro-Wilk rejected normality)"
    return {**out, "method": method, "shapiro_p": round(shapiro_p, 6), "statistic": round(stat, 4),
            "p_value": round(pval, 6), "significant": bool(pval < alpha)}
