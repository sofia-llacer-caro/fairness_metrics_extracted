# Copied verbatim from metric-robustness/toy-example/skew/phase3_full_sweep/3c_skew_curve_4words.py
# Source lines: 48, 51, 73, 90-106
# Nothing below this header was changed.


import math


import numpy as np


GROUPS = ["Male", "Female"]


def log_ratio(p_actual, p_desired):
    # ln(0) is undefined in math.log; the limit is -inf, which is what Skew means here.
    if p_actual == 0:
        return float("-inf")
    return math.log(p_actual / p_desired)


def skew_at_k(ranked_labels, k):
    """Counts and Skew per group, MaxSkew, and the group reaching it ("tie" if several)."""
    p_desired = 1 / len(GROUPS)
    top_k_labels = ranked_labels[:k]
    counts = {g: int(np.sum(top_k_labels == g)) for g in GROUPS}
    skews = {g: log_ratio(counts[g] / k, p_desired) for g in GROUPS}
    max_skew = max(skews.values())
    max_groups = [g for g in GROUPS if skews[g] == max_skew]
    max_group = max_groups[0] if len(max_groups) == 1 else "tie"
    return counts, skews, max_skew, max_group
