# Copied verbatim from metric-robustness/corrections/analysis/normalized_entropy_rephrasing_distribution_4words.py
# Source lines: 38, 42, 57-58, 80-98
# Nothing below this header was changed.


import math


import numpy as np


K = 100
GROUPS = ["Male", "Female"]


def entropy_term(p):
    # 0 * ln(0) is taken as 0 (its limit), so a group absent from the top k contributes nothing.
    if p == 0:
        return 0.0
    return -p * math.log(p) + 0.0


def normalized_entropy(counts):
    """H_k^norm from the top-k counts per group (same computation as Phase 1-3b)."""
    k = sum(counts.values())
    return sum(entropy_term(counts[g] / k) for g in GROUPS) / math.log(len(GROUPS))


def top_k_counts(image_embeddings, labels, query_embedding, k=K):
    """Rank the images by cosine to the query (highest first, stable sort)
    and count each group in the top k."""
    order = np.argsort(-(image_embeddings @ query_embedding), kind="stable")
    top_k_labels = labels[order[:k]]
    return {g: int(np.sum(top_k_labels == g)) for g in GROUPS}
