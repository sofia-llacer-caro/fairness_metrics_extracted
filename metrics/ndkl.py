# Copied verbatim from metric-robustness/corrections/analysis/ndkl_rephrasing_distribution_4words.py
# Source lines: 42, 57, 79-97
# Nothing below this header was changed.


import numpy as np


GROUPS = ["Male", "Female"]


def ndkl(ranked_labels):
    """NDKL of one ranking (labels highest-cosine first), same computation as
    toy-example/ndkl Phase 3a-3b."""
    n = len(ranked_labels)
    d_desired = np.full(len(GROUPS), 1 / len(GROUPS))
    positions = np.arange(1, n + 1)
    counts = np.stack([np.cumsum(ranked_labels == g) for g in GROUPS], axis=1)  # (N, groups)
    d_actual = counts / positions[:, None]
    # 0 * ln(0) = 0: the log is only taken where the share is positive.
    safe = np.where(d_actual > 0, d_actual, 1.0)
    kl = np.sum(np.where(d_actual > 0, d_actual * np.log(safe / d_desired), 0.0), axis=1)
    w = 1 / np.log2(positions + 1)
    return float(np.sum(w * kl) / np.sum(w))


def ndkl_for_query(image_embeddings, labels, query_embedding):
    """Rank the images by cosine to the query (highest first, stable sort) and return NDKL."""
    order = np.argsort(-(image_embeddings @ query_embedding), kind="stable")
    return ndkl(labels[order])
