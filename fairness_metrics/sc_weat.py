# Copied verbatim from metric-robustness/toy-example/scweat/phase4_statistics/4f_label_reshuffling.py
# Source lines: 40, 60-67
# Nothing below this header was changed.


import numpy as np


def sc_weat_effect_size(values_a, values_b):
    """SC-WEAT formula, matching Phase 1's validated version exactly."""
    mean_a = values_a.mean()
    mean_b = values_b.mean()
    raw_diff = mean_a - mean_b
    pooled = np.concatenate([values_a, values_b])
    pooled_std = pooled.std(ddof=1)
    return raw_diff / pooled_std
