# Copied verbatim from metric-robustness/corrections/analysis/markedness_rephrasing_distribution_4words.py
# Source lines: 55, 95-98
# Nothing below this header was changed.


import numpy as np


def markedness_percent(cos_neutral, cos_marked):
    """100 * count(cos_neutral > cos_marked) / n_images. Strict >: a tie is
    NOT a win for the neutral prompt (Phase 1 formula)."""
    return 100 * np.count_nonzero(cos_neutral > cos_marked) / len(cos_neutral)
