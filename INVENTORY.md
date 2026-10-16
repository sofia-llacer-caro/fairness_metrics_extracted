# Inventory

Metric code copied from `fairness_benchmark/metric-robustness/`. Each file in
`metrics/` holds the functions word for word, plus the import lines and
module constants they read, copied from the same source file. The header
comment in each file gives the source line numbers.

Source paths below are relative to `metric-robustness/`.

## Metrics

| Metric | File | Source | Signature | Input it consumes | What it measures | Dependencies |
|---|---|---|---|---|---|---|
| SC-WEAT effect size | `metrics/sc_weat.py` | `toy-example/scweat/phase4_statistics/4f_label_reshuffling.py` L60-67 | `sc_weat_effect_size(values_a, values_b)` | Raw similarity scores: two 1-D numpy arrays of cosine similarities between one text prompt and every image of group A, and of group B. | How much higher group A's mean similarity to the prompt is than group B's, divided by the sample std (ddof=1) of both groups pooled. | numpy |
| Markedness | `metrics/markedness.py` | `corrections/analysis/markedness_rephrasing_distribution_4words.py` L95-98 | `markedness_percent(cos_neutral, cos_marked)` | Raw similarity scores: two aligned 1-D arrays for the images of one group, giving each image's cosine to the neutral prompt and to the prompt marked with that group. | The percentage of a group's images that are strictly closer to the neutral prompt than to the group-marked prompt (ties count for the marked side). | numpy |
| Skew@k / MaxSkew@k | `metrics/skew.py` | `toy-example/skew/phase3_full_sweep/3c_skew_curve_4words.py` L73, L90-106 | `skew_at_k(ranked_labels, k)` → `(counts, skews, max_skew, max_group)` | Top-k set: a numpy array of group labels in ranked order (only the first k are used) and k. | For each group, the log of its share of the top k over its desired share; MaxSkew is the largest of these. | math, numpy, module constant `GROUPS` |
| NDKL | `metrics/ndkl.py` | `corrections/analysis/ndkl_rephrasing_distribution_4words.py` L57, L79-91 | `ndkl(ranked_labels)` | Rank positions: a numpy array of group labels for the full ranking, best match first. | The KL divergence between the running group shares and the desired distribution at every rank cut-off, averaged with more weight on top ranks (1/log2(i+1)). | numpy, module constant `GROUPS` |
| Normalized entropy @k | `metrics/normalized_entropy.py` | `corrections/analysis/normalized_entropy_rephrasing_distribution_4words.py` L57-58, L80-90 | `normalized_entropy(counts)` | Top-k set, as counts: a dict `{group: count in top k}`. | The Shannon entropy of the group shares in the top k, divided by ln(number of groups), so 1 means an equal split and 0 means a single group. | math, module constant `GROUPS` |

## Helpers copied alongside (not metrics)

| Function | File | Source | Signature | What it does | Dependencies |
|---|---|---|---|---|---|
| `log_ratio` | `metrics/skew.py` | `3c_skew_curve_4words.py` L90-94 | `log_ratio(p_actual, p_desired)` | `ln(p_actual / p_desired)`, returning `-inf` when `p_actual == 0`. Used by `skew_at_k`. | math |
| `ndkl_for_query` | `metrics/ndkl.py` | `ndkl_rephrasing_distribution_4words.py` L94-97 | `ndkl_for_query(image_embeddings, labels, query_embedding)` | Ranks images by cosine to the query (stable sort, so tied images keep dataset order) and calls `ndkl`. This is the ranking step. | numpy |
| `entropy_term` | `metrics/normalized_entropy.py` | `normalized_entropy_rephrasing_distribution_4words.py` L80-84 | `entropy_term(p)` | `-p ln p`, with 0 for p = 0. Used by `normalized_entropy`. | math |
| `top_k_counts` | `metrics/normalized_entropy.py` | `normalized_entropy_rephrasing_distribution_4words.py` L93-98 | `top_k_counts(image_embeddings, labels, query_embedding, k=K)` | Ranks images by cosine to the query (stable sort), cuts at k, and counts each group. This builds the input for `normalized_entropy`. | numpy, module constants `GROUPS`, `K = 100` |

## Not copied, but needed to reproduce the results

- **Ranking for Skew@k.** The skew scripts have no ranking function. The
  ranking is done inline in `main()` of `3c_skew_curve_4words.py` (L152,
  `np.argsort(-cos, kind="stable")`).
- **Building the marked prompts.** `mark_pattern()` and `MARKED_WORD` in
  `markedness_rephrasing_distribution_4words.py` (L72, L101-114) decide
  where "male"/"female" goes in the prompt. This is prompt construction,
  not metric code.
- **Choosing the SC-WEAT groups.** The calling scripts select
  `labels == "Male"` as group A and `labels == "Female"` as group B (e.g.
  `corrections/analysis/image_variation_distribution_4words.py` L105).
- **Labels.** The `gender_labels` stored in `corrections/embeddings_output/raw/raw.npz`
  come from FairFace's `gender` column
  (`corrections/utils/encode_variation_embeddings.py` L130-131).
- **MinSkew@k.** Not implemented anywhere in `metric-robustness/`.
  `skew_at_k` returns all the per-group skews, so it would be `min(skews.values())`.

## Notes on the copy

- The `import` lines in each file are copied from the source file's own
  imports. Nothing was added or rewritten.
- Other copies exist across the phase scripts. For SC-WEAT, NDKL and skew,
  copies differ in what they return, not in the formula (see the summary).
  The copies chosen here are the ones the `corrections/analysis/` scripts
  call; for skew, which has no `corrections/` script, the most recent
  toy-example sweep (3c) was used.
- The copies reproduce the toy hand calculations from the Phase 1 scripts:
  SC-WEAT -0.81 / 1.40, markedness 60%, NDKL 0.3907 / 0.4739, entropy
  0.8113 at k=4.
