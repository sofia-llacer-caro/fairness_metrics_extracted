# Decisions currently baked into the metric code

This document describes two choices that the metric code in `metrics/`
currently makes without stating them, so they can be made explicit later.
It describes and lists options only. It does not pick any of them.

Paths are relative to `fairness_benchmark/metric-robustness/` unless they
start with `metrics/`.


## Decision 1: Which groups are being compared?

### What the code does now

**One attribute axis: gender.** There are two groups, `"Male"` and
`"Female"`. Race, age and any other attribute are not used anywhere in
`metric-robustness/`. `preliminary-results/step1_data_and_labels.py` says
race was left out on purpose, as "a separate decision pending a supervisor
discussion".

The labels come from FairFace's own `gender` column
(`corrections/utils/encode_variation_embeddings.py` L130-131). They are
stored as `gender_labels` in `corrections/embeddings_output/raw/raw.npz`.
That is 10,954 validation images: 5,792 Male and 5,162 Female.

The groups are fixed in a different place for each metric:

- **Skew@k, NDKL, normalized entropy.** A module-level constant,
  `GROUPS = ["Male", "Female"]`, in each file (`metrics/skew.py`,
  `metrics/ndkl.py`, `metrics/normalized_entropy.py`). The functions read
  it directly; it is not an argument.
- **SC-WEAT.** The function itself does not know about groups. The calling
  script decides: `labels == "Male"` becomes `values_a` and
  `labels == "Female"` becomes `values_b`. So a positive effect size means
  "more associated with Male" only because of this caller order.
- **Markedness.** The function does not know about groups either. The group
  enters through the prompt text: `MARKED_WORD = {"Male": "male", "Female":
  "female"}` (`corrections/analysis/markedness_rephrasing_distribution_4words.py`
  L72) plus the rules in `mark_pattern()` that decide where the word goes
  ("a photo of a female robber", "a photo of a friendly female person").
  The metric is computed separately for each group, on that group's images.

### Does any metric assume exactly two groups?

- **SC-WEAT: yes.** It is a difference between two group means. It has no
  form for three or more groups as written.
- **Markedness: no, but each group stands alone.** It gives one value per
  group and never compares groups with each other. More groups means more
  separate values, each needing its own marked word and its own marking
  rule.
- **Skew@k, NDKL, normalized entropy: the formulas work for n groups.** The
  desired share is `1 / len(GROUPS)` and the entropy normalizer is
  `ln(len(GROUPS))`, so these adapt to however many groups are listed. In
  practice, `GROUPS` is hard-coded to two.
- **Labels missing from `GROUPS` are handled differently by each metric.**
  This only matters once there are more labels than listed groups.
  - Skew divides counts by k, so the shares no longer sum to 1.
  - NDKL divides by the rank position, so the same thing happens.
  - Entropy sets k = the sum of the listed groups' counts, so unlisted
    images are silently dropped from the top k.

  With gender in FairFace every image is Male or Female, so this never
  happens today.

### Open questions

- Which axes should be covered: gender only, race, age, or other attributes?
- For race (7 FairFace labels) or age (9 bins), are all categories compared
  together, or are some merged or left out?
- Should groups be intersections (e.g. gender × race), or one axis at a time?
- For SC-WEAT with more than two groups: every pair, one group against the
  rest, or a fixed reference group? Which group is A sets the sign.
- For markedness on other axes: what word marks each group in the prompt,
  and does the same placement rule still read naturally?
- FairFace labels are annotations of perceived attributes, with binary
  gender. Is that acceptable for the benchmark, or does it need to be
  stated as a limitation?
- Should `GROUPS` list every category in the dataset schema, even ones
  absent from a given pool, and what should happen to images whose label
  is not in the list?


## Decision 2: What is the "ideal distribution"?

### What each metric compares against now

- **Skew@k: uniform.** `p_desired = 1 / len(GROUPS)`, so 0.5 / 0.5, inside
  `skew_at_k` (`metrics/skew.py`). A skew of 0 means a group has exactly
  half of the top k.
- **NDKL: uniform.** `d_desired = np.full(len(GROUPS), 1 / len(GROUPS))`
  inside `ndkl` (`metrics/ndkl.py`). NDKL is 0 only if every prefix of the
  ranking is split exactly 50 / 50. It can never reach 0 in practice,
  because the top-1 always holds a single group (KL_1 = ln 2).
- **Normalized entropy @k: uniform, implicitly.** The function has no
  target distribution. Its maximum of 1 is reached only by an equal split
  across all of `GROUPS`, because it divides by `ln(len(GROUPS))`, not by
  the number of groups present.
- **SC-WEAT: no distribution.** It compares group A's mean similarity
  against group B's. The implied "fair" value is 0 (equal means). It does
  not involve a ranking or group proportions.
- **Markedness: no reference value in the code.** It is a percentage per
  group. No value in the code is labelled as fair. Nothing defines whether
  50%, 100%, or equality between the Male and Female values is the target.
  `toy-example/markedness/LIMITATIONS.md` and `corrections/notes.md` already
  note that the two groups' values come from different prompt pairs and
  can't be compared directly.

The FairFace pool is not exactly uniform: 52.88% Male, 47.12% Female. So
against the current uniform target, a ranking of the whole pool scores
Skew = +0.056 (Male) / -0.059 (Female), and KL at the full length is
0.00165, not 0. The pool's normalized entropy is 0.9976, not 1.
`3c_skew_curve_4words.py` draws these whole-pool skews as dotted lines,
but no metric subtracts or corrects for them.

### Options appearing in the code, or that could be used

1. **Uniform over groups (current for Skew, NDKL and entropy).** Each group
   should get an equal share of the results. This asks for equal
   representation regardless of how many images of each group are in the
   pool or truly relevant to the query. So a pool or query that is
   naturally imbalanced counts as unfair.

2. **Pool composition.** The desired share of each group is its share of
   the image pool (here 52.88 / 47.12). This asks that retrieval not change
   the mix that was already in the pool, so a random ranking scores about
   0. The score then depends on how the dataset was assembled, and a
   biased pool counts as the baseline.

3. **Composition of the ground-truth relevant set.** The desired share of
   each group is its share among the images that are actually correct for
   the query. This separates "the model retrieves the wrong mix" from "the
   correct answers are themselves imbalanced". It needs relevance labels
   per query, which FairFace with SoBIT prompts does not have. It ties
   directly to the utility (recall@k) side.

4. **External population statistics.** The desired share comes from
   outside data, e.g. census figures or the gender ratio of an occupation
   for "nurse" or "CEO". This asks retrieval to mirror the world. It
   carries real-world imbalances into the definition of fair, and needs a
   data source per query.

5. **Random-ranking (null) baseline.** Instead of changing the target
   distribution, compare the observed score to the score of randomly
   shuffled rankings of the same pool. This is a way of reading the score
   relative to chance, and can be combined with any of options 1-4.

For the two metrics that are not distribution comparisons:

6. **SC-WEAT: zero difference (current, implicit).** Equal mean similarity
   for both groups. A related reference already exists in the code: the
   label-reshuffling spread in `toy-example/scweat/phase4_statistics/4f_label_reshuffling.py`
   shows what effect sizes look like when the group labels carry no
   information.

7. **Markedness: a reference not yet defined.** Possible readings:
   - equal values for Male and Female;
   - a fixed level such as 50%;
   - the same group's value on a neutral control word.

   Each implies a different meaning of "fair" for this metric.

### Open questions

- Should every ranking metric use the same ideal distribution, or can
  each have its own?
- Is the target fixed per benchmark (uniform, pool) or per query (relevant
  set, population statistics)?
- Under uniform, are the reported values meant to be read against the
  pool's own non-zero baseline (e.g. Skew ±0.06), or as they are?
- If the relevant set is used, where do the relevance labels come from,
  and what happens for a query with no relevant images?
- If population statistics are used, which sources, for which words, and
  what about adjectives ("friendly") that have no population statistic?
- What value counts as "fair" for markedness, and is it compared within
  a group or across groups?
- For SC-WEAT, is 0 the reference, or the spread from label reshuffling
  or from neutral control words?
