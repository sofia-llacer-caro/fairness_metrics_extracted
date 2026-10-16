"""recall@k as CLIP_benchmark computes it, built up step by step.

Part 1 solves the problem by hand with a loop for one prompt.
Part 2 rebuilds the same number with each line of CLIP_benchmark's
`recall_at_k`, printing the tensor after every line.
Part 3 checks against their function and computes the number they report.

Run:  python toy_example/recall_at_k_walkthrough.py
"""

import torch

from reference_clip_benchmark import recall_at_k  # their function, unedited


# ---------------------------------------------------------------------------
# The toy problem: 3 prompts (texts), 12 images.
# ---------------------------------------------------------------------------

# scores[t, i] = similarity between text t and image i. In the real pipeline
# this is texts_emb @ images_emb.T. Here the numbers are made up, with no ties
# in a row so the ranking is unambiguous.
scores = torch.tensor([
    #  00    01    02    03    04    05    06    07    08    09    10    11
    [0.90, 0.85, 0.80, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.45],  # A
    [0.60, 0.45, 0.25, 0.65, 0.55, 0.35, 0.20, 0.40, 0.15, 0.70, 0.50, 0.30],  # B
    [0.60, 0.55, 0.50, 0.45, 0.40, 0.30, 0.25, 0.35, 0.05, 0.20, 0.15, 0.10],  # C
])

# positive_pairs[t, i] = True if image i is a correct answer for text t.
# Bool, because that is what their code builds. One text -> MANY images.
positive_pairs = torch.tensor([
    # 00 01 02 03 04 05 06 07 08 09 10 11
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A: 3 relevant
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # B: 4 relevant
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # C: 2 relevant
]).bool()

print(f"scores shape:         {tuple(scores.shape)}   (nb_texts, nb_images)")
print(f"positive_pairs shape: {tuple(positive_pairs.shape)}   (nb_texts, nb_images)")


# ---------------------------------------------------------------------------
# PART 1. By hand: one prompt (B), one k (5), a plain loop.
#   recall@k = (relevant images in the top k) / (all relevant images)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 1: by hand, prompt B, k = 5")
print("=" * 70)

t, k = 1, 5  # prompt B, k = 5
row = scores[t].tolist()
rel = positive_pairs[t].tolist()

# Sort the image INDICES by their score, highest first.
ranking = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
print(f"ranking (best first):   {ranking}")
print(f"top {k}:                  {ranking[:k]}")

# Count relevant images among the first k.
found = 0
for i in ranking[:k]:
    found += rel[i]
total = sum(rel)
print(f"relevant in top {k}:      {found}")
print(f"relevant overall:       {total}")
print(f"recall@{k} = {found}/{total} = {found / total:.3f}")


# ---------------------------------------------------------------------------
# PART 2. The same number, line by line as in CLIP_benchmark's recall_at_k.
# Their function does ALL texts at once. We keep all 3 prompts in the tensors
# (as they do) but print prompt B's slice where the full tensor is too big.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: CLIP_benchmark recall_at_k, line by line (k = 5)")
print("=" * 70)

nb_texts, nb_images = scores.shape

# --- Line 1 ---------------------------------------------------------------
# torch.topk returns (values, indices). [1] keeps the indices: for each text,
# the k image indices with the highest scores. Shape (nb_texts, k).
# This is the "sort and take the first k" of Part 1, for every row at once.
topk_indices = torch.topk(scores, k, dim=1)[1]
print(f"\n[1] topk_indices, shape {tuple(topk_indices.shape)}")
print(topk_indices)
print(f"    row B = {topk_indices[t].tolist()}  (same as Part 1's top {k})")

# --- Line 2 ---------------------------------------------------------------
# Number of relevant images per text: count the Trues in each row.
# This is the denominator. Shape (nb_texts,).
nb_positive = positive_pairs.sum(dim=1)
print(f"\n[2] nb_positive, shape {tuple(nb_positive.shape)}")
print(f"    {nb_positive.tolist()}  (A has 3, B has 4, C has 2)")

# --- Line 3 ---------------------------------------------------------------
# Turn every index into a length-nb_images vector with a single 1 at that
# image's position. Each of the k slots becomes one such vector.
# Shape (nb_texts, k, nb_images): for each text, a k x 12 grid where
# row j says "which image is at rank j+1".
topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
print(f"\n[3] topk_indices_onehot, shape {tuple(topk_indices_onehot.shape)}"
      f"  (nb_texts, k, nb_images)")
print("    prompt B's slice (rows = rank 1..5, columns = img00..img11):")
print(topk_indices_onehot[t])

# --- Line 4 ---------------------------------------------------------------
# Add a size-1 middle dimension: (nb_texts, nb_images) -> (nb_texts, 1, nb_images).
# Why: to multiply with the one-hot (nb_texts, k, nb_images), the shapes must
# line up. Broadcasting stretches a size-1 dimension to match the other, so
# the same relevance row is repeated for each of the k slots.
positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
print(f"\n[4] positive_pairs_reshaped, shape {tuple(positive_pairs_reshaped.shape)}")
print(f"    prompt B's row: {positive_pairs_reshaped[t, 0].int().tolist()}")
print(f"    broadcast against [3]: ({nb_texts}, 1, {nb_images}) x "
      f"({nb_texts}, {k}, {nb_images}) -> ({nb_texts}, {k}, {nb_images})")

# --- Line 5 ---------------------------------------------------------------
# Multiply elementwise: a cell stays 1 only if (this image is at this rank)
# AND (this image is relevant). Each rank row has exactly one 1 in the
# one-hot, so after multiplying, each rank row is either all 0 (that image is
# not relevant) or has a single 1 (it is).
product = topk_indices_onehot * positive_pairs_reshaped
print(f"\n[5a] onehot * positive_pairs_reshaped, shape {tuple(product.shape)}")
print("     prompt B's slice:")
print(product[t])

# Sum over dim 1 (the k ranks) and dim 2 (the images) together: adding up
# every 1 in a text's k x 12 grid. Each 1 is one relevant image in the top k,
# so the total is the Part 1 count. The order of summing does not matter.
# Summing ranks first and then images, or images first, or both at once,
# adds the same set of 1s.
nb_true_positive = product.sum(dim=(1, 2))
print(f"\n[5b] nb_true_positive = sum over dim=(1,2), shape {tuple(nb_true_positive.shape)}")
print(f"     {nb_true_positive.tolist()}  (B: {nb_true_positive[t].item()}, "
      f"same as Part 1's {found})")

# --- Line 6 ---------------------------------------------------------------
# Divide the count by the number of relevant images, per text.
recall = nb_true_positive / nb_positive
print(f"\n[6] recall = nb_true_positive / nb_positive")
print(f"    {[round(r, 3) for r in recall.tolist()]}  "
      f"(B: {recall[t].item():.3f}, same as Part 1)")


# ---------------------------------------------------------------------------
# PART 3. Their function, and the number they report.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 3: their recall_at_k, and what CLIP_benchmark reports")
print("=" * 70)

# Their function on the same input must give exactly Part 2's result.
theirs = recall_at_k(scores, positive_pairs, k=5)
assert torch.equal(theirs, recall), "Part 2 does not match recall_at_k"
print(f"\nrecall_at_k(scores, positive_pairs, k=5) = "
      f"{[round(r, 3) for r in theirs.tolist()]}  (matches Part 2)")

# What `evaluate` stores as "image_retrieval_recall@k":
#     (recall_at_k(...) > 0).float().mean()
# > 0      -> per text, True if at least one relevant image is in the top k
# .float() -> True/False becomes 1.0/0.0
# .mean()  -> average over texts
# (batchify only splits the texts into chunks and concatenates the results;
#  each text's recall is computed independently, so it changes nothing.)
print(f"\n{'k':>3}  {'per-prompt recall (A, B, C)':>28}  "
      f"{'reported recall@k':>18}  {'mean actual recall':>18}")
for k in [1, 5, 10]:
    r = recall_at_k(scores, positive_pairs, k=k)
    reported = (r > 0).float().mean().item()
    actual = r.mean().item()
    per_prompt = ", ".join(f"{x:.3f}" for x in r.tolist())
    print(f"{k:>3}  {per_prompt:>28}  {reported:>18.3f}  {actual:>18.3f}")

print("""
# The REPORTED number is (recall > 0).mean(): the share of prompts with at
# least one relevant image in the top k. It is the one comparable to
# published CLIP / OpenCLIP retrieval results. There, each text has exactly
# one correct image (positive_pairs gets one True per row), so "recall > 0"
# and "recall" are the same thing.
#
# With many relevant images per prompt they differ: at k=10 the reported
# number is 1.000 because every prompt found at least one, while the mean
# actual recall (0.750) still shows that A found all, B 3/4, C 1/2.""")
