"""Fairness metrics on your own annotated images, for several CLIP models.

Same setup as utility_metrics/recall_at_k.py: each term in the CSV is a
prompt, and for each prompt all labeled images are ranked by similarity.
The fairness metrics then look at the perceived gender of the ranked images.

Labels: perceived_gender 0 -> "Male", 2 -> "Female". Only "single"
annotations are used (couple images have two people, so no single label),
and androgynous/unclear (1) and cannot-determine (-1) are left out, since the
metric files only handle these two groups.

Markedness is not computed: it needs a neutral/marked prompt pair, which the
CSV does not have.

Run from the repo root:  python fairness_metrics/run_fairness.py
"""

import csv
import os

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from ndkl import ndkl_for_query
from normalized_entropy import normalized_entropy, top_k_counts, K
from sc_weat import sc_weat_effect_size
from skew import skew_at_k

# A few downloaded images are slightly cut off; load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Settings: edit these.
# ---------------------------------------------------------------------------
IMAGE_DIR = "data_annotation/images"                          # folder with all the images
ANNOTATIONS = "data_annotation/intervisions_annotations.csv"  # CSV export of the annotation tool
GENDER_LABELS = {"0": "Male", "2": "Female"}

# Models to compare, as (architecture, pretrained weights) for open_clip.
MODELS = [
    ("ViT-B-32", "openai"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Load the data: one gender label per image, and the list of prompts.
# ---------------------------------------------------------------------------
image_names, labels, prompts = [], [], []
with open(ANNOTATIONS, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["term"] not in prompts:
            prompts.append(row["term"])
        if (row["annotation_type"] == "single"
                and row["perceived_gender"] in GENDER_LABELS
                and os.path.isfile(os.path.join(IMAGE_DIR, row["image_path"]))):
            image_names.append(row["image_path"])
            labels.append(GENDER_LABELS[row["perceived_gender"]])
labels = np.array(labels)
print(f"{len(image_names)} labeled images "
      f"({np.sum(labels == 'Male')} Male, {np.sum(labels == 'Female')} Female), "
      f"{len(prompts)} prompts")


# ---------------------------------------------------------------------------
# Run every model.
# ---------------------------------------------------------------------------
for arch, pretrained in MODELS:
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device).eval()

    with torch.no_grad():
        # Embed all images (one at a time to keep it simple), then normalize
        # so the dot product below is the cosine similarity.
        images_emb = []
        for name in image_names:
            image = preprocess(Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB"))
            images_emb.append(model.encode_image(image.unsqueeze(0).to(device)))
        images_emb = F.normalize(torch.cat(images_emb), dim=-1).cpu().numpy()

        # Embed all prompts.
        texts_emb = F.normalize(model.encode_text(tokenizer(prompts).to(device)), dim=-1).cpu().numpy()

    print(f"\n{arch} ({pretrained})")
    print(f"  {'NDKL':>6} {'NE@' + str(K):>7} {'Skew@' + str(K):>8} {'SC-WEAT':>8}  prompt")
    for prompt, query in zip(prompts, texts_emb):
        cos = images_emb @ query
        order = np.argsort(-cos, kind="stable")

        ndkl_value = ndkl_for_query(images_emb, labels, query)
        ne_value = normalized_entropy(top_k_counts(images_emb, labels, query))
        _, _, max_skew, max_group = skew_at_k(labels[order], K)
        # Positive: the prompt is closer to the Female images than to the Male ones.
        sc_weat = sc_weat_effect_size(cos[labels == "Female"], cos[labels == "Male"])

        print(f"  {ndkl_value:6.3f} {ne_value:7.3f} {max_skew:5.3f} {max_group[0]} "
              f"{sc_weat:+8.3f}  {prompt}")
