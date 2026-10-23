"""Text-to-image recall@k on your own annotated images, for several CLIP models.

Idea: for each prompt, rank all images by similarity to the prompt and check
how many of the images annotated as "right" for that prompt land in the top k.

    recall@k (one prompt) = (right images in the top k) / (all right images)

The final number per model is the average over all prompts.

Based on toy_example/reference_clip_benchmark.py (same scores matrix,
same positive_pairs matrix, same recall formula).

Annotation file: the CSV export of the annotation tool. Each row's `term` is
the prompt and its `image_path` is a right image for it (filename in IMAGE_DIR).

Run from the repo root:  python utility_metrics/recall_at_k.py
"""

import csv
import os

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

# A few downloaded images are slightly cut off; load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Settings: edit these.
# ---------------------------------------------------------------------------
IMAGE_DIR = "data_annotation/images"                          # folder with all the images
ANNOTATIONS = "data_annotation/intervisions_annotations.csv"  # CSV export of the annotation tool
K_LIST = [1, 5, 10]

# Models to compare, as (architecture, pretrained weights) for open_clip.
MODELS = [
    ("ViT-B-32", "openai"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Load the data.
# ---------------------------------------------------------------------------
# Build prompt -> list of right images from the CSV: the term is the prompt,
# image_path is the filename in IMAGE_DIR. Rows whose image is not in the
# folder are skipped.
annotations = {}
skipped = 0
with open(ANNOTATIONS, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if not os.path.isfile(os.path.join(IMAGE_DIR, row["image_path"])):
            skipped += 1
            continue
        annotations.setdefault(row["term"], []).append(row["image_path"])
print(f"Skipped {skipped} annotations with no image in {IMAGE_DIR}")

# Every image in the folder is a candidate in the retrieval, not only the
# annotated ones. Sorted so the image order is the same for every model.
# Empty files (failed downloads on the annotation server) are left out.
image_names = sorted(n for n in os.listdir(IMAGE_DIR)
                     if os.path.getsize(os.path.join(IMAGE_DIR, n)) > 0)
prompts = list(annotations.keys())

# positive_pairs[t, i] = True if image i is a right answer for prompt t.
# Same matrix as in the reference code: one prompt -> MANY images.
positive_pairs = torch.zeros(len(prompts), len(image_names), dtype=torch.bool)
for t, prompt in enumerate(prompts):
    for name in annotations[prompt]:
        positive_pairs[t, image_names.index(name)] = True


# ---------------------------------------------------------------------------
# recall@k: same computation as the reference `recall_at_k`, written simply.
# ---------------------------------------------------------------------------
def recall_at_k(scores, positive_pairs, k):
    # Indices of the k highest-scoring images for each prompt.
    topk_indices = torch.topk(scores, k, dim=1).indices
    # For each prompt, is each of those k images a right one? Count them.
    nb_true_positive = positive_pairs.gather(1, topk_indices).sum(dim=1)
    # How many right images each prompt has in total.
    nb_positive = positive_pairs.sum(dim=1)
    # One recall value per prompt.
    return nb_true_positive / nb_positive


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
        images_emb = F.normalize(torch.cat(images_emb), dim=-1)

        # Embed all prompts.
        texts_emb = F.normalize(model.encode_text(tokenizer(prompts).to(device)), dim=-1)

    # scores[t, i] = similarity between prompt t and image i.
    scores = (texts_emb @ images_emb.T).cpu()

    # Average recall over prompts, for each k.
    print(f"\n{arch} ({pretrained})")
    for k in K_LIST:
        recall = recall_at_k(scores, positive_pairs, k).mean().item()
        print(f"  recall@{k}: {recall:.3f}")
