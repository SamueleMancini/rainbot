import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# === CONFIG ===
BASE_DIR = "/home/andreafabbricatore/rainbot/datasets/final_datasets"
OUTPUT_DIR = "/home/andreafabbricatore/rainbot/pre_processing/segmented/"
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-768-768"

# Cityscapes class mapping
CITYSCAPES_ID2LABEL = {
    0: 'road', 1: 'sidewalk', 2: 'building', 4: 'fence',
    5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation', 9: 'terrain', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck',
    15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
}

# === SETUP ===
# Load model
feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).eval()

def segment_and_save(image_path, country, split):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for class_idx, class_name in CITYSCAPES_ID2LABEL.items():
        mask = (predicted == class_idx).astype(np.uint8)

        if np.any(mask):
            masked_img = image_np.copy()
            masked_img[mask == 0] = 0
            masked_pil = Image.fromarray(masked_img)

            class_country_dir = os.path.join(OUTPUT_DIR, class_name, country, split)
            os.makedirs(class_country_dir, exist_ok=True)

            save_path = os.path.join(class_country_dir, f"{base_name}_{class_name}.png")
            masked_pil.save(save_path)

# === MAIN ===
splits = ['train', 'test', 'val']
image_paths = []

for split in splits:
    split_dir = os.path.join(BASE_DIR, split)
    for root, _, files in os.walk(split_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                # Get country from the path structure
                rel_path = os.path.relpath(full_path, split_dir)
                country = rel_path.split(os.sep)[0]  # First directory after split
                image_paths.append((full_path, country, split))

for image_path, country, split in tqdm(image_paths, desc="Segmenting images"):
    segment_and_save(image_path, country, split)

print("Done. Masks saved in:", OUTPUT_DIR)
