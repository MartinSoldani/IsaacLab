import torch
import numpy as np
import skimage.data
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt

# Load OWL-ViT model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

# Define image path
image_path = r"C:\Users\marti\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\martin_lift\images\above_blue_fill.jpg"

# Load and process image
image = Image.open(image_path).convert("RGB")
image_resized = processor(images=image, return_tensors="pt")["pixel_values"] # type: ignore

text_queries = ["red cube", "blue cube"]

# Process inputs
inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device) # type: ignore

# Model inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits, scores, and bounding boxes
logits = torch.max(outputs["logits"][0], dim=-1)
scores = torch.sigmoid(logits.values).cpu().detach().numpy()
labels = logits.indices.cpu().detach().numpy()
boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

# Debugging output
print(f"Scores: {scores}")
print(f"Boxes: {boxes}")
print(f"Labels: {labels}")

# Threshold to eliminate low-confidence predictions
score_threshold = 0.015

# Convert bounding boxes to pixel coordinates
def plot_predictions(input_image, text_queries, scores, boxes, labels):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image)
    ax.set_axis_off()

    height, width, _ = input_image.shape

    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue

        cx, cy, w, h = box
        cx, cy, w, h = cx * width, cy * height, w * width, h * height

        rect = plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor="red", linewidth=2) # type: ignore
        ax.add_patch(rect)

        ax.text(
            cx - w / 2,
            cy + h / 2 + 5,
            f"{text_queries[label]}: {score:.2f}",
            color="red",
            bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "round,pad=0.3"},
        )

plot_predictions(np.asarray(image), text_queries, scores, boxes, labels)
plt.show()
