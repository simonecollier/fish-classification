import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

# filepath: /Users/simone/Documents/UofT MSc/CaltechFishCounting/Plot_Labels.py

# Set the image and label paths
images_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/images/'
labels_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/labels/'
seg_labels_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/seg_labels/'
output_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/example_bbox_seg_images/'

os.makedirs(output_path, exist_ok=True)

# Get all image files in the directory and sort them alphanumerically
image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])

# Select image for plotting
selected_image = image_files[300]

def show_box(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Plot the images, bounding boxes, and masks
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Read the image
image_path = os.path.join(images_path, selected_image)
label_file = os.path.join(labels_path, os.path.splitext(selected_image)[0] + ".txt")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Read the label file
with open(label_file, "r") as f:
    lines = f.readlines()

# Generate bounding boxes
bounding_boxes = []
for line in lines:
    parts = line.strip().split()
    _, x_center, y_center, width, height = map(float, parts)
    x_min = (x_center - width / 2) * image.shape[1]
    y_min = (y_center - height / 2) * image.shape[0]
    x_max = (x_center + width / 2) * image.shape[1]
    y_max = (y_center + height / 2) * image.shape[0]
    bounding_boxes.append([x_min, y_min, x_max, y_max])

# Plot the original image with bounding boxes
ax1 = axes[0]
ax1.imshow(image)
show_box(bounding_boxes, ax1)
ax1.axis('off')
ax1.set_title("Original Image with Bounding Boxes")

# Plot the original image with masks overlayed
ax2 = axes[1]
ax2.imshow(image)
for i in range(len(bounding_boxes)):
    mask_path = os.path.join(seg_labels_path, f"{os.path.splitext(selected_image)[0]}_mask_{i}.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_mask[..., 0] = 255  # Red channel
        rgba_mask[..., 3] = mask  # Alpha channel based on mask values
        ax2.imshow(rgba_mask, alpha=0.3)  # Transparent mask overlay
ax2.axis('off')
ax2.set_title("Original Image with Masks Overlayed")

# Save the output
output_file = os.path.join(output_path, f"{os.path.splitext(selected_image)[0]}_comparison.png")
plt.savefig(output_file, bbox_inches='tight')
plt.show()