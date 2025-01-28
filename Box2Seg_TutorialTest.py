import cv2
from matplotlib import pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import os
import time

# Use GPU if available, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load the SAM model
sam_checkpoint = "/Users/simone/Documents/UofT MSc/CaltechFishCounting/sam_vit_h_4b8939.pth"  # SAM checkpoint file
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Set the image and label paths
images_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/images/'
labels_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/labels/'
seg_labels_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/seg_labels/'
failed_bbox_masks_path = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/failed_bbox_masks.txt'
os.makedirs(seg_labels_path, exist_ok=True)

# Get all image files in the directory and sort them alphanumerically
image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])

# Filter out files that already have masks
image_files_to_process = [f for f in image_files if not os.path.exists(os.path.join(seg_labels_path, os.path.splitext(f)[0] + "_mask_0.png"))]
total_files = len(image_files_to_process)

# Start the timer
start_time = time.time()

# Crate masks for each image in a loop
for idx, img_file in enumerate(image_files_to_process):
    image_path = os.path.join(images_path, img_file)
    label_file = os.path.join(labels_path, os.path.splitext(img_file)[0] + ".txt")
    seg_label_file = os.path.join(seg_labels_path, os.path.splitext(img_file)[0] + "_mask.png")
    
    # Skip image if the mask already exists
    if os.path.exists(seg_label_file):
        print(f"Mask for {img_file} already exists, skipping...")
        continue

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    predictor.set_image(image)

    # Read the label file
    if not os.path.exists(label_file):
        print(f"No labels found for {img_file}, skipping...")
        continue
    with open(label_file, "r") as f:
        lines = f.readlines()

    # Generate masks for each bounding box
    bounding_boxes = []
    for line in lines:
        parts = line.strip().split()
        _, x_center, y_center, width, height = map(float, parts)
        x_min = (x_center - width / 2) * image.shape[1]
        y_min = (y_center - height / 2) * image.shape[0]
        x_max = (x_center + width / 2) * image.shape[1]
        y_max = (y_center + height / 2) * image.shape[0]
        bounding_boxes.append([x_min, y_min, x_max, y_max])
    
    input_boxes = torch.tensor(bounding_boxes, device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Save the masks and track failed bounding boxes
    with open(failed_bbox_masks_path, "a") as failed_file:
        for i, mask in enumerate(masks):
            if mask.any():  # Check if the mask is not all False
                mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_path = os.path.join(seg_labels_path, f"{os.path.splitext(img_file)[0]}_mask_{i}.png")
                cv2.imwrite(mask_path, mask_np)
                print(f"Saved mask to {mask_path}")
            else:
                failed_file.write(f"{os.path.splitext(img_file)[0]}_mask_{i}\n")
                print(f"Failed to create mask for {os.path.splitext(img_file)[0]}_mask_{i}")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print(f"Processed {img_file}")
    print(f"Processed {idx + 1}/{total_files} files")
    print(f"Elapsed time: {elapsed_time_str}")

print("Processing complete.")

