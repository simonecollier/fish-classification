# Required libraries: segment-anything, opencv-python-headless, matplotlib, torch, torchvision, tqdm

from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Load SAM model
def load_sam_model(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    return SamPredictor(sam), device

def plot_image_with_bboxes_and_masks(image, bounding_boxes, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    for mask in masks:
        plt.imshow(mask[0].cpu().numpy(), alpha=0.5)
    plt.show()

# def process_images_in_batches(base_path, sam_predictor, device, batch_size=10, target_size=(640, 640)):
#     images_path = os.path.join(base_path, "images")
#     labels_path = os.path.join(base_path, "labels")
#     seg_labels_path = os.path.join(base_path, "seg_labels")
#     os.makedirs(seg_labels_path, exist_ok=True)

#     image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]

#     for i in range(0, len(image_files), batch_size):
#         batch_files = image_files[i:i + batch_size]
#         images = []
#         bounding_boxes_batch = []
#         for file in batch_files:
#             img_path = os.path.join(images_path, file)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"Error reading image {img_path}")
#                 continue
#             print(f"Processing image {img_path} with shape {img.shape}")

#             # Resize image to target size
#             resized_img = cv2.resize(img, target_size)
#             images.append(resized_img)

#             # Load labels
#             label_file = os.path.join(labels_path, os.path.splitext(file)[0] + ".txt")
#             if not os.path.exists(label_file):
#                 print(f"No labels found for {file}, skipping...")
#                 continue
#             with open(label_file, "r") as f:
#                 lines = f.readlines()

#             # Generate bounding boxes for each label
#             bounding_boxes = []
#             for line in lines:
#                 parts = line.strip().split()
#                 _, x_center, y_center, width, height = map(float, parts)
#                 x_min = (x_center - width / 2) * target_size[1]
#                 y_min = (y_center - height / 2) * target_size[0]
#                 x_max = (x_center + width / 2) * target_size[1]
#                 y_max = (y_center + height / 2) * target_size[0]
#                 bounding_boxes.append([x_min, y_min, x_max, y_max])

#             if not bounding_boxes:
#                 print(f"No bounding boxes for {file}, skipping...")
#                 continue

#             bounding_boxes_batch.append(bounding_boxes)

#         if not images:
#             continue

#         # Convert list of images to a batch tensor
#         images_tensor = torch.stack([torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in images]).to(device)

#         # Process each image and its bounding boxes with SAM predictor
#         for img, bounding_boxes in zip(images_tensor, bounding_boxes_batch):
#             img_with_channel = img.repeat(3, 1, 1)  # Repeat to create 3 channels
#             sam_predictor.set_image(img_with_channel.cpu().numpy().transpose(1, 2, 0))  # Transpose to (H, W, C)
#             input_boxes = torch.tensor(bounding_boxes, device=device)
#             transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, img.shape[1:])
#             masks, _, _ = sam_predictor.predict_torch(img_with_channel.unsqueeze(0), transformed_boxes)

#             # Save the masks
#             for j, mask in enumerate(masks):
#                 mask_path = os.path.join(seg_labels_path, f"{batch_files[j].split('.')[0]}_mask.png")
#                 cv2.imwrite(mask_path, mask.cpu().numpy())
#                 print(f"Saved mask to {mask_path}")

# Process images in smaller batches and visualize the first image
def process_images_in_batches(base_path, sam_predictor, device, batch_size=10):
    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")
    seg_labels_path = os.path.join(base_path, "seg_labels")
    os.makedirs(seg_labels_path, exist_ok=True)

    image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing Batches"):
        batch = image_files[i:i + batch_size]
        
        # Update the loop in the processing function
        for img_name in batch:
            try:
                # Load and resize image
                image_path = os.path.join(images_path, img_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

                if image is None:
                    print(f"Error loading image {img_name}, skipping...")
                    continue

                # Ensure image is grayscale (single channel) before processing
                if len(image.shape) == 2:  # If the image is grayscale (1 channel)
                    print(f"Grayscale image shape before conversion: {image.shape}")
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                    print(f"Image shape after conversion to RGB: {image.shape}")

                # Resize to 640x640 (note that image now has 3 channels)
                image_resized = cv2.resize(image, (640, 640))  # Resize to 640x640

                print(f"Resized image shape: {image_resized.shape}")  # Debugging line

                # Convert image to a tensor and normalize
                image_resized_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()  # Convert to CxHxW format
                image_resized_tensor /= 255.0  # Normalize to [0, 1]

                print(f"Resized tensor shape: {image_resized_tensor.shape}")  # Debugging line

                # Move to the correct device (GPU or CPU)
                image_resized_tensor = image_resized_tensor.to(device)

                # Set the image for the SAM predictor
                sam_predictor.set_image(image_resized_tensor)

                # Load labels
                label_file = os.path.join(labels_path, os.path.splitext(img_name)[0] + ".txt")
                if not os.path.exists(label_file):
                    print(f"No labels found for {img_name}, skipping...")
                    continue
                with open(label_file, "r") as f:
                    lines = f.readlines()

                # Generate masks for each bounding box
                bounding_boxes = []
                for line in lines:
                    parts = line.strip().split()
                    _, x_center, y_center, width, height = map(float, parts)
                    x_min = (x_center - width / 2) * image_resized.shape[1]
                    y_min = (y_center - height / 2) * image_resized.shape[0]
                    x_max = (x_center + width / 2) * image_resized.shape[1]
                    y_max = (y_center + height / 2) * image_resized.shape[0]
                    bounding_boxes.append([x_min, y_min, x_max, y_max])
                
                print(f"Bounding Boxes: {bounding_boxes}")

                if not bounding_boxes:
                    print(f"No bounding boxes for {img_name}, skipping...")
                    continue

                input_boxes = torch.tensor(bounding_boxes, device=device)
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image_resized.shape[:2])
                print(f"Transformed Boxes: {transformed_boxes}")
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
                print(f"Masks: {masks}")

                if i == 0:  # Only plot the first image for inspection
                    plot_image_with_bboxes_and_masks(image_resized, bounding_boxes, masks)


                # Save masks
                for j, mask in enumerate(masks):
                    mask_path = os.path.join(seg_labels_path, f"{os.path.splitext(img_name)[0]}_{j}.png")
                    cv2.imwrite(mask_path, (mask.cpu().numpy() * 255).astype(np.uint8))
                    print(f"Mask {j} for {img_name} saved at {mask_path}")

                # Clear memory
                del masks, transformed_boxes, input_boxes
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {img_name}: {e}")


# Set up paths and initialize
BASE_PATH = '/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak'
MODEL_TYPE = "vit_h"  # Change if needed
CHECKPOINT_PATH = "/Users/simone/Documents/UofT MSc/CaltechFishCounting/sam_vit_h_4b8939.pth"  # SAM checkpoint file

# Download SAM checkpoint if not already available
# if not os.path.exists(CHECKPOINT_PATH):
#     import subprocess
#     subprocess.run(["wget", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])

# Load SAM and process images
sam_predictor, device = load_sam_model(model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH)
process_images_in_batches(BASE_PATH, sam_predictor, device, batch_size=2)