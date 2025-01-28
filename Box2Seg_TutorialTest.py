import cv2
from matplotlib import pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
image = cv2.cvtColor(cv2.imread('/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/images/RB_Nusagak_Sonar_Files_2018_RB_2018-08-06_171000_6300_6600_71.jpg'), cv2.COLOR_BGR2RGB)
# Convert to RGB (pseudo-color)
#image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
sam_checkpoint = "/Users/simone/Documents/UofT MSc/CaltechFishCounting/sam_vit_h_4b8939.pth"  # SAM checkpoint file
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

#label_file = os.path('/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/labels/RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_900_1200_0.txt')
with open('/Users/simone/Documents/UofT MSc/CaltechFishCounting/nushagak/labels/RB_Nusagak_Sonar_Files_2018_RB_2018-08-06_171000_6300_6600_71.txt', "r") as f:
    lines = f.readlines()

buffer = 5  # Adjust the buffer size as needed (in pixels)

# Generate masks for each bounding box
bounding_boxes = []
for line in lines:
    parts = line.strip().split()
    _, x_center, y_center, width, height = map(float, parts)
    x_min = (x_center - width / 2) * image.shape[1]
    y_min = (y_center - height / 2) * image.shape[0]
    x_max = (x_center + width / 2) * image.shape[1]
    y_max = (y_center + height / 2) * image.shape[0]
    
    # Apply buffer while clamping to image bounds
    x_min = max(0, x_min - buffer)
    y_min = max(0, y_min - buffer)
    x_max = min(image.shape[1], x_max + buffer)
    y_max = min(image.shape[0], y_max + buffer)
    
    bounding_boxes.append([x_min, y_min, x_max, y_max])

print(f"Bounding boxes: {bounding_boxes}")

def show_box(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                
plt.figure(figsize=(10, 10))
plt.imshow(image)
#show_mask(masks[0], plt.gca())
show_box(bounding_boxes, plt.gca())
plt.axis('off')
plt.show()

# input_boxes = torch.tensor(bounding_boxes, device=device)
# transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
# print(f"Transformed Boxes: {transformed_boxes}")
# masks, _, _ = predictor.predict_torch(
#     point_coords=None,
#     point_labels=None,
#     boxes=transformed_boxes,
#     multimask_output=False
# )
# print(f"Masks: {masks}")

input_boxes = torch.tensor(bounding_boxes, device=device)
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)


# # Check if any entries in the masks tensor are not False
# if masks.any().item():
#     print("There are entries in the masks tensor that are not False.")
# else:
#     print("All entries in the masks tensor are False.")

# # Find and print the indices of the tensors that have entries that are not False
# non_false_indices = [i for i, tensor in enumerate(masks) if tensor.any().item()]
# print(f"Indices of tensors with entries that are not False: {non_false_indices}")

# # Print the tensors that have entries that are not False
# for i in non_false_indices:
#     print(f"Tensor at index {i} with entries that are not False:")
#     print(masks[i])

# Plot the image and masks overlayed
plt.figure(figsize=(10, 10))
plt.imshow(image)  # Show the background image

# Plot each mask as an overlay
for mask in masks:
    if mask.any():  # Ensure the mask isn't just all False
        mask_np = mask.squeeze().cpu().numpy()
        rgba_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        rgba_mask[..., 0] = 255  # Red channel
        rgba_mask[..., 3] = mask_np * 255  # Alpha channel based on mask values
        plt.imshow(rgba_mask, alpha=0.3)  # Transparent mask overlay

plt.axis('off')
show_box(bounding_boxes, plt.gca())
plt.title("Image with Masks Overlayed")
plt.show()


