import cv2
import numpy as np
import os

# Define paths to the dataset
train_images_dir = r"D:\Sem 6\DIP\train\images"
train_masks_dir = r"D:\Sem 6\DIP\train\masks"
test_image_path = r"D:\Sem 6\DIP\test\images\299.bmp"
test_mask_path = r"D:\Sem 6\DIP\test\masks\299.png"

# Function to load a single image and mask
def load_image_and_mask(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return img, mask


train_images = []
for f in os.listdir(train_images_dir):
    if f.endswith(".bmp"):
        full_path = os.path.join(train_images_dir, f)  # Create the full file path
        train_images.append(full_path)  # Add the full path to the list
train_images = sorted(train_images)


train_masks = []
for f in os.listdir(train_masks_dir):
    if f.endswith(".png"):
        full_path = os.path.join(train_masks_dir, f)
        train_masks.append(full_path)
train_masks = sorted(train_masks)  # Sort the list alphabetically

# Load the actual images and masks into memory
train_data = []
for img_path, mask_path in zip(train_images, train_masks):
    img, mask = load_image_and_mask(img_path, mask_path)
    train_data.append((img, mask))  # Add the loaded pair to the list

# Load the test image and mask
test_image, test_mask = load_image_and_mask(test_image_path, test_mask_path)

# 8-connectivity connected components function
def cc_8_connectivity(padded_image, r, c):
    label = 1
    labels = np.zeros_like(padded_image, dtype=int)

    for i in range(1, r - 1):
        for j in range(1, c - 1):
            if padded_image[i, j] == 255:
                neighbor_labels = [
                    labels[i - 1, j],  # Top
                    labels[i, j - 1],  # Left
                    labels[i - 1, j - 1],  # Top-left
                    labels[i - 1, j + 1]  # Top-right
                ]
                filtered_labels = []
                for lbl in neighbor_labels:
                    if lbl > 1:
                        filtered_labels.append(lbl)
                neighbor_labels = filtered_labels

                if neighbor_labels:
                    min_label = min(neighbor_labels)
                    labels[i, j] = min_label
                    for lbl in neighbor_labels:
                        if lbl != min_label:
                            labels[labels == lbl] = min_label
                else:
                    labels[i, j] = label
                    label += 1
    return labels


def power(image,r,c,gamma):
    binary_image = np.zeros_like(image)
    for i in range(r):
        for j in range(c):
            value = int(np.power((image[i, j]/255),gamma)*255)
            binary_image[i, j] = value
    return binary_image

# Pre-processing function
def preprocess_image(image):
    r,c=image.shape
    image = power(image,r,c,1.5)  # Gamma > 1 brightens the image

    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


# Define V set for WBC extraction
def define_v_set_wbc():
    wbc_pixels = []

    for img, mask in train_data:
        img_preprocessed = preprocess_image(img)  # Pre-process the image
        wbc_pixels.extend(img_preprocessed[mask > 0])

    wbc_pixels = np.array(wbc_pixels)
    mean_intensity = np.mean(wbc_pixels)
    std_intensity = np.std(wbc_pixels)
    v_set_wbc = (mean_intensity - 1.5 * std_intensity, mean_intensity + 1.5 * std_intensity)

    return v_set_wbc


# Step 3: Define V set for nucleus and cytoplasm
def define_v_set_nucleus_cytoplasm():
    nucleus_pixels = []
    cytoplasm_pixels = []

    for img, mask in train_data:
        img_preprocessed = preprocess_image(img)  # Pre-process the image
        nucleus_pixels.extend(img_preprocessed[mask == 255])
        cytoplasm_pixels.extend(img_preprocessed[mask == 128])

    nucleus_pixels = np.array(nucleus_pixels)
    cytoplasm_pixels = np.array(cytoplasm_pixels)

    mean_nucleus = np.mean(nucleus_pixels)
    std_nucleus = np.std(nucleus_pixels)
    mean_cytoplasm = np.mean(cytoplasm_pixels)
    std_cytoplasm = np.std(cytoplasm_pixels)

    v_set_nucleus = (max(0, mean_nucleus - 1 * std_nucleus), mean_nucleus + 1 * std_nucleus)
    v_set_cytoplasm = (max(0, mean_cytoplasm - 1.5 * std_cytoplasm), mean_cytoplasm + 1.5 * std_cytoplasm)

    return v_set_nucleus, v_set_cytoplasm


# Function to compute Dice Coefficient
def dice_coefficient(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    intersection = np.logical_and(mask1, mask2)
    denominator = mask1.sum() + mask2.sum()

    if denominator == 0:
        return 1.0

    return 2 * intersection.sum() / denominator


# Segmentation Pipeline
def segmentation_pipeline(image, mask):
    # Pre-process the image
    image = preprocess_image(image)

    # Define V sets using pre-processed images
    v_set_wbc = define_v_set_wbc()
    v_set_nucleus, v_set_cytoplasm = define_v_set_nucleus_cytoplasm()

    print("V set for WBC:", v_set_wbc)
    print("V set for nucleus:", v_set_nucleus)
    print("V set for cytoplasm:", v_set_cytoplasm)

    # Extract WBC
    wbc_mask = np.zeros_like(image)
    wbc_mask[(image >= v_set_wbc[0]) & (image <= v_set_wbc[1])] = 255

    # Extract nucleus and cytoplasm
    nucleus_mask = np.zeros_like(image)
    cytoplasm_mask = np.zeros_like(image)

    nucleus_mask[(image >= v_set_nucleus[0]) & (image <= v_set_nucleus[1]) & (wbc_mask == 255)] = 255
    cytoplasm_mask[(image >= v_set_cytoplasm[0]) & (image <= v_set_cytoplasm[1]) & (wbc_mask == 255)] = 255

    # Apply CCA to WBC mask
    padded_wbc_mask = np.pad(wbc_mask, ((1, 1), (1, 1)), mode='constant')
    labels_wbc = cc_8_connectivity(padded_wbc_mask, *padded_wbc_mask.shape)
    labels_wbc = labels_wbc[1:-1, 1:-1]

    # Retain the largest connected component for WBC
    unique_labels, label_counts = np.unique(labels_wbc, return_counts=True)
    if len(unique_labels) > 1:
        largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]
        wbc_mask = np.where(labels_wbc == largest_label, 255, 0).astype(np.uint8)

    # Apply CCA to nucleus mask
    padded_nucleus_mask = np.pad(nucleus_mask, ((1, 1), (1, 1)), mode='constant')
    labels_nucleus = cc_8_connectivity(padded_nucleus_mask, *padded_nucleus_mask.shape)
    labels_nucleus = labels_nucleus[1:-1, 1:-1]

    # Retain the largest connected component for nucleus
    unique_labels, label_counts = np.unique(labels_nucleus, return_counts=True)
    if len(unique_labels) > 1:
        largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]
        nucleus_mask = np.where(labels_nucleus == largest_label, 255, 0).astype(np.uint8)

    # Apply CCA to cytoplasm mask
    padded_cytoplasm_mask = np.pad(cytoplasm_mask, ((1, 1), (1, 1)), mode='constant')
    labels_cytoplasm = cc_8_connectivity(padded_cytoplasm_mask, *padded_cytoplasm_mask.shape)
    labels_cytoplasm = labels_cytoplasm[1:-1, 1:-1]

    # Retain the largest connected component for cytoplasm
    unique_labels, label_counts = np.unique(labels_cytoplasm, return_counts=True)
    if len(unique_labels) > 1:
        largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]
        cytoplasm_mask = np.where(labels_cytoplasm == largest_label, 255, 0).astype(np.uint8)

    combined_mask = np.zeros_like(image, dtype=np.uint8)
    combined_mask[nucleus_mask > 0] = 255  # Nucleus (255)
    combined_mask[cytoplasm_mask > 0] = 128  # Cytoplasm (128)

    # Compute Dice coefficients
    background_mask = np.where(mask == 0, 1, 0)
    dice_background = dice_coefficient(background_mask, np.where(wbc_mask == 0, 1, 0))
    dice_nucleus = dice_coefficient(np.where(mask == 255, 1, 0), nucleus_mask)
    dice_cytoplasm = dice_coefficient(np.where(mask == 128, 1, 0), cytoplasm_mask)
    dice_total = dice_coefficient(mask, combined_mask)
    print("Dice Coefficients - Background:", dice_background)
    print("Dice Coefficients - Nucleus:", dice_nucleus)
    print("Dice Coefficients - Cytoplasm:", dice_cytoplasm)
    print("Dice Coefficients - Combined:", dice_total)

    return image, mask, wbc_mask, nucleus_mask, cytoplasm_mask, combined_mask


# Run the pipeline
result = segmentation_pipeline(test_image, test_mask)

# Unpack the returned values
image, ground_truth_mask, wbc_mask, nucleus_mask, cytoplasm_mask, combined_mask = result

# Now you can use the variables directly
cv2.imshow("Original Image", image)
cv2.imshow("Ground Truth", ground_truth_mask)
cv2.imshow("WBC Mask", wbc_mask)
cv2.imshow("Nucleus Mask", nucleus_mask)
cv2.imshow("Cytoplasm Mask", cytoplasm_mask)
cv2.imshow("Final Combined Mask", combined_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
