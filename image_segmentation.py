# EC7212 – Computer Vision and Image Processing
# Take Home Assignment 2
# Python implementation for:
# 1. Adding Gaussian noise and Otsu's algorithm
# 2. Region-growing technique for image segmentation

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def add_gaussian_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def otsu_threshold(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

def region_growing(image, seed_points, threshold=0.05):
    segmented = np.zeros_like(image, dtype=bool)
    visited = np.zeros_like(image, dtype=bool)
    height, width = image.shape
    
    def in_bounds(x, y):
        return 0 <= x < height and 0 <= y < width
    
    stack = list(seed_points)
    seed_value = np.mean([image[x, y] for x, y in seed_points])
    
    while stack:
        x, y = stack.pop()
        if not in_bounds(x, y) or visited[x, y]:
            continue
        visited[x, y] = True
        if abs(image[x, y] - seed_value) <= threshold:
            segmented[x, y] = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        stack.append((x + dx, y + dy))
    return segmented

def main():
    # Create a synthetic image: 64x64 with 2 objects and background
    image = np.zeros((64, 64), dtype=np.float32)
    image[16:32, 16:32] = 0.5  # Object 1
    image[40:56, 40:56] = 0.8  # Object 2
    
    # Normalize for consistency
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Add Gaussian noise
    noisy_image = add_gaussian_noise(image, mean=0, std=0.05)

    # Apply Otsu's algorithm
    otsu_result = otsu_threshold(noisy_image)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title('Noisy Image')
    axes[2].imshow(otsu_result, cmap='gray')
    axes[2].set_title("Otsu's Thresholding")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Region Growing with seeds inside the two objects
    seeds = [(24, 24), (48, 48)]
    region_result = region_growing(noisy_image, seeds, threshold=0.1)

    # Display region growing result
    plt.figure(figsize=(6, 6))
    plt.imshow(region_result, cmap='gray')
    plt.title('Region Growing Result')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
