import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def growregion(temp_kernel, seed_pixel, threshold=8):
    row, col = temp_kernel.shape
    region_grow_image = np.zeros((row, col), dtype=np.uint8)

    i, j = seed_pixel
    region_grow_image[i, j] = 255
    region_points = [(i, j)]

   
    xp = [-1, -1, -1, 0, 0, 1, 1, 1]
    yp = [-1, 0, 1, -1, 1, -1, 0, 1]

    while region_points:
        pt = region_points.pop(0)
        i, j = pt
        intensity = temp_kernel[i, j]
        low = intensity - threshold
        high = intensity + threshold

        for k in range(8):
            ni = i + xp[k]
            nj = j + yp[k]
            if 0 <= ni < row and 0 <= nj < col:
                if region_grow_image[ni, nj] == 0:
                    if low <= temp_kernel[ni, nj] <= high:
                        region_grow_image[ni, nj] = 255
                        region_points.append((ni, nj))
    return region_grow_image

def region_growing_segmentation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found or path is incorrect")
        return

    rows, cols = image.shape
    
    seed_x = random.randint(1, rows - 2)
    seed_y = random.randint(1, cols - 2)
    seed_pixel = (seed_x, seed_y)
    print(f"Randomly selected seed point: {seed_pixel}")

  
    segmented = growregion(image, seed_pixel)

    
    image_with_seed = cv2.imread(image_path)
    cv2.circle(image_with_seed, (seed_y, seed_x), radius=3, color=(255, 0, 0), thickness=-1)
    
    
    image_with_seed_rgb = cv2.cvtColor(image_with_seed, cv2.COLOR_BGR2RGB)

  
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_with_seed_rgb)
    axes[0].set_title("Original Image with Seed")
    axes[0].axis("off")

    axes[1].imshow(segmented, cmap='gray')
    axes[1].set_title("Segmented Region (Region Growing)")
    axes[1].axis("off")

    plt.tight_layout()
    
    
   
    os.makedirs('output', exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f'output/segmented_result__{filename}.png'
    plt.savefig(output_path)
    print(f"Segmented result saved to {output_path}")
    
    
    plt.show()

