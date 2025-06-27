import os
import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def add_gaussian_noise_and_threshold(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    noisy_image = add_gaussian_noise(image)


    _, otsu_binarized = cv2.threshold(
    noisy_image, 
    0,                
    255,              
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

    os.makedirs('output', exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f'output/{filename}_original.jpg', image)
    cv2.imwrite(f'output/{filename}_noisy.jpg', noisy_image)
    cv2.imwrite(f'output/{filename}_otsu_binarized.jpg', otsu_binarized)
    

