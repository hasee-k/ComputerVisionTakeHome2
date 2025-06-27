

from region_growing import region_growing_segmentation
from gaussian_noise_otsu_algorithm import add_gaussian_noise_and_threshold



def main():  
    image_path = 'image3.png'  
    region_growing_segmentation(image_path)
    add_gaussian_noise_and_threshold('image6.png')  



if __name__ == "__main__":
    main()
