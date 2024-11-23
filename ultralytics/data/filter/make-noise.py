import os
import cv2
import numpy as np

def add_salt_and_pepper_noise(image, salt_prob=0.2, pepper_prob=0.2):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    
    # Add salt (white) noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Add pepper (black) noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def process_images_in_directory(directory_path):
    # Create a directory to save the noisy images
    output_dir = os.path.join(directory_path, "noisy-images")
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            
            # Apply salt and pepper noise
            noisy_image = add_salt_and_pepper_noise(image)
            
            # Save the noisy image
            noisy_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(noisy_image_path, noisy_image)
            print(f"Saved noisy image: {noisy_image_path}")

# Path to your image directory
data_directory = "Data"
process_images_in_directory(data_directory)