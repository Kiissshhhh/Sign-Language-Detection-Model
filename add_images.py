import os
import shutil

# Path to your folder with 'T' images
folder_path = './data/t'

# Count the number of images currently in the folder
images = os.listdir(folder_path)
num_images = len(images)

# Duplicate images until we have 70
for i in range(70 - num_images):
    # Duplicate an existing image
    src_image_path = os.path.join(folder_path, images[i % num_images])
    new_image_path = os.path.join(folder_path, f"duplicate_{i}.jpg")
    shutil.copy(src_image_path, new_image_path)

print("Images duplicated to reach 70.")
