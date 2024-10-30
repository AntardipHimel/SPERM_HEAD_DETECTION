import cv2
import os

# Define the size of the smaller segments (512x512)
segment_size = 512

# Folder where large images are stored
input_folder = r'C:\Users\rayre\sperm_head_detection\images\train_tif'
output_folder = r'C:\Users\rayre\sperm_head_detection\images\split_images'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Split each image in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_name}")
        continue  # Skip if the image can't be loaded

    img_height, img_width, _ = img.shape

    # Split the image into smaller segments
    for y in range(0, img_height, segment_size):
        for x in range(0, img_width, segment_size):
            segment = img[y:y+segment_size, x:x+segment_size]

            # Define the new image name and save it
            segment_name = f"{img_name.split('.')[0]}_{x}_{y}.jpg"
            segment_path = os.path.join(output_folder, segment_name)
            cv2.imwrite(segment_path, segment)
            print(f"Saved: {segment_path}")  # Print full path of saved image

print("Image splitting completed.")
