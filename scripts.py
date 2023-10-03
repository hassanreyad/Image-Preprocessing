import os
from PIL import Image
import rembg
import cv2
from skimage.io import imread, imshow
from skimage.morphology import closing
import numpy as np
from matplotlib import pyplot as plt

input_folder = "./input"
output_folder = "./bg_removed"
binary_folder = "./binary_image"
desired_width = 800

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(binary_folder):
    os.makedirs(binary_folder)
image_files = [
    f for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png", ".gif"))
]
i = 1
size = len(image_files)
for image_file in image_files:
    print(f"Processing {i} of {size}")
    input_image_path = os.path.join(input_folder, image_file)
    image = Image.open(input_image_path)

    # resize
    width, height = image.size
    aspect_ratio = float(height) / float(width)
    new_height = int(desired_width * aspect_ratio)
    resized_image = image.resize((desired_width, new_height))
    # bg remove
    bg_removed = rembg.remove(resized_image)
    output_image_path = os.path.join(output_folder, "bg_removed_" + str(i) + ".png")
    bg_removed.save(output_image_path, format="PNG")

    # binary
    b_image = cv2.imread(output_image_path)
    imshow(b_image)
    gray = cv2.cvtColor(b_image, cv2.COLOR_BGR2GRAY)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = closing(binary, np.ones((7, 7)))
    output_binary_path = os.path.join(binary_folder, "binary_" + str(i) + ".png")
    cv2.imwrite(output_binary_path, binary)

    image.close()
    i = i + 1

print("Images processing completed.")
