import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from animatools import animahelpers

# Crops Image/Frame to Centred Square
def crop_to_center_square(frame):
    height, width, _,   = frame.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    end_x = start_x + min_dim
    end_y = start_y + min_dim
    return frame[start_y:end_y, start_x:end_x]



def rescale_image(frame, output_height=512):
    height, width, _,   = frame.shape
    scale = output_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))



def add_uniform_noise(img, noise_level=0.1, offset=0):
    noise = np.random.uniform(low=noise_level*-255, high=noise_level*255, size=img.shape).astype(np.uint8)
    noised_img = np.clip(img + offset + noise, 0, 255).astype(np.uint8)
    return noised_img

def add_normal_noise(img, noise_level=0.1, offset=0):
    noise = np.random.normal(0, noise_level*255, img.shape)
    print(noise.min(), noise.max())
    noised_img = np.clip(img + offset + noise, 0, 255).astype(np.uint8)
    return noised_img
    

def show_cv_imagepair(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Display original image
    axes[0].imshow(img1)
    axes[0].axis('off')
    # Display noised image
    axes[1].imshow(img2)
    axes[1].axis('off')
    plt.show()

# Displays 2D + 3Ch Numpy Array as Image
def display_image(image_data):
    cv2.imshow('Image', image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()