import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Generates Iterative Filenames
def generate_next_filename(base_name, extension):
    index = 1
    while True:
        new_filename = f"{base_name}{index}"
        if not os.path.exists(new_filename + extension):
            return new_filename + extension
        index += 1