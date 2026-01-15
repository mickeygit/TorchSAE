import cv2
import numpy as np

def blur_mask(mask, ksize=21):
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)

def erode_mask(mask, size=5):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)
