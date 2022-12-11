import cv2
import numpy as np

lower_green = np.array([36,0,0])
upper_green = np.array([86, 255, 255])
stride_kernel = (25,25)
stride_aud = (30,30)
stride_close = (150,150)

def mask_crowd(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = 255 - mask
    kernel_aud = np.ones(stride_aud,np.uint8)
    kernel_close = np.ones(stride_close,np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_aud)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close)
    mask = 255 - mask
    new_fr = np.where(mask.reshape(*image.shape[:2], 1)!=0, image, [0,0,0]).astype(np.uint8)
    return new_fr