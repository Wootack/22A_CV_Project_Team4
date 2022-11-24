import os

import numpy as np
import cv2

from detect_human import dh_yolo

datapath = './data/'

def main():
    os.makedirs('./results', exist_ok=True)
    print('Hi')
    video1 = cv2.VideoCapture(datapath+'2022-11-2217-13-37.mp4')
    dh_yolo.dh_yolo(video1)

if __name__ == '__main__':
    main()