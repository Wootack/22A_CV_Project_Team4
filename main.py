import os

import cv2

from detect_human import dh_yolo
from video_track import vt_bytetrack

datapath = './data/'

def main():
    os.makedirs('./results', exist_ok=True)
    print('Hi')
    video1 = cv2.VideoCapture(datapath+'Game.mp4')
    # dh_yolo.dh_yolo(video1)
    vt_bytetrack.vt_bytetrack(video1)
    print('DONE')

if __name__ == '__main__':
    main()
