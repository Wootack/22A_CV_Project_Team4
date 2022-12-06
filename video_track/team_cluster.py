import cv2
import numpy as np
import os

def get_color(frame,frameHeight, frameWidth, x1,y1,w,h):
    # tlwh = x1, y1, w, h

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_high = np.array([70,255,255])
    green_low = np.array([20,20,30])

    mask_ground = cv2.inRange(frame_hsv, green_low, green_high)
    foot_x = min(frameWidth-1,int((max(0,x1)+min(frameWidth-1,x1+w))/2)) # x+w/2
    foot_y = min(frameHeight-1, int((max(0,y1)+min(frameHeight-1,y1+h))/2)) # y+h/2
    if mask_ground[int(min(frameHeight-1,y1+h))][int(min(frameWidth-1,x1+w))]!=0 or mask_ground[int(min(frameHeight-1,y1+h))][int(x1)]:  # player
        color = tuple([int(x) for x in frame[foot_y][foot_x]])

        # depends on team color
        if color[2]>80 and color[0]<80:
            return (0,0,255) # red team
        elif color[0]>80:
            return (255,0,0) # blue team
        else:
            return (0,0,0) # refree
    else:                               # else
        return (255,255,255) # not on ground