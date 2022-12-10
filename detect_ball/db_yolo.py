from time import time

import numpy as np
import cv2

def db_yolo(video, init_ball):
    print('Detecting Ball with YOLO v3 ...')
    SPORTS_BALL = 32
    h = 10
    w = 10

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with open('darknet/cfg/coco.names') as f:
        labels = [line.strip() for line in f]
    network = cv2.dnn.readNet('darknet/cfg/yolov3.weights', 'darknet/cfg/yolov3.cfg')

    ln = network.getLayerNames()
    ln = [ln[i-1] for i in network.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    x, y = init_ball
    fr=0
    ball_loc = - np.ones((int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 4), dtype=int)
    while True:
        starttime = time()
        ret, frame = video.read()
        rs = max(y-200, 0)
        re = min(y+h+200, frame.shape[0])
        cs = max(x-200, 0)
        ce = min(x+w+200, frame.shape[1])
        cframe = frame[rs:re, cs:ce]
        height, width = cframe.shape[:2]
        if not ret: break
        blob = cv2.dnn.blobFromImage(cframe, 1/255., (416, 416), swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(ln)
        confidences = []
        boxes = []
        for out in output_from_network:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != SPORTS_BALL: continue
                confidence = scores[class_id]
                if confidence>0.1:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int((detection[0] - detection[2]/2) * width)
                    y = int((detection[1] - detection[3]/2) * height)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        boxes = np.asarray(boxes)[indexes]
        if len(boxes) > 0:
            min_dist = 9000000
            for xc, yc, wc, hc in boxes:
                dist = (xc+cs - x)**2 + (yc+rs - y)**2
                if dist < min_dist:
                    min_dist = dist
                    x, y, w, h = xc, yc, wc, hc
            x += cs
            y += rs
            ball_loc[fr] = x, y, w, h
            color = colors[0]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.imwrite('./results/{}.png'.format(fr), frame)
        fr+=1
        if fr > 10: break
    return ball_loc

