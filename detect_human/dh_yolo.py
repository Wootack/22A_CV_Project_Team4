from time import time

import numpy as np
import cv2

def dh_yolo(video):
    print('yolo_hi')
    PERSON_CLASS = 0
    SPORTS_BALL = 32

    with open('darknet/cfg/coco.names') as f:
        labels = [line.strip() for line in f]
    network = cv2.dnn.readNet('darknet/cfg/yolov3.weights', 'darknet/cfg/yolov3.cfg')

    ln = network.getLayerNames()
    print('noprob')
    ln = [ln[i-1] for i in network.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    fr=0
    while True:
        starttime = time()
        ret, frame = video.read()
        height, width = frame.shape[:2]
        if not ret: break
        blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(ln)
        confidences = []
        boxes = []
        for out in output_from_network:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # if class_id != SPORTS_BALL: continue
                confidence = scores[class_id]
                if confidence>0.1:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int((detection[0] - detection[2]/2) * width)
                    y = int((detection[1] - detection[3]/2) * height)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        boxes = np.asarray(boxes)[indexes]
        for i, box in enumerate(boxes):
            x, y, w, h = box
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.imwrite('./results/{}.png'.format(fr), frame)
        print('Single iter :', time()-starttime, 'Seconds')
        fr+=1
        if fr > 10: break

