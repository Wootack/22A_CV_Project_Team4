# from time import time

import numpy as np
import cv2

from .yolov7.detect import prepare_yolov7, detect

def db_yolo_v7(video, init_ball):
    print('Detecting Ball with YOLO v7 ...')
    h = 10
    w = 10
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    x, y = init_ball
    model, stride, device = prepare_yolov7()
    fr = 0
    ball_loc = np.zeros((int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 4), dtype=int)
    while True:
        ret, frame = video.read()
        if not ret: break
        rs = max(y-200, 0)
        re = min(y+h+200, frame.shape[0])
        cs = max(x-200, 0)
        ce = min(x+w+200, frame.shape[1])
        cframe = frame[rs:re, cs:ce]
        xywhs = detect(model, stride, device, cframe)
        if len(xywhs)!=0:
            min_dist = 9000000
            for xc, yc, wc, hc in xywhs:
                dist = (xc+cs - x)**2 + (yc+rs - y)**2
                if dist < min_dist:
                    min_dist = dist
                    x, y, w, h = xc, yc, wc, hc
            x += cs
            y += rs
            ball_loc[fr] = x, y, w, h
        fr += 1
    return ball_loc


def db_yolo(video, init_ball, isWrite=False):
    print(' with YOLO v3 ...')
    SPORTS_BALL = 32
    h = 10
    w = 10
    vid_writer = None
    if isWrite:
        vid_writer = cv2.VideoWriter('./results/ball_result.mp4',
                cv2.VideoWriter_fourcc(*"mp4v"),
                video.get(cv2.CAP_PROP_FPS),
                (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with open('darknet/cfg/coco.names') as f:
        labels = [line.strip() for line in f]
    network = cv2.dnn.readNet('darknet/cfg/yolov3.weights', 'darknet/cfg/yolov3.cfg')

    ln = network.getLayerNames()
    # print(ln)
    # print(network.getUnconnectedOutLayers())
    # for python=3.7
    ln = [ln[i[0]-1] for i in network.getUnconnectedOutLayers()]
    
    # ln = [ln[i-1] for i in network.getUnconnectedOutLayers()]
    # colors = np.random.uniform(0, 255, size=(len(labels), 3))

    x, y = init_ball
    fr=0
    ball_loc = np.zeros((int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 4), dtype=int)
    while True:
        # starttime = time()
        ret, frame = video.read()
        if not ret: break
        rs = max(y-200, 0)
        re = min(y+h+200, frame.shape[0])
        cs = max(x-200, 0)
        ce = min(x+w+200, frame.shape[1])
        cframe = frame[rs:re, cs:ce]
        height, width = cframe.shape[:2]
        blob = cv2.dnn.blobFromImage(cframe, 1/255., (416, 416), swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(ln)
        confidences = []
        boxes = []
        dists = []
        for out in output_from_network:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != SPORTS_BALL: continue
                confidence = scores[class_id]
                if confidence>0.10:
                    wc = int(detection[2] * width)
                    hc = int(detection[3] * height)
                    xc = int((detection[0] - detection[2]/2) * width)
                    yc = int((detection[1] - detection[3]/2) * height)
                    boxes.append([xc, yc, wc, hc])
                    confidences.append(float(confidence))
                    dists.append((xc+cs-x)**2+(yc+rs-y)**2)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
        boxes = np.asarray(boxes)[indexes]
        confs = np.asarray(confidences)[indexes]
        dsts = np.asarray(dists)[indexes]
        # print(boxes)
        if isWrite:
            if len(boxes) > 0:
                for cf, (xc, yc, wc, hc) in zip(dsts, boxes[0]):
                    x1loc = int(xc+cs)
                    y1loc = int(yc+rs)
                    x2loc = x1loc + int(wc)
                    y2loc = y1loc + int(hc)
                    cv2.rectangle(frame, (x1loc,y1loc), (x2loc,y2loc), (255,255,0), 2)
                    cv2.putText(frame, str(cf), (x1loc,y1loc), 2, 2, (0,0,255), 2)
                vid_writer.write(frame)
        if len(boxes) > 0:
            min_dist = 9000000
            for xc, yc, wc, hc in boxes[0]:
                dist = (xc+cs - x)**2 + (yc+rs - y)**2
                if dist < min_dist:
                    min_dist = dist
                    xcc, ycc, wcc, hcc = xc, yc, wc, hc
            if min_dist < 1200:
                x, y, w, h = xcc, ycc, wcc, hcc
                x += cs
                y += rs
                ball_loc[fr] = x, y, w, h
            # color = colors[0]
            # cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        # cv2.imwrite('./results/{}.png'.format(fr), frame)
        fr+=1
    if isWrite:
        vid_writer.release()
    ball_loc, touch_frames= smooth_ball(ball_loc)
    return ball_loc, touch_frames


def smooth_ball(ball_xywh_array):
    target_ball_col = np.nonzero(ball_xywh_array[:,2])[0]
    frame_len = ball_xywh_array.shape[0]
    inter_ball = np.zeros_like(ball_xywh_array)
    for i in range(4):
        fp = ball_xywh_array[target_ball_col, i].flatten()
        inter_ball[:,i] = np.interp(np.arange(frame_len), target_ball_col, fp)
    # kick_frame = np.argmax(np.abs(np.diff(np.diff(inter_ball, axis=0), axis=0))[:,0])
    a = np.abs(np.diff(np.diff(inter_ball, axis=0), axis=0))
    b = np.square(a[:,:2]).sum(axis=1)
    touch_frames = np.where(b>19)[0]
    return inter_ball, touch_frames