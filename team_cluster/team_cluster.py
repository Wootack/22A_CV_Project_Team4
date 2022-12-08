import cv2
import numpy as np
# ISSUE
# get_team 함수가 비효율적
# 원인 1. 하나의 bounding box를 처리하는데 frame 전체를 HSV로 변환하는 과정을 거침
# TODO : 하나의 frame은 한번만 HSV로 바꾸고 각 box에 대해서는 후처리만 하도록 바꾸기.


def team_cluster(id_tlwh_score_per_frame, save_video, video1):
    # INPUT
    # id_tlwh_score_per_frame
    # Content : List of [frame_id, track_ids, tlwhs, scores]
    # tlwh = x1, y1, w, h

    # OUTPUT
    # red_tids = [tid1, tid3, ..., tidn]
    # blue_tids = [tid2, tid5, ..., tidn-1]
    cate_color = {  'red':(0,0,255),
                    'blue':(255,0,0),
                    'referee':(0,0,1),
                    'not on ground':(255,255,254)}

    framd_id=0
    video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frameWidth = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임

    red_tids = []
    blue_tids = []
    color_tid=[(0,0,0) for i in range(100)] # 왜 100? -> tid의 길이가 계속 늘어날것같아서 걍 크게 잡음
    frame_id=0
    while True:
        ret, frame = video1.read()
        if not ret: break
        
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_high = np.array([80,255,255])
        green_low = np.array([40,50,30])

        mask_ground = cv2.inRange(frame_hsv, green_low, green_high)
        
        if frame_id == id_tlwh_score_per_frame[frame_id][0]:
            for tid, (x1, y1, w, h) in zip(*id_tlwh_score_per_frame[frame_id][1:3]):
                center_x = min(frameWidth-1,int((max(0,x1)+min(frameWidth-1,x1+w))/2)) # x+w/2
                center_y = min(frameHeight-1, int((max(0,y1)+min(frameHeight-1,y1+h))/2)) # y+h/2
                # center_x = int(x1 + w/2)
                # center_y = int(y1 + h/2)
                
                # 이거 돌렸을 때 배열 크기 넘어가던데

                
                if color_tid[tid]==(0,0,0) or color_tid[tid]==(255,255,254): # not initialized
                    if mask_ground[int(min(frameHeight-1,y1+h))][int(min(frameWidth-1,x1+w))]!=0\
                        or mask_ground[int(min(frameHeight-1,y1+h))][int(x1)]:  # player
                    # if mask_ground[int(y1+h)][int(x1+w)]!=0 or mask_ground[int(y1+h)][int(x1)]!=0: # bottom-left and right is grass
                        color = tuple([int(x) for x in frame[center_y][center_x]])

                        # depends on team color => parameter tuning
                        if color[2]>80 and color[0]<80:
                            color_tid[tid] = cate_color['red']
                            red_tids.append([frame_id, tid])
                        elif color[0]>80:
                            color_tid[tid] = cate_color['blue']
                            blue_tids.append([frame_id, tid])
                        else:
                            color_tid[tid] = cate_color['referee']
                    else:                               # else
                        color_tid[tid] = cate_color['not on ground']
        frame_id+=1
    if save_video: save_video_result(video1, id_tlwh_score_per_frame)
    return red_tids, blue_tids


def save_video_result(video, id_tlwh_score_per_frame):
    cate_color = {  'red':(0,0,255),
                    'blue':(255,0,0),
                    'referee':(0,0,1),
                    'not on ground':(255,255,254)}

    i = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임

    vid_writer = cv2.VideoWriter('./results/bytetrack_result.mp4',
                cv2.VideoWriter_fourcc(*"mp4v"),
                video.get(cv2.CAP_PROP_FPS),
                (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    color_tid=[(0,0,0) for i in range(100)] # 왜 100? -> tid의 길이가 계속 늘어날것같아서 걍 크게 잡음
    while True:
        ret, frame = video.read()
        if not ret: break
        
        
        if i == id_tlwh_score_per_frame[i][0]:
            for tid, (x1, y1, w, h) in zip(*id_tlwh_score_per_frame[i][1:3]):
                if color_tid[tid]==(0,0,0): # not initialized
                    color_tid[tid] = cate_color[get_team(frame, frameHeight, frameWidth,x1, y1, w, h)]
                elif color_tid[tid]==(255,255,254):
                    color_tid[tid] = cate_color[get_team(frame, frameHeight, frameWidth,x1, y1, w, h)]
                frame = cv2.rectangle(frame, (int(x1), int(y1)),
                            (int(x1+w), int(y1+h)), 
                            color=color_tid[tid], thickness=3)
                frame = cv2.putText(frame,  str(tid), (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_PLAIN, 2, color_tid[tid], 2)
        vid_writer.write(frame)
        i+=1
    print('Result of tracking saved as ./results/bytetrack_result.mp4')


def get_team(frame, frameHeight, frameWidth, x1,y1,w,h):
    # tlwh = x1, y1, w, h
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_high = np.array([80,255,255])
    green_low = np.array([40,50,30])

    mask_ground = cv2.inRange(frame_hsv, green_low, green_high)
    center_x = min(frameWidth-1,int((max(0,x1)+min(frameWidth-1,x1+w))/2)) # x+w/2
    center_y = min(frameHeight-1, int((max(0,y1)+min(frameHeight-1,y1+h))/2)) # y+h/2
    # center_x = x1 + int(w/2)
    # center_y = y1 + int(h/2)
###########################################################################3
    # 이거 돌렸을 때 배열 크기 안넘어감?? 

    if mask_ground[int(min(frameHeight-1,y1+h))][int(min(frameWidth-1,x1+w))]!=0\
        or mask_ground[int(min(frameHeight-1,y1+h))][int(x1)]:  # player
    # if mask_ground[y1+h][x1+w]!=0 or mask_ground[y1+h][x1]!=0: # bottom-left and right is grass
        color = tuple([int(x) for x in frame[center_y][center_x]])

        # depends on team color => parameter tuning
        if color[2]>80 and color[0]<80:
            return 'red'
        elif color[0]>80:
            return 'blue'
        else:
            return 'referee'
    else:                               # else
        return 'not on ground'