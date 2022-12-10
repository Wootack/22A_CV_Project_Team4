import cv2
import numpy as np


def get_butt_HSV(box):
    h, w = box.shape[:2]
    butt = box[h//2:3*h//4, 2*w//5:3*w//5]
    bh, bw = butt.shape[:2]
    butt_hsv = cv2.cvtColor(butt, cv2.COLOR_BGR2HSV)
    butt_H = butt_hsv[:,:,0]
    butt_S = butt_hsv[:,:,1]
    butt_V = butt_hsv[:,:,2]
    some_butt = np.where(butt_S.reshape(bh, bw, 1)>100, butt_hsv, [0,0,0]).astype(np.uint8)
    some_butt = np.where(butt_V.reshape(bh, bw, 1)>50, some_butt, [0,0,0]).astype(np.uint8)
    some_butt = np.where((butt_H.reshape(bh,bw,1)>80)|(butt_H.reshape(bh,bw,1)<30), some_butt, [0,0,0]).astype(np.uint8)
    bins = np.array([1, 30, 80, 135, 160, 180])
    hist, _ = np.histogram(some_butt[:,:,0].flatten(), bins)
    category = np.argmax(hist)
    if hist.sum() < 10 or category not in [0, 2, 4]: return 0 #'else'
    if category in [0, 4]: return 1 #'red'
    return 2 #'blue'


def team_cluster(human_tids_tlwhs_list, video1):
    print('Clustering Teams ...')
    video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tid_set = set([])
    for tids, tlwhs in human_tids_tlwhs_list:
        tid_set = tid_set.union(set(tids))
    tid_num = len(tid_set)
    tid_list = np.array(tid_set)
    tid_category_counter = np.zeros((tid_num, 3), dtype=int)
    fr = 0
    while True:
        ret, frame = video1.read()
        if not ret: break
        tids, tlwhs = human_tids_tlwhs_list[fr]
        for tid, (x1, y1, w, h) in zip(tids, tlwhs):
            x, y, w, h = int(x1), int(y1), int(w), int(h)
            category = get_butt_HSV(frame[y:y+h, x:x+w])
            tid_category_counter[tid_list.index(tid), category] += 1
        fr += 1
    tid_category = tid_category_counter.argmax(axis=1)
    red_tids = tid_list[tid_category==1]
    blue_tids = tid_list[tid_category==2]
    return red_tids, blue_tids


def team_cluster_obsolete(human_tids_tlwhs_list, save_video, video1):
    # INPUT
    # human_tids_tlwhs_list
    # Content : List of [track_ids, tlwhs]
    # tlwh = x1, y1, w, h

    # OUTPUT
    # red_tids = [tid1, tid3, ..., tidn]
    # blue_tids = [tid2, tid5, ..., tidn-1]
    cate_color = {  'red':(0,0,255),
                    'blue':(255,0,0),
                    'referee':(0,0,1),
                    'not on ground':(255,255,254)}

    tid_set = set([])
    for tids, tlwhs in human_tids_tlwhs_list:
        tid_set = tid_set.union(set(tids))

    video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frameWidth = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임

    red_tids = []
    blue_tids = []
    color_tid=[(0,0,0) for i in range(1+max(list(tid_set)))] # 왜 100? -> tid의 길이가 계속 늘어날것같아서 걍 크게 잡음
    frame_id=0
    while True:
        ret, frame = video1.read()
        if not ret: break
        
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_high = np.array([80,255,255])
        green_low = np.array([40,50,30])

        mask_ground = cv2.inRange(frame_hsv, green_low, green_high)
        
        if frame_id == human_tids_tlwhs_list[frame_id][0]:
            for tid, (x1, y1, w, h) in zip(*human_tids_tlwhs_list[frame_id][1:3]):
                center_x = min(frameWidth-1,int((max(0,x1)+min(frameWidth-1,x1+w))/2)) # x+w/2
                center_y = min(frameHeight-1, int((max(0,y1)+min(frameHeight-1,y1+h))/2)) # y+h/2
                # center_x = int(x1 + w/2)
                # center_y = int(y1 + h/2)
                       
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
    if save_video: save_video_result(video1, human_tids_tlwhs_list)
    return red_tids, blue_tids


def save_video_result(video, human_tids_tlwhs_list):
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
    tid_set = set([])
    for fid, tids, tlwh, scs in human_tids_tlwhs_list:
        tid_set = tid_set.union(set(tids))
    color_tid=[(0,0,0) for i in range(1+max(list(tid_set)))] # 왜 100? -> tid의 길이가 계속 늘어날것같아서 걍 크게 잡음
    while True:
        ret, frame = video.read()
        if not ret: break
        
        
        if i == human_tids_tlwhs_list[i][0]:
            for tid, (x1, y1, w, h) in zip(*human_tids_tlwhs_list[i][1:3]):
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