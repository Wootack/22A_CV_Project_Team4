from easydict import EasyDict as edict
import cv2

from .ByteTrack.tools.track_video import external_call
from .team_cluster import get_color

TRACK_PARAMS = edict({'fps' : 30,
                'track_thresh' : 0.5,
                'track_buffer' : 30,
                'match_thresh' : 0.8,
                'aspect_ratio_thresh' : 1.6,
                'min_box_area' : 10,
                'mot20' : False})

SAVE_TRACK_VIDEO = True

def vt_bytetrack(video):
    # id_tlwh_score_per_frame
    # Content : List of [frame_id, track_ids, tlwhs, scores]
    # tlwh = x1, y1, w, h
    id_tlwh_score_per_frame = external_call(video, TRACK_PARAMS)
    print('Frame Length :', len(id_tlwh_score_per_frame))
    for frame_id, tids, tlwhs, scores in id_tlwh_score_per_frame:
        assert len(tids) == len(tlwhs)
        assert len(tids) == len(scores)
    if SAVE_TRACK_VIDEO: save_video_result(video, id_tlwh_score_per_frame)


def save_video_result(video, id_tlwh_score_per_frame):
    # def get_color(idx):
    #     idx = idx * 3
    #     color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    #     return color

    i = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임

    vid_writer = cv2.VideoWriter('./results/bytetrack_result.mp4',
                cv2.VideoWriter_fourcc(*"mp4v"),
                video.get(cv2.CAP_PROP_FPS),
                (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    color_tid=[(0,0,0) for i in range(100)]
    while True:
        ret, frame = video.read()
        if not ret: break
        
        
        if i == id_tlwh_score_per_frame[i][0]:
            for tid, (x1, y1, w, h) in zip(*id_tlwh_score_per_frame[i][1:3]):
                if color_tid[tid]==(0,0,0): # not initialized
                    color_tid[tid] = get_color(frame, frameHeight, frameWidth,x1, y1, w, h)
                elif color_tid[tid]==(255,255,254):
                    color_tid[tid] = get_color(frame, frameHeight, frameWidth,x1, y1, w, h)
                frame = cv2.rectangle(frame, (int(x1), int(y1)),
                            (int(x1+w), int(y1+h)), 
                            color=color_tid[tid], thickness=3)
                frame = cv2.putText(frame,  str(tid), (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_PLAIN, 2, color_tid[tid], 2)
        vid_writer.write(frame)
        i+=1
    print('Result of tracking saved as ./results/bytetrack_result.mp4')
    