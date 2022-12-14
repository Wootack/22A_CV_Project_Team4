from easydict import EasyDict as edict
import cv2

from .ByteTrack.tools.track_video import external_call

TRACK_PARAMS = edict({'fps' : 30,
                'track_thresh' : 0.5,
                'track_buffer' : 30,
                'match_thresh' : 0.8,
                'aspect_ratio_thresh' : 1.6,
                'min_box_area' : 10,
                'mot20' : False})

save_video = True

def vt_bytetrack(video, save_video):
    # tids_tlwhs_list
    # Content : List of [frame_id, track_ids, tlwhs, scores]
    # tlwh = x1, y1, w, h
    print('Tracking Human with ByteTrack ...')
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tids_tlwhs_list = external_call(video, TRACK_PARAMS)
    save_video_result(video, tids_tlwhs_list)
    return tids_tlwhs_list



def save_video_result(video, tids_tlwhs_list):
    def gen_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    i = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid_writer = cv2.VideoWriter('./results/bytetrack_result.mp4',
                cv2.VideoWriter_fourcc(*"mp4v"),
                video.get(cv2.CAP_PROP_FPS),
                (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while True:
        ret, frame = video.read()
        if not ret: break
        tids, tswhs = tids_tlwhs_list[i]
        for tid, (x1, y1, w, h) in zip(tids, tswhs):
            frame = cv2.rectangle(frame, (int(x1), int(y1)),
                        (int(x1+w), int(y1+h)), 
                        color=gen_color(tid), thickness=2)
            frame = cv2.putText(frame, str(tid), (int(x1), int(y1)),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        vid_writer.write(frame)
        i+=1
    print('Result of tracking saved as ./results/bytetrack_result.mp4')