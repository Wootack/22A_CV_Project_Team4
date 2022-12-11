import cv2

BALL_COLOR = (255,255,0)
RED_COLOR = (0,0,255)
BLUE_COLOR = (255,0,0)

def save_video(video, ball_xywh_array, red_tlwhs_array, blue_tlwhs_array, out_path):
    print('Saving Result Video ...')
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid_writer = cv2.VideoWriter(out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video.get(cv2.CAP_PROP_FPS),
                    (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fr = 0
    while True:
        ret, frame = video.read()
        if not ret: break
        ball_lu = ball_xywh_array[fr, :2]
        ball_rd = ball_lu + ball_xywh_array[fr, 3:]
        cv2.rectangle(frame, ball_lu, ball_rd, BALL_COLOR, 2)
        for rtid, fr_xlwh in enumerate(red_tlwhs_array):
            rtid_lu = fr_xlwh[fr, :2]
            rtid_rd = rtid_lu + fr_xlwh[fr, 3:]
            cv2.rectangle(frame, rtid_lu, rtid_rd, RED_COLOR, 2)
            cv2.putText(frame, str(rtid), rtid_lu,
                    cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 2)
        for btid, fr_xlwh in enumerate(blue_tlwhs_array):
            btid_lu = fr_xlwh[fr, :2]
            btid_rd = btid_lu + fr_xlwh[fr, 3:]
            cv2.rectangle(frame, btid_lu, btid_rd, BLUE_COLOR, 2)
            cv2.putText(frame, str(btid), btid_lu,
                    cv2.FONT_HERSHEY_PLAIN, 2, BLUE_COLOR, 2)
        vid_writer.write(frame)
        fr += 1
    vid_writer.release()
    print('Result Video Saved at', out_path)