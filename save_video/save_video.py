import cv2

BALL_COLOR = (255,255,0)
RED_COLOR = (0,0,255)
BLUE_COLOR = (255,0,0)
# KEEPER COLOR
YELLOW_COLOR = (0,255,255)
# OFFSIDER COLOR
PINK_COLOR = (255, 153, 255)

GRAY_COLOR = (200, 200, 200)

# def save_video(video, ball_xywh_array, red_tlwhs_array, blue_tlwhs_array, out_path):
def save_video(video, isBall, touchFrameList, startFrame, ball_xywh_array, red_tlwhs_array, blue_tlwhs_array, redPlayerInfo, bluePlayerInfo, out_path):
    print('Saving Result Video ...')
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid_writer = cv2.VideoWriter(out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video.get(cv2.CAP_PROP_FPS),
                    (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fr = 0
    height = 0
    width = 0
    if isBall == True:
        if len(touchFrameList)!=0:
            passStart = touchFrameList[0]
            passGot = touchFrameList[-1]
            print("passStart Frame: ", passStart)
            print("passGot Frame: ", passGot)
            print(touchFrameList)
    
    while True:
        ret, frame = video.read()
        if fr==0:
            height, width, _ = frame.shape
        # if fr == startFrame:
        #     cv2.imwrite("./outputs/testimage.jpg", frame)
        if not ret: break
        cv2.putText(frame, "Frame: "+str(fr), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, GRAY_COLOR, 2)
        if isBall==True:
            # print(ball_xywh_array)
            # print(ball_xywh_array[fr])
            ball_lu = ball_xywh_array[fr, :2]
            ball_rd = ball_lu + ball_xywh_array[fr, 3:]
            # print(ball_lu)
            # print(ball_rd)
            cv2.rectangle(frame, tuple(ball_lu), tuple(ball_rd), BALL_COLOR, 2)
            # cv2.rectangle(frame, ball_lu, ball_rd, BALL_COLOR, 2)
            if len(touchFrameList)!=0:
                if fr>=passStart and fr<passGot:
                    cv2.putText(frame, "Pass Started", (width-500, height-200),
                        cv2.FONT_HERSHEY_PLAIN, 2, YELLOW_COLOR, 7)
                if fr>=passGot and fr <passGot+40:
                    cv2.putText(frame, "OFFSIDE DETECTED", (width-500, height-200),
                        cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 7)
            
        for rtid, fr_xlwh in enumerate(red_tlwhs_array):
            rtid_lu = fr_xlwh[fr, :2]
            rtid_rd = rtid_lu + fr_xlwh[fr, 2:]
            if fr < startFrame:
                cv2.rectangle(frame, tuple(rtid_lu), tuple(rtid_rd), RED_COLOR, 2)
                cv2.putText(frame, str(rtid), tuple(rtid_lu),
                    cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 2)
            else:
                if redPlayerInfo[rtid][-1]=='keeper':
                    cv2.rectangle(frame, tuple(rtid_lu), tuple(rtid_rd), YELLOW_COLOR, 2)
                elif redPlayerInfo[rtid][-1]=='off':
                    cv2.rectangle(frame, tuple(rtid_lu), tuple(rtid_rd), PINK_COLOR, 2)
                else:
                    cv2.rectangle(frame, tuple(rtid_lu), tuple(rtid_rd), RED_COLOR, 2)
                # cv2.rectangle(frame, rtid_lu, rtid_rd, RED_COLOR, 2)
                cv2.putText(frame, str(rtid)+": "+redPlayerInfo[rtid][-1], tuple(rtid_lu),
                        cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 2)
                
                # cv2.putText(frame, str(rtid), rtid_lu,
                        # cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 2)
        for btid, fr_xlwh in enumerate(blue_tlwhs_array):
            btid_lu = fr_xlwh[fr, :2]
            btid_rd = btid_lu + fr_xlwh[fr, 2:]
            if fr < startFrame:
                cv2.rectangle(frame, tuple(btid_lu), tuple(btid_rd), BLUE_COLOR, 2)
                cv2.putText(frame, str(btid), tuple(btid_lu),
                    cv2.FONT_HERSHEY_PLAIN, 2, BLUE_COLOR, 2)
            else:
                if bluePlayerInfo[btid][-1]=='keeper':
                    cv2.rectangle(frame, tuple(btid_lu), tuple(btid_rd), YELLOW_COLOR, 2)
                elif bluePlayerInfo[btid][-1]=='off':
                    cv2.rectangle(frame, tuple(btid_lu), tuple(btid_rd), PINK_COLOR, 2)
                else:
                    cv2.rectangle(frame, tuple(btid_lu), tuple(btid_rd), BLUE_COLOR, 2)
                # cv2.rectangle(frame, btid_lu, btid_rd, BLUE_COLOR, 2)
                cv2.putText(frame, str(btid)+": "+bluePlayerInfo[btid][-1], tuple(btid_lu),
                        cv2.FONT_HERSHEY_PLAIN, 2, BLUE_COLOR, 2)
                # cv2.putText(frame, str(btid), btid_lu,
                #         cv2.FONT_HERSHEY_PLAIN, 2, BLUE_COLOR, 2)
        vid_writer.write(frame)
        fr += 1
    vid_writer.release()
    print('Result Video Saved at', out_path)
    
    
def save_video_basic(video, ball_xywh_array, red_tlwhs_array, blue_tlwhs_array, out_path):
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
        cv2.putText(frame, "Frame: "+str(fr), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, GRAY_COLOR, 2)
        # ball_lu = ball_xywh_array[fr, :2]
        # ball_rd = ball_lu + ball_xywh_array[fr, 3:]
        # # print(ball_lu)
        # # print(ball_rd)
        # cv2.rectangle(frame, tuple(ball_lu), tuple(ball_rd), BALL_COLOR, 2)
        # cv2.rectangle(frame, ball_lu, ball_rd, BALL_COLOR, 2)
        for rtid, fr_xlwh in enumerate(red_tlwhs_array):
            rtid_lu = fr_xlwh[fr, :2]
            rtid_rd = rtid_lu + fr_xlwh[fr, 2:]
            cv2.rectangle(frame, tuple(rtid_lu), tuple(rtid_rd), RED_COLOR, 2)
            # cv2.rectangle(frame, rtid_lu, rtid_rd, RED_COLOR, 2)
            cv2.putText(frame, str(rtid), tuple(rtid_lu),
                    cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 2)
            # cv2.putText(frame, str(rtid), rtid_lu,
                    # cv2.FONT_HERSHEY_PLAIN, 2, RED_COLOR, 2)
        for btid, fr_xlwh in enumerate(blue_tlwhs_array):
            btid_lu = fr_xlwh[fr, :2]
            btid_rd = btid_lu + fr_xlwh[fr, 2:]
            cv2.rectangle(frame, tuple(btid_lu), tuple(btid_rd), BLUE_COLOR, 2)
            # cv2.rectangle(frame, btid_lu, btid_rd, BLUE_COLOR, 2)
            cv2.putText(frame, str(btid), tuple(btid_lu),
                    cv2.FONT_HERSHEY_PLAIN, 2, BLUE_COLOR, 2)
            # cv2.putText(frame, str(btid), btid_lu,
            #         cv2.FONT_HERSHEY_PLAIN, 2, BLUE_COLOR, 2)
        vid_writer.write(frame)
        fr += 1
    vid_writer.release()
    print('Result Video Saved at', out_path)