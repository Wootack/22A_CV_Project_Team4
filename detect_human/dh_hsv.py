import cv2
import numpy as np
import os

#def tracking_player(path):
# path = '/home/wooseong/Course/CV/22A_CV_Project_Team4' # path는 수정 가능
# filepath = os.path.join(path,"data/offside1.mp4")
# print(filepath)

def dh_hsv(video1, save_video):
    id_tlwh_score_per_frame = []
    frame_id=0
    while True:
        ret, frame = video1.read()
        if not ret: break
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        green_high = np.array([70,255,255])
        green_low = np.array([20,20,30])

        mask_ground = cv2.inRange(frame_hsv, green_low, green_high)
        # cv2.imshow('a',mask_ground)
        ground_hsv = cv2.bitwise_and(frame, frame, mask=mask_ground)
        player_hsv = frame_hsv - ground_hsv
        player_bgr = cv2.cvtColor(player_hsv,cv2.COLOR_HSV2BGR)


        high_1 = np.array([128, 255, 255])
        low_1= np.array([90, 50, 70])

        low_2 = np.array([0, 100, 100])
        high_2 = np.array([20, 255, 200])
    

        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([20, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([200,100,100])
        upper2 = np.array([255,255,255])
        
        lower_mask = cv2.inRange(player_hsv, lower1, upper1)
        upper_mask = cv2.inRange(player_hsv, lower2, upper2)
        
        mask_2 = lower_mask + upper_mask

        mask_1 = cv2.inRange(player_hsv, low_1, high_1)
        # mask_2 = cv2.inRange(player_hsv, low_2, high_2)


        player_1_hsv = cv2.bitwise_and(player_bgr, player_bgr, mask = mask_1)
        player_2_hsv = cv2.bitwise_and(player_bgr, frame, mask = mask_2)

        p1_bgr = cv2.cvtColor(player_1_hsv, cv2.COLOR_HSV2BGR)
        p2_bgr = cv2.cvtColor(player_2_hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('frame', p2_bgr)

        blue_contours = cv2.findContours(mask_1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        b_c = sorted(blue_contours, key=cv2.contourArea, reverse=True)

        red_contours = cv2.findContours(mask_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        r_c = sorted(red_contours, key=cv2.contourArea, reverse=True)
        tid=0
        for c in b_c:
            x,y,w,h = cv2.boundingRect(c)
            if(h>=1.3*w and w>15 and w<30):
                player_box = frame[y:y+h,x:x+w]
                res1 = cv2.bitwise_and(frame, frame, mask=mask_1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)
                if(nzCount >= 1):
                    #Mark p1
                    if save_video:
                        cv2.putText(frame, 'p1', (x, y-4), font, 0.5, (255,0,0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                    tlwhs=(x,y,w,h)
                    id_tlwh_score_per_frame.append([frame_id,tid,tlwhs,1])
                    tid+=1
        tid=0
        for c in r_c:
            x,y,w,h = cv2.boundingRect(c)
            if(h>=1.3*w and w>15 and w<30):
                player_box = frame[y:y+h,x:x+w]
                res1 = cv2.bitwise_and(frame, frame, mask=mask_2)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)
                if(nzCount >= 1):
                    #Mark p1
                    if save_video:
                        cv2.putText(frame, 'p2', (x, y-4), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                    tlwhs=(x,y,w,h)
                    id_tlwh_score_per_frame.append([frame_id,tid,tlwhs,1])
                    tid+=1
        frame_id += 1
    return id_tlwh_score_per_frame

# def main():
#     if os.path.isfile(filepath):
#         print("File exist")
    
#         try:
#             vid = cv2.VideoCapture(filepath)
#             print("Open file")
#         except:
#             print("Cannnot open file")
#             # return

#         # 프레임을 정수형으로 형 변환
    
#         frameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
#         frameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임


#         fps = vid.get(cv2.CAP_PROP_FPS) # 36
#         # print(fps)
#         frameRate = 30
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         while(vid.isOpened()):
#             # 한 장의 이미지(frame)를 가져오기
#             # 영상 : 이미지(프레임)의 연속
#             # 정상적으로 읽어왔는지 -> ret
#             # 읽어온 프레임 -> frame
#             ret, frame = vid.read()
#             if not(ret):	# 프레임정보를 정상적으로 읽지 못하면
#                 print("read error")
#                 break  # while문을 빠져나가기
    
#             # cv2.imshow('frame',frame)
#             # blurred = cv2.GaussianB   lur(frame, (13, 13), 0)
#             # frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#             frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#             green_high = np.array([70,255,255])
#             green_low = np.array([20,20,30])
    
#             mask_ground = cv2.inRange(frame_hsv, green_low, green_high)
#             cv2.imshow('a',mask_ground)
#             ground_hsv = cv2.bitwise_and(frame, frame, mask=mask_ground)
#             player_hsv = frame_hsv - ground_hsv
#             player_bgr = cv2.cvtColor(player_hsv,cv2.COLOR_HSV2BGR)
    
    
#             high_1 = np.array([128, 255, 255])
#             low_1= np.array([90, 50, 70])
    
#             low_2 = np.array([0, 100, 100])
#             high_2 = np.array([20, 255, 200])
     
    
#             # lower boundary RED color range values; Hue (0 - 10)
#             lower1 = np.array([0, 100, 100])
#             upper1 = np.array([20, 255, 255])
            
#             # upper boundary RED color range values; Hue (160 - 180)
#             lower2 = np.array([200,100,100])
#             upper2 = np.array([255,255,255])
            
#             lower_mask = cv2.inRange(player_hsv, lower1, upper1)
#             upper_mask = cv2.inRange(player_hsv, lower2, upper2)
            
#             mask_2 = lower_mask + upper_mask
    
#             mask_1 = cv2.inRange(player_hsv, low_1, high_1)
#             # mask_2 = cv2.inRange(player_hsv, low_2, high_2)
    
    
#             player_1_hsv = cv2.bitwise_and(player_bgr, player_bgr, mask = mask_1)
#             player_2_hsv = cv2.bitwise_and(player_bgr, frame, mask = mask_2)
    
#             p1_bgr = cv2.cvtColor(player_1_hsv, cv2.COLOR_HSV2BGR)
#             p2_bgr = cv2.cvtColor(player_2_hsv, cv2.COLOR_HSV2BGR)
#             # cv2.imshow('frame', p2_bgr)
    
#             blue_contours = cv2.findContours(mask_1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#             b_c = sorted(blue_contours, key=cv2.contourArea, reverse=True)
    
#             red_contours = cv2.findContours(mask_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#             r_c = sorted(red_contours, key=cv2.contourArea, reverse=True)
    
#             for c in b_c:
#                 x,y,w,h = cv2.boundingRect(c)
#                 if(h>=1.3*w and w>15 and w<30):
#                     player_box = frame[y:y+h,x:x+w]
#                     res1 = cv2.bitwise_and(frame, frame, mask=mask_1)
#                     res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
#                     res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
#                     nzCount = cv2.countNonZero(res1)
#                     if(nzCount >= 1):
# 					    #Mark p1
#                         cv2.putText(frame, 'p1', (x, y-4), font, 0.5, (255,0,0), 2, cv2.LINE_AA)
#                         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    
#             for c in r_c:
#                 x,y,w,h = cv2.boundingRect(c)
#                 if(h>=1.3*w and w>15 and w<30):
#                     player_box = frame[y:y+h,x:x+w]
#                     res1 = cv2.bitwise_and(frame, frame, mask=mask_2)
#                     res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
#                     res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
#                     nzCount = cv2.countNonZero(res1)
#                     if(nzCount >= 1):
# 					    #Mark p1
#                         cv2.putText(frame, 'p2', (x, y-4), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
#                         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    
#             cv2.imshow('p1',frame)
#             # cv2.imshow('player', frame)
            
    
#             key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다
            
#             # 키 입력을 받으면 키값을 key로 저장 -> esc == 27(아스키코드)
#             if key == 27:
#                 break	# while문을 빠져나가기
                
#         if vid.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
#             vid.release()	# 영상 파일(카메라) 사용을 종료
            
#         cv2.destroyAllWindows()
#         # return
#     else:
#         print("File doesn't exist")
#         # return
    
    
# if __name__=='__main__':
#     main()