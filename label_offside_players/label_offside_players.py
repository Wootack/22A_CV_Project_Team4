import cv2
# import sys
# sys.path.append('/home/donghohan/compvision/final/22A_CV_Project_Team4/')
from VanishingPointUtils import *

def makeAngles(verticalVP, playerInfo, direction):
    for player in playerInfo:
        anglewithVP = get_angle(verticalVP, (player[1][1], player[1][0]), None, direction)
        player.append(anglewithVP)
        
    return playerInfo
        
def findLastDefender(rtPlayerInfo, btPlayerInfo, attacker):
    # attack:blue, defend:red
    if attacker == 'blue':
        
        keeperAngle = 360.0
        defenderAngle = 360.0
        keeper = -1
        lastDefender = -1        
        for player in rtPlayerInfo:
            if player[2] < keeperAngle:
                keeperAngle = player[2]
                keeper = player[0]
        for player in rtPlayerInfo:
            if player[0] == keeper:
                player.append('keeper')
            else:
                player.append('def')
            # player.append('def')
            if player[2] < defenderAngle and player[0] != keeper:
                defenderAngle = player[2]
                lastDefender = player[0]
                    
        # find on/off
        for player in btPlayerInfo:
            if player[2] < defenderAngle:
                player.append('off')
            else:
                player.append('on')
    # attack:red, defend:blue
    else:
        keeperAngle = 360.0
        defenderAngle = 360.0
        keeper = -1
        lastDefender = -1        
        for player in btPlayerInfo:
            if player[2] < keeperAngle:
                keeperAngle = player[2]
                keeper = player[0]
        for player in btPlayerInfo:
            if player[0] == keeper:
                player.append('keeper')
            else:
                player.append('def')
            # player.append('def')
            if player[2] < defenderAngle and player[0] != keeper:
                defenderAngle = player[2]
                lastDefender = player[0]
                    
        # find on/off
        for player in rtPlayerInfo:
            if player[2] < defenderAngle:
                player.append('off')
            else:
                player.append('on')
                
    return rtPlayerInfo, btPlayerInfo



def label_offside_players(video, startFrame, red_tlwhs_array, blue_tlwhs_array, attacker, direction):
    # red_tlwhs_array
    # np.array object with shape (red_player_num, frame_len, 4)
    # For each red player, for every frame, (x1, y1, w, h) is stored.

    # blue_tlwhs_array
    # Similar with red_tlwhs_array

    # attacker : 'red' or 'blue'
    # direction : 'left' or 'right
    # ATTACKER = 'red'
    # DIRECTION = 'left'
    
    
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fr = 0
    rtPlayerInfo=[]
    btPlayerInfo=[]
    red_tags_list = []    
    blue_tags_list = []
    print("Deciding Offside ...")
    while True:
        ret, frame = video.read()
        if not ret: break
        
        # check offside
        if fr == startFrame:
            verticalVP = get_vertical_vanishing_point(frame, direction)           
            for rtid, fr_xlwh in enumerate(red_tlwhs_array):
                rtid_lu = fr_xlwh[fr, :2]
                if direction=='left':
                    rtidLocate = rtid_lu + [fr_xlwh[fr, 3:4], 0]
                else:
                    rtidLocate = rtid_lu + fr_xlwh[fr, 3:]
                rtPlayerInfo.append([rtid, rtidLocate])
                
            for btid, fr_xlwh in enumerate(blue_tlwhs_array):
                btid_lu = fr_xlwh[fr, :2]
                if direction == 'left':
                    btidLocate = btid_lu + [fr_xlwh[fr, 3:4], 0]
                else:
                    btidLocate = btid_lu + fr_xlwh[fr, 3:]
                btPlayerInfo.append([btid, btidLocate])
                
            rtPlayerInfo = makeAngles(verticalVP, rtPlayerInfo, direction)
            btPlayerInfo = makeAngles(verticalVP, btPlayerInfo, direction)
            
            
            rtPlayerInfo, btPlayerInfo, = findLastDefender(rtPlayerInfo, btPlayerInfo, attacker)
            break
        else:
            fr += 1
        
        
    return rtPlayerInfo, btPlayerInfo
        
        # if fr==0:
        #     verticalVP = get_vertical_vanishing_point(frame, direction)
            
        #     for rtid, fr_xlwh in enumerate(red_tlwhs_array):
        #         rtid_lu = fr_xlwh[fr, :2]
        #         rtid_ld = rtid_lu + [fr_xlwh[fr, 3:4], 0]
        #         rtPlayerInfo.append([rtid, rtid_ld])
                
        #     for btid, fr_xlwh in enumerate(blue_tlwhs_array):
        #         btid_lu = fr_xlwh[fr, :2]
        #         btid_ld = btid_lu + [fr_xlwh[fr, 3:4], 0]
        #         btPlayerInfo.append([btid, btid_ld])
                
        #     rtPlayerInfo = makeAngles(verticalVP, rtPlayerInfo, direction)
        #     btPlayerInfo = makeAngles(verticalVP, btPlayerInfo, direction)
        # else:
        #     continue
        
        
        
        # fr += 1
    # red_tags_list
    # list object with len(red_tags_list) = red_player_num
    # each element is tag to show on output video. {'off', 'on', 'def'}
    
    
    # return red_tags_list, blue_tags_list