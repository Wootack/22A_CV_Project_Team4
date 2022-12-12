import cv2
# import sys
# sys.path.append('/home/donghohan/compvision/final/22A_CV_Project_Team4/')
from VanishingPointUtils import *
from newVPUtils import getVP

def makeAngles(verticalVP, playerInfo, direction):
    for player in playerInfo:
        anglewithVP = get_angle(verticalVP, (player[1][0], player[1][1]), None, direction)
        player.append(anglewithVP)
        
    return playerInfo
        
def findLastDefender(rtPlayerInfo, btPlayerInfo, attacker):
    # attack:blue, defend:red
    if attacker == 'blue':
        
        # uses keeper
        # keeperAngle = 360.0
        # defenderAngle = 360.0
        # keeper = -1
        # lastDefender = -1        
        # for player in rtPlayerInfo:
        #     if player[2] < keeperAngle:
        #         keeperAngle = player[2]
        #         keeper = player[0]
        # for player in rtPlayerInfo:
        #     if player[0] == keeper:
        #         player.append('keeper')
        #     else:
        #         player.append('def')
        #     # player.append('def')
        #     if player[2] < defenderAngle and player[0] != keeper:
        #         defenderAngle = player[2]
        #         lastDefender = player[0]


        defenderAngle = 360.0
        lastDefender = -1        
        for player in rtPlayerInfo:
            if player[3] < defenderAngle and player[2]!=0:
                defenderAngle = player[3]
                lastDefender = player[0]
        for player in rtPlayerInfo:
            player.append('def')

        # first_player_degree=0
        # find on/off
        for player in btPlayerInfo:
            if player[3] < defenderAngle and player[2]!=0:
                player.append('off')
            else:
                player.append('on')
            # if player[0]==1:
            #     first_player_degree = player[3]
            print("Player "+str(player[0])+"'s degree is "+str(player[3]))


        print("Last Defender is "+str(lastDefender)+", degree is "+str(defenderAngle))
        # print("Player 1's degree is "+str(first_player_degree))
    # attack:red, defend:blue
    else:
        # uses keeper
        # keeperAngle = 360.0
        # defenderAngle = 360.0
        # keeper = -1
        # lastDefender = -1        
        # for player in btPlayerInfo:
        #     if player[3] < keeperAngle:
        #         keeperAngle = player[3]
        #         keeper = player[0]
        # for player in btPlayerInfo:
        #     if player[0] == keeper:
        #         player.append('keeper')
        #     else:
        #         player.append('def')
        #     # player.append('def')
        #     if player[3] < defenderAngle and player[0] != keeper:
        #         defenderAngle = player[3]
        #         lastDefender = player[0]

        
        defenderAngle = 360.0
        lastDefender = -1        
        for player in btPlayerInfo:
            if player[3] < defenderAngle and player[2]!=0:
                defenderAngle = player[3]
                lastDefender = player[0]
        for player in btPlayerInfo:
            player.append('def')
                    
        # find on/off
        for player in rtPlayerInfo:
            if player[3] < defenderAngle and player[2]!=0:
                player.append('off')
            else:
                player.append('on')
            print("Player "+str(player[0])+"'s degree is "+str(player[3]))
        print("Last Defender is "+str(lastDefender)+", degree is "+str(defenderAngle))
                
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
            print("Get vanishing Points ...")
            # original
            # verticalVP = get_vertical_vanishing_point(frame, direction)           
            # heuristic
            verticalVP = getVP(frame, direction)
            for rtid, fr_xlwh in enumerate(red_tlwhs_array):
                rtid_lu = fr_xlwh[fr, :2]
                if direction=='left':
                    rtidLocate = rtid_lu + [0, fr_xlwh[fr, 3:]]
                else:
                    rtidLocate = rtid_lu + fr_xlwh[fr, 2:]
                rtPlayerInfo.append([rtid, rtidLocate, fr_xlwh[fr, 3]])
                
            for btid, fr_xlwh in enumerate(blue_tlwhs_array):
                btid_lu = fr_xlwh[fr, :2]
                if direction == 'left':
                    btidLocate = btid_lu + [0, fr_xlwh[fr, 3:]]
                else:
                    btidLocate = btid_lu + fr_xlwh[fr, 2:]
                btPlayerInfo.append([btid, btidLocate, fr_xlwh[fr, 3]])
            print("Find each players' angles - Red Team")
            rtPlayerInfo = makeAngles(verticalVP, rtPlayerInfo, direction)
            print("Find each players' angles - Blue Team")
            btPlayerInfo = makeAngles(verticalVP, btPlayerInfo, direction)
            
            print("Deciding last defend player ...")
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