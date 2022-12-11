import cv2

def label_offside_players(video, red_tlwhs_array, blue_tlwhs_array, attacker, direction):
    # red_tlwhs_array
    # np.array object with shape (red_player_num, frame_len, 4)
    # For each red player, for every frame, (x1, y1, w, h) is stored.

    # blue_tlwhs_array
    # Similar with red_tlwhs_array

    # attacker : 'red' or 'blue'
    # direction : 'left' or 'right
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fr = 0
    red_tags_list = []
    blue_tags_list = []
    while True:
        ret, frame = video.read()
        if not ret: break
        fr += 1
    # red_tags_list
    # list object with len(red_tags_list) = red_player_num
    # each element is tag to show on output video. {'off', 'on', 'def'}
    return red_tags_list, blue_tags_list