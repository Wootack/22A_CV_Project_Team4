import pickle

# import cv2

def main():
    # filename = './data/offside_1.mp4'
    # cap = cv2.VideoCapture(filename)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    b = 'blue'
    r = 'red'
    o = 'touch'
    x = 'no_touch'
    attack_teams =  [b, r, b, r, b, b, r, r, b, r, b, r, r, r, b, r, r, b, r, b]
    receive_touch = [x, o, o, x, x, x, o, x, x, o, o, x, o, o, o, o, x, o, o, o]
    with open('./data/offside_1_state.pickle', 'wb') as fp:
        pickle.dump((attack_teams, receive_touch), fp, pickle.HIGHEST_PROTOCOL)
    return


if __name__=='__main__':
    main()