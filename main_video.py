# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from PoseEstimationUtils import *
from VanishingPointUtils import *
from TeamClassificationUtils import *
from CoreOffsideUtils import *
import demo.demo_multiperson as PoseGetter
from scipy.misc import imread, imsave
# import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import math
import json
import sys
import os
import warnings
warnings.filterwarnings("ignore")



#Image folder path
# base_path = '/home/ameya/Projects/Offside_Detection_Final/Offside/pose_estimation/image_data/filtered_images/'
# base_path = '/home/donghohan/compvision/Computer-Vision-based-Offside-Detection-in-Soccer/outputs/'
base_path = './dataset/Offside_Videos/'
videoFiles = os.listdir(base_path)
fileNames = []
for fileName in videoFiles:
    fileNames.append(base_path+str(fileName))

#Output image paths
out_path = './outputs_video/'
# vanishing_point_viz_base_path = out_path+'vp/'
# pose_estimation_viz_base_path = out_path+'pe/'
# team_classification_viz_base_path = out_path+'tc/'
# offside_viz_base_path = out_path+'final/'
# contour_path = out_path + 'ct/'
# files=[]
# if os.path.exists(vanishing_point_viz_base_path):
# 	files = os.listdir(vanishing_point_viz_base_path)
# 	for file in files:
# 		os.remove(vanishing_point_viz_base_path+file)
# if os.path.exists(pose_estimation_viz_base_path):
# 	files = os.listdir(pose_estimation_viz_base_path)    
# 	for file in files:
# 		os.remove(pose_estimation_viz_base_path+file)
# if os.path.exists(team_classification_viz_base_path):
# 	files = os.listdir(team_classification_viz_base_path)    
# 	for file in files:
# 		os.remove(team_classification_viz_base_path+file)
# if os.path.exists(offside_viz_base_path):
# 	files = os.listdir(offside_viz_base_path)    
# 	for file in files:
# 		os.remove(offside_viz_base_path+file)
# if os.path.exists(contour_path):
# 	files = os.listdir(contour_path)    
# 	for file in files:
# 		os.remove(contour_path+file)
# if os.path.exists(out_path):
# 	dirs = os.listdir(out_path)    
# 	for dir in dirs:
# 		os.rmdir(out_path+dir)
# 	os.rmdir(out_path)
# os.makedirs(vanishing_point_viz_base_path, exist_ok=True)
# os.makedirs(pose_estimation_viz_base_path, exist_ok=True)
# os.makedirs(team_classification_viz_base_path, exist_ok=True)
# os.makedirs(offside_viz_base_path, exist_ok=True)
# os.makedirs(contour_path, exist_ok=True)
# print("Directory Setting Finish.")

#Direction of goal
goalDirection = 'right'

keeper = [4.4931905696916325e-06, 4.450801979411523e-06, 5.510516736414265e-07, 0.00021567314734519837, 0.002188183807439825, 0.0015186984125557716, 0.7527352297592997, 1.0, 0.20787746170678337]
referee = [8.72783130847647e-06, 1.5868784197229944e-07, 0.0, 0.0010298840944002235, 0.0002880184331797235, 0.002688172043010753, 0.3064516129032258, 0.05913978494623656, 1.0]


for file_itr in range(len(fileNames)):
	print('\n\n', fileNames[file_itr])
	cap = cv2.VideoCapture(fileNames[file_itr])
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	fps = cap.get(cv2.CAP_PROP_FPS)
	test = cap.get(cv2.CAP_PROP_FOURCC)
	size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	out = cv2.VideoWriter(out_path+videoFiles[file_itr], fourcc, fps, size)
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			imageForPoseEstimation = frame.copy()
			imageForPoseEstimation_2 = frame.copy()
			imageForOffside = frame.copy()


			vertical_vanishing_point = get_vertical_vanishing_point(frame, goalDirection)
			horizontal_vanishing_point = get_horizontal_vanishing_point(frame)
			# cv2.imwrite(vanishing_point_viz_base_path+videoFiles[file_itr], frame)
			print('Finished Vanishing Point calculation')
			# get pose estimaitons and team classifications
			
			# contour check
			# _, contour_image = PoseGetter.get_contours(imageForPoseEstimation)
			# cv2.imwrite(contour_path+videoFiles[file_itr], contour_image)
			
			pose_estimations, isKeeperFound, isRefFound, temp_image = PoseGetter.return_pose(imageForPoseEstimation_2, imageForPoseEstimation, keeper, referee)
			# cv2.imwrite(base_path+'sub/'+videoFiles[file_itr], temp_image)
			pose_estimations = sorted(pose_estimations, key=lambda x : x[-1][0])
			pose_estimations = update_pose_left_most_point(vertical_vanishing_point, horizontal_vanishing_point, pose_estimations, imageForPoseEstimation, goalDirection)
			print('Finished Pose Estimation & Team Classifiaciton')
			# pose_estimations structure -> [id, teamId, keyPoints, leftmostPoint]
			pose_estimations = get_leftmost_point_angles(vertical_vanishing_point, pose_estimations, imageForPoseEstimation, goalDirection)
			print('Finished updating leftmost point using angle')
			# pose_estimations structure -> [id, teamId, keyPoints, leftmostPoint, angleAtVanishingPoint]
			pose_estimations = sorted(pose_estimations, key=lambda x : x[-1])
			font = cv2.FONT_HERSHEY_SIMPLEX
			# for pose in pose_estimations:
			# 	cv2.putText(imageForPoseEstimation, str(str(pose[0])), (int(pose[-2][-1]), int(pose[-2][0])), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 	cv2.line(imageForPoseEstimation, (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-2][-1]), int(pose[-2][0])), (0,255,0) , 2 )
			# cv2.imwrite(pose_estimation_viz_base_path+videoFiles[file_itr], imageForPoseEstimation)
			# visualize teams
			# imageForTeams = cv2.imread(fileNames[file_itr])
			# for pose in pose_estimations:
			# 	font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(imageForTeams, str(pose[1]), (int(pose[-2][-1]), int(pose[-2][0])), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# cv2.imwrite(team_classification_viz_base_path+videoFiles[file_itr], imageForTeams)
			# get offside decisions
			pose_estimations, last_defending_man = get_offside_decision(pose_estimations, vertical_vanishing_point, 0, 1, isKeeperFound)
			# pose_estimations structure -> [id, teamId, keyPoints, leftmostPoint, angleAtVanishingPoint, offsideDecision]
			print('Starting Core Offside Algorithm')
			
			for pose in pose_estimations:
				if pose[1] == 0:
					if pose[-1] == 'off':
						font = cv2.FONT_HERSHEY_SIMPLEX
						cv2.putText(imageForOffside, 'off', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
						cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
					else:
						font = cv2.FONT_HERSHEY_SIMPLEX
						cv2.putText(imageForOffside, 'on', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
				elif pose[1] == 1:
					if pose[0] == last_defending_man:
						cv2.putText(imageForOffside, 'last man', (int(pose[-3][-1]), int(pose[-3][0]-15)), font, 1, (200,255,155), 2, cv2.LINE_AA)
						cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
					else:
						cv2.putText(imageForOffside, 'def', (int(pose[-3][-1]), int(pose[-3][0]-15)), font, 1, (200,255,155), 2, cv2.LINE_AA)
						cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )				
				elif pose[1] == 2:
					cv2.putText(imageForOffside, 'keep', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
					cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
				elif pose[1] == 3:
					cv2.putText(imageForOffside, 'ref', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
					cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
			# cv2.imwrite(offside_viz_base_path+videoFiles[file_itr][:-4]+'_1.jpg', imageForOffside)
			# exchange attacking and defending teams, get offside decisions
			# pose_estimations, last_defending_man = get_offside_decision(pose_estimations, vertical_vanishing_point, 1, 0, isKeeperFound)
			# pose_estimations structure -> [id, teamId, keyPoints, leftmostPoint, angleAtVanishingPoint, offsideDecision]
			# imageForOffside = cv2.imread(fileNames[file_itr])
			# for pose in pose_estimations:
			# 	if pose[1] == 1:
			# 		if pose[-1] == 'off':
			# 			font = cv2.FONT_HERSHEY_SIMPLEX
			# 			cv2.putText(imageForOffside, 'off', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 			cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
			# 		else:
			# 			font = cv2.FONT_HERSHEY_SIMPLEX
			# 			cv2.putText(imageForOffside, 'on', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 	elif pose[1] == 0:
			# 		if pose[0] == last_defending_man:
			# 			cv2.putText(imageForOffside, 'last man', (int(pose[-3][-1]), int(pose[-3][0]-15)), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 			cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
			# 		else:
			# 			cv2.putText(imageForOffside, 'def', (int(pose[-3][-1]), int(pose[-3][0]-15)), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 			cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
			# 	elif pose[1] == 2:
			# 		cv2.putText(imageForOffside, 'keep', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 		cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
			# 	elif pose[1] == 3:
			# 		cv2.putText(imageForOffside, 'ref', (int(pose[-3][-1]), int(pose[-3][0]-10)), font, 1, (200,255,155), 2, cv2.LINE_AA)
			# 		cv2.line(imageForOffside , (int(vertical_vanishing_point[0]) , int(vertical_vanishing_point[1])) , (int(pose[-3][-1]), int(pose[-3][0])), (0,255,0) , 2 )
			# cv2.imwrite(offside_viz_base_path+videoFiles[file_itr][:-4]+'_2.jpg', imageForOffside)

			out.write(imageForOffside)
		else:
			break
	cap.release()
	out.release()
	# if file_itr == 0:
	# 	break