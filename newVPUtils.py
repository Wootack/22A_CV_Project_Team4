import cv2
import numpy as np

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return [x, y]

def find_intersections_Up(lines, height, width):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:
                    # if intersection[0]<0 or intersection[0]>width or intersection[1]<0 or intersection[1]>height:
                    #     intersections.append(intersection)
                        # print("Intersection Point:"+str(intersection[0])+", "+str(intersection[1]))
                    if intersection[0]>0 and intersection[0]<width and intersection[1]<0:
                        intersections.append(intersection)

    return intersections

def get_intersections(lineSet, height, width):
    intersections = find_intersections_Up(lineSet, height, width)
    vanishingPointX = 0.0
    vanishingPointY = 0.0
    for point in intersections:
        vanishingPointX += point[0]
        vanishingPointY += point[1]
    if len(intersections)!=0:
        return (vanishingPointX/len(intersections), vanishingPointY/len(intersections))
    else:
        return (0, 0)

def getVP(image, direction):
    img = image
    height, width, _ = img.shape
    selectedLines = []
    selectedLinesParams = []
    linesFound = False
    BlueRedMask = 100
    
    while linesFound == False:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, BlueRedMask, BlueRedMask), (70, 255,255))
        imask = mask>0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        gray = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        blurImage = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blurImage,150,250,apertureSize = 3) 
        lines = cv2.HoughLinesP(edges,1,np.pi/180, 100, None, 20, 2)
        if lines is None:
            BlueRedMask -= 10
        else:
            if lines.any():
                if len(lines) > 2:  
                    linesFound = True  
                else: 
                    BlueRedMask -= 10
    linePlus=[]
    lineMinus=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist = (x2-x1, y2-y1)
        if dist[0]*dist[1]>=0:
            linePlus.append([(x1,y1),(x2,y2)])
        else:
            lineMinus.append([(x1,y1),(x2,y2)])

    selectedVP = (0, 0)
    if direction == 'left':
        if len(lineMinus)!=0:
            selectedVP = get_intersections(lineMinus, height, width)

    else:
        if len(linePlus)!=0:
            selectedVP = get_intersections(linePlus, height, width)

    # To pixelwise
    # selectedVPpixel = (int(selectedVP[0]), int(selectedVP[1]))
    # return selectedVPpixel
    return selectedVP
