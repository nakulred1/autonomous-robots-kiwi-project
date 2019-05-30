#!/usr/bin/env python3

import numpy as np
import cv2

# range for identifying blue cones in HSV
bluRanges = [
    [(97, 78, 35), (130, 255, 100)], # regular cone blue
    [(112, 30, 30), (150, 80, 70)]  # more of a dark gray
]

# range for identifying yellow cones in HSV
ylwRanges = [
    [(23, 60, 140), (32, 255, 255)]
]

# range for identifying orange cones in HSV
orgRanges = [
    [(0, 80, 110), (8, 180, 200)]
]

minOrgArea = 40 # only detect cones that are larger (closer) than this


def _findConesInImg(img, hsvRanges, minArea=0):
    cones = None
    for i in range(len(hsvRanges)):
        inRange = cv2.inRange(img, hsvRanges[i][0], hsvRanges[i][1])
        if i == 0: cones = inRange
        else:      cones = cv2.bitwise_or(cones, inRange)

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(cones, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)

    _, contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    conePos = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > minArea:
            conePos.append((x + int(w/2), y + h))
    conePos.sort(key=lambda pt: pt[1])
    return conePos

def findCones(img):
    img = img[200:480, 0:640] # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
        (640, 280), (0, 280))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0)) # black out the car

    bluCones = _findConesInImg(hsv, bluRanges)
    ylwCones = _findConesInImg(hsv, ylwRanges)
    orgCones = _findConesInImg(hsv, orgRanges, minArea=minOrgArea)
    return bluCones, ylwCones, orgCones, img.shape[1], img.shape[0]






def checkForCar(img,atintersection):
    if(atintersection):
        img = img[200:330, 250:640] 
    else:
        img = img[200:330, 420:640] # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
     #   (640, 280), (0, 280))).astype(np.int32)
    #cv2.fillPoly(hsv, [pts], (255, 255, 255)) # white out the car

    #find the black part, the range can be calibrated in the future
    inRange = cv2.inRange(hsv, (0,0,0), (30,30,30))
    #cv2.imshow("inRange", inRange)

    kernel = np.ones((3, 3), np.uint8)
    #erode = cv2.erode(inRange, kernel, iterations=1)
    #cv2.imshow("erode", erode)
    dilate = cv2.dilate(inRange, kernel, iterations=12)
    # cv2.imshow("dilate", dilate)

    _, contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Flag_CarFound = False


    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(dilate, contours, -1, 255, 3)

        
        #c = max(contours, key = cv2.contourArea)

        #find the biggest area
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        #to make it simple, guess the largest one is the target car
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1] 

        #print(cv2.contourArea(biggest_contour))

        #filter out the false positie
        if cv2.contourArea(biggest_contour) > 750: #2800  we may even set a maxmium limit according to the test

            x,y,w,h = cv2.boundingRect(biggest_contour)
            mu = cv2.moments(biggest_contour)
            #get the mass center
            mc = (mu['m10']  / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))
            # draw 
            cv2.rectangle(img,(x,y),(x+w,y+h),(10,255,250),2)
            cv2.circle(img, (int(mc[0]), int(mc[1])), 4, (10,255,250), -1)

            #cv2.imshow("dilate", img)

           # if int(mc[0] >= 200): # width = 640, guess the car in 250 should already pass the intersection.
            Flag_CarFound = True
    
            ## I guess dy/dx and current speed can show us whether the target car is running or parking
            #print("x:",(int(mc[0])))
            #print("y:",(int(mc[1])))
            
    return Flag_CarFound
