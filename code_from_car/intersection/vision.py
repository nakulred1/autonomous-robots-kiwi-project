#!/usr/bin/env python3

from multiprocessing import Pool
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

minOrgArea = 60 # only detect cones that are larger (closer) than this


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
    largeCones = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        conePos.append((x + int(w/2), y + h))
        if w*h > minArea:
            largeCones = True
    conePos.sort(key=lambda pt: pt[1])

    if minArea > 0:
        return conePos, largeCones
    return conePos

def _findCarInImg(img):
    #find the black part, the range can be calibrated in the future
    inRange = cv2.inRange(img, (0, 0, 0, 0), (30, 30, 30, 30))

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(inRange, kernel, iterations=12)

    _, contours, _ = cv2.findContours(dilate, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
    Flag_CarFound = False

    if len(contours) != 0:
        # TODO see if boundingRect is faster than contourArea and use the fastest
        #find the biggest area
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        #to make it simple, guess the largest one is the target car
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1] 

        #filter out the false positie
        if cv2.contourArea(biggest_contour) > 750: #2800  we may even set a maxmium limit according to the test
            Flag_CarFound = True
            
    return Flag_CarFound

pool = Pool(processes=2)
def processImage(img, atIntersection):
    img = img[200:480, 0:640] # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
        (640, 280), (0, 280))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0)) # black out the car

    if atIntersection:
        carHsv = hsv[0:130, 250:640]
    else:
        carHsv = hsv[0:130, 420:640]

    carRes = pool.apply_async(_findCarInImg, (carHsv,))
    bluRes = pool.apply_async(_findConesInImg, (hsv, bluRanges))
    ylwRes = pool.apply_async(_findConesInImg, (hsv, ylwRanges))
    orgRes = pool.apply_async(_findConesInImg, (hsv, orgRanges, minOrgArea))

    carFound = carRes.get(1000)
    bluCones = bluRes.get(1000)
    ylwCones = ylwRes.get(1000)
    orgCones, intersectionFound = orgRes.get(1000)

    return (bluCones, ylwCones, orgCones, img.shape[1], img.shape[0], carFound,
        intersectionFound)
