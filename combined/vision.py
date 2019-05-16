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

orgMinArea = 0 # only detect cones that are larger (closer) than this


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

def findCones(buf):
    img = np.frombuffer(buf, np.uint8).reshape(480, 640, 4)
    img = img[200:480, 0:640] # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
        (640, 280), (0, 280))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0)) # black out the car

    bluCones = _findConesInImg(hsv, bluRanges)
    ylwCones = _findConesInImg(hsv, ylwRanges)
    orgCones = _findConesInImg(hsv, orgRanges, minArea=minOrgArea)
    return bluCones, ylwCones, img.shape[1], img.shape[0]
