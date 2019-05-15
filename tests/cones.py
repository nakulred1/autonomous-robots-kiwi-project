#!/usr/bin/env python3

import numpy as np
import cv2

def cross(img, x, y, color):
    cv2.line(img, (x-5,y+5), (x+5,y-5), color)
    cv2.line(img, (x+5,y+5), (x-5,y-5), color)

def findCones(img, hsvRanges, verbose=False):
    cones = None
    for i in range(len(hsvRanges)):
        inRange = cv2.inRange(img, hsvRanges[i][0], hsvRanges[i][1])
        if i == 0:
            cones = inRange
        else:
            cones = cv2.bitwise_or(cones, inRange)

    if verbose:
        cv2.imshow("cones", cones)
        cv2.moveWindow("cones", 660, 0)
        cv2.imshow("hsv", img)
        cv2.moveWindow("hsv", 0, 0)
        cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(cones, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)

    if verbose:
        cv2.imshow("dilate", dilate)
        cv2.moveWindow("dilate", 0, 0)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    conePos = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        conePos.append((x + int(w/2), y + h))
    return conePos

img = np.load("img.npy")
img = img[200:480, 0:640]

blur = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
    (640, 280), (0, 280))).astype(np.int32)
cv2.fillPoly(hsv, [pts], (0, 0, 0))

# set verbose to True for the cones you want to see
blueCones = findCones(hsv, [[(97, 78, 35), (130, 255, 100)],
        [(112, 30, 30), (150, 80, 70)]], verbose=False)
yellowCones = findCones(hsv, [[(23, 60, 140), (32, 255, 255)]], verbose=False)
orangeCones = findCones(hsv, [[(0, 80, 110), (8, 180, 200)]], verbose=True)
for cone in blueCones:
    cross(img, cone[0], cone[1], (255, 0, 0))
for cone in yellowCones:
    cross(img, cone[0], cone[1], (0, 255, 255))
for cone in orangeCones:
    cross(img, cone[0], cone[1], (0, 140, 255))

cv2.imshow("img", img)
cv2.moveWindow("img", 0, 0)
cv2.waitKey(0)
