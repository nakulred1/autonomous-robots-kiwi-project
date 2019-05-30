#!/usr/bin/env python3

from multiprocessing import Pool
import numpy as np
import cv2

# range for identifying blue cones in HSV
bluRanges = [
    [(97, 78, 35), (130, 255, 100)]
]

# range for identifying blue cones with contrast filter in HSV
bluCRanges = [
    [(94, 50, 35), (130, 255, 100)],
    [(70, 30, 25), (98, 110, 80)],
    [(100, 10, 30), (180, 80, 140)]
]

# range for identifying yellow cones in HSV
ylwRanges = [
    [(23, 60, 140), (32, 255, 255)]
]

# range for identifying orange cones in HSV
orgRanges = [
    [(0, 80, 110), (8, 180, 200)]
]


def _findConesInImg(img, hsvRanges, contrastFilter=[], hsvCRanges=[]):
    cones = None
    for i in range(len(hsvRanges)):
        inRange = cv2.inRange(img, hsvRanges[i][0], hsvRanges[i][1])
        if i == 0: cones = inRange
        else:      cones = cv2.bitwise_or(cones, inRange)

    if len(contrastFilter) > 0 and len(hsvCRanges) > 0:
        cCones = []
        for i in range(len(hsvCRanges)):
            inRange = cv2.inRange(img, hsvCRanges[i][0], hsvCRanges[i][1])
            if i == 0: cCones = inRange
            else:      cCones = cv2.bitwise_or(cCones, inRange)
        cCones = cv2.bitwise_and(cCones, contrastFilter)
        cones = cv2.bitwise_or(cones, cCones)

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(cones, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)

    _, contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    conePos = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        conePos.append((x + int(w/2), y + h))
    return conePos

def _findBluCones(hsv, blur, pts):
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0))
    cv2.fillPoly(grad, [pts], 0)
    grad = cv2.boxFilter(grad, -1, (18, 18))
    contrastFilter = cv2.inRange(grad, 18, 255)
    return _findConesInImg(hsv, bluRanges, contrastFilter, bluCRanges)

pool = Pool(processes=2)
def findCones(buf):
    img = np.frombuffer(buf, np.uint8).reshape(480, 640, 4)
    img = img[200:390, 0:640] # remove the top and bottom of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    pts = np.array(((0, 190), (0, 170), (200, 130), (420, 135), (640, 190),
        (0, 190))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0)) # black out the car

    bluRes = pool.apply_async(_findBluCones, (hsv, blur, pts))
    ylwRes = pool.apply_async(_findConesInImg, (hsv, ylwRanges))
    orgRes = pool.apply_async(_findConesInImg, (hsv, orgRanges))

    bluCones = bluRes.get(1000)
    ylwCones = ylwRes.get(1000)
    orgCones = orgRes.get(1000)

    orgCones.sort(key=lambda pt: pt[0])
    if len(orgCones) == 2 and orgCones[1][0] - orgCones[0][0] > 50:
        bluCones.append(orgCones[1])
        ylwCones.append(orgCones[0])

    bluCones.sort(key=lambda pt: pt[1])
    ylwCones.sort(key=lambda pt: pt[1])

    return bluCones, ylwCones, img.shape[1], img.shape[0]
