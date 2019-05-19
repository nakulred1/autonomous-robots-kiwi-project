#!/usr/bin/env python3

import sysv_ipc
import numpy as np
import cv2
import linefuncs as f

# range for identifying blue cones in HSV
bluRanges = [[(97, 78, 35), (130, 255, 100)]]

bluCRanges = [
    [(94, 50, 35), (130, 255, 100)],
    [(70, 30, 25), (98, 110, 80)],
    [(100, 12, 30), (180, 80, 130)]
]

# range for identifying yellow cones in HSV
ylwRanges = [
    [(23, 60, 140), (32, 255, 255)]
]

orgRanges = [
    [(0, 80, 110), (8, 180, 200)]
]

# y-coordinate of line to check for steering direction
ySteering = 100


imgPath = "/tmp/img.argb"
keySharedMemory = sysv_ipc.ftok(imgPath, 1, True)
keySemMutex = sysv_ipc.ftok(imgPath, 2, True)
keySemCondition = sysv_ipc.ftok(imgPath, 3, True)
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemMutex)
cond = sysv_ipc.Semaphore(keySemCondition)

while True:
    cond.Z()

    mutex.acquire()
    shm.attach()
    buf = shm.read()
    shm.detach()
    mutex.release()

    img = np.frombuffer(buf, np.uint8).reshape(480, 640, 4)
    img = img[200:390, 0:640] # remove the top and bottom of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0))

    pts = np.array(((0, 190), (0, 170), (200, 130), (420, 135), (640, 190),
        (0, 190))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0)) # black out the car
    cv2.fillPoly(grad, [pts], 0)

    grad = cv2.boxFilter(grad, -1, (18, 18))
    contrastFilter = cv2.inRange(grad, 18, 255)

    bluCones = f.findCones(hsv, bluRanges, contrastFilter=contrastFilter,
        hsvCRanges=bluCRanges)
    ylwCones = f.findCones(hsv, ylwRanges)
    orgCones = f.findCones(hsv, orgRanges)
    for cone in bluCones:
        f.cross(img, cone, (255, 0, 0))
    for cone in ylwCones:
        f.cross(img, cone, (0, 255, 255))
    for cone in orgCones:
        f.cross(img, cone, (0, 140, 255))

    if len(orgCones) > 0:
        f.classifyOrgCones(orgCones, bluCones, ylwCones)

    if len(bluCones) == 0: bluCones = [(img.shape[1]-1, 0)]
    if len(ylwCones) == 0: ylwCones = [(0, 0)]

    bluPts = f.addPoints(bluCones, img.shape[1], img.shape[0])
    ylwPts = f.addPoints(ylwCones, img.shape[1], img.shape[0])
    midPts = f.calcMiddleLine(bluPts, ylwPts)
    for i in range(1, len(bluPts)):
        cv2.line(img, bluPts[i-1], bluPts[i], (255, 0, 0))
    for i in range(1, len(ylwPts)):
        cv2.line(img, ylwPts[i-1], ylwPts[i], (0, 255, 255))
    for i in range(1, len(midPts)):
        cv2.line(img, midPts[i-1], midPts[i], (0, 255, 0))

    xMid = round(img.shape[1]/2)
    dx = f.distanceFromMiddle(midPts, round(img.shape[1])/2, ySteering)
    cv2.arrowedLine(img, (xMid, ySteering), (xMid + dx, ySteering), (0, 0, 255))

    cv2.imshow("img", img)
    cv2.waitKey(1)
