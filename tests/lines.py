#!/usr/bin/env python3

import sysv_ipc
import numpy as np
import cv2
import linefuncs as f

# range for identifying blue cones in HSV
bluLow = (95, 90, 20)
bluHigh = (130, 255, 255)
count = 0
# range for identifying yellow cones in HSV
ylwLow = (23, 60, 100)
ylwHigh = (28, 255, 255)

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
    img = img[200:480, 0:640]  # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
                    (640, 280), (0, 280))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0))  # black out the car

    bluCones = f.findCones(hsv, bluLow, bluHigh)
    ylwCones = f.findCones(hsv, ylwLow, ylwHigh)
    for cone in bluCones:
        f.cross(img, cone, (255, 0, 0))
    for cone in ylwCones:
        f.cross(img, cone, (0, 255, 255))

    if len(bluCones) == 0: bluCones = [(0, 0)]
    if len(ylwCones) == 0: ylwCones = [(img.shape[1] - 1, 0)]

    bluPts = f.addPoints(bluCones, img.shape[1], img.shape[0])
    ylwPts = f.addPoints(ylwCones, img.shape[1], img.shape[0])
    midPts = f.calcMiddleLine(bluPts, ylwPts)
    for i in range(1, len(bluPts)):
        cv2.line(img, bluPts[i - 1], bluPts[i], (255, 0, 0))
    for i in range(1, len(ylwPts)):
        cv2.line(img, ylwPts[i - 1], ylwPts[i], (0, 255, 255))
    for i in range(1, len(midPts)):
        cv2.line(img, midPts[i - 1], midPts[i], (0, 255, 0))

    xMid = round(img.shape[1] / 2)
    dx = f.distanceFromMiddle(midPts, round(img.shape[1]) / 2, ySteering)
    cv2.arrowedLine(img, (xMid, ySteering), (xMid + dx, ySteering),
                    (0, 0, 255))

    #Object detection using haar cascading
    face_cascade = cv2.CascadeClassifier('cascade.xml')

    def detect_face(img):
        face_img = img.copy()

        face_rects = face_cascade.detectMultiScale(face_img)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255),
                          10)
        return face_img

    img = detect_face(img)

    cv2.imshow("img", img)

    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
