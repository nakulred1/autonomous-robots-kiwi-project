#!/usr/bin/env python3

import sys, signal
import sysv_ipc
import numpy as np
import cv2
import OD4Session
import message_set_pb2 as messages

# range for identifying blue cones in HSV
bluLow = (95, 90, 20)
bluHigh = (130, 255, 255)

# range for identifying yellow cones in HSV
ylwLow = (23, 60, 100)
ylwHigh = (28, 255, 255)

cid = 112


def findCones(img, hsvLow, hsvHigh):
    cones = cv2.inRange(img, hsvLow, hsvHigh)

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(cones, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)

    _, contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    conePos = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        conePos.append((x + int(w/2), y + h))
    conePos.sort(key=lambda pt: pt[1])
    return conePos


def onSigterm():
    sys.exit(0)
signal.signal(signal.SIGTERM, onSigterm)


imgPath = "/tmp/img.argb"
keySharedMemory = sysv_ipc.ftok(imgPath, 1, True)
keySemMutex = sysv_ipc.ftok(imgPath, 2, True)
keySemCondition = sysv_ipc.ftok(imgPath, 3, True)
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemMutex)
cond = sysv_ipc.Semaphore(keySemCondition)

session = OD4Session.OD4Session(cid)
session.connect()

while True:
    cond.Z()

    mutex.acquire()
    shm.attach()
    buf = shm.read()
    shm.detach()
    mutex.release()

    img = np.frombuffer(buf, np.uint8).reshape(480, 640, 4)
    img = img[200:480, 0:640] # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
        (640, 280), (0, 280))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (0, 0, 0)) # black out the car

    bluCones = findCones(hsv, bluLow, bluHigh)
    ylwCones = findCones(hsv, ylwLow, ylwHigh)
    print(bluCones, ylwCones)

    conesMsg = messages.tme290_Cones()
    conesMsg.xSize = img.shape[1]
    conesMsg.ySize = img.shape[0]
    conesMsg.blueCones = np.array(bluCones, dtype='uint16').tobytes()
    conesMsg.yellowCones = np.array(ylwCones, dtype='uint16').tobytes()
    session.send(1500, conesMsg.SerializeToString())
