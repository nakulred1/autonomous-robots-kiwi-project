#!/usr/bin/env python3

import sysv_ipc
import numpy as np
import cv2
import linefuncs as f

imgPath = "/tmp/img.argb"
keySharedMemory = sysv_ipc.ftok(imgPath, 1, True)
keySemMutex = sysv_ipc.ftok(imgPath, 2, True)
keySemCondition = sysv_ipc.ftok(imgPath, 3, True)
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemMutex)
cond = sysv_ipc.Semaphore(keySemCondition)

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=60, detectShadows=False)

while True:
    cond.Z()

    mutex.acquire()
    shm.attach()
    buf = shm.read()
    shm.detach()
    mutex.release()

    img = np.frombuffer(buf, np.uint8).reshape(480, 640, 4)
    img = img[200:380, 140:640] # remove the top, bottom and left parts of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    motion = fgbg.apply(blur)

    cv2.imshow("motion", motion)
    cv2.waitKey(1)
