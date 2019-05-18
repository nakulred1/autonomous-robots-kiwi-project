#!/usr/bin/env python3

import sysv_ipc
import numpy as np
import cv2
import linefuncs as f
import random as rng
rng.seed(12345)

# range for identifying blue cones in HSV
bluRanges = [
    [(97, 78, 35), (130, 255, 100)], # regular cone blue
    [(112, 30, 30), (150, 80, 70)]  # more of a dark gray
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
    img = img[200:480, 0:640] # remove the top of the image

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
        (640, 280), (0, 280))).astype(np.int32)
    cv2.fillPoly(hsv, [pts], (255, 255, 255)) # black out the car

    #find the black part, the range can be calibrated in the future
    inRange = cv2.inRange(hsv, (0,0,0), (30,30,30))
    cv2.imshow("inRange", inRange)

    kernel = np.ones((3, 3), np.uint8)
    #erode = cv2.erode(inRange, kernel, iterations=1)
    #cv2.imshow("erode", erode)
    dilate = cv2.dilate(inRange, kernel, iterations=12)
    # cv2.imshow("dilate", dilate)

    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(dilate, contours, -1, 255, 3)

        
        #c = max(contours, key = cv2.contourArea)

        #find the biggest area
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        print(cv2.contourArea(biggest_contour))

        #filter out the false positie
        if cv2.contourArea(biggest_contour) > 750: #2800:

            x,y,w,h = cv2.boundingRect(biggest_contour)
            mu = cv2.moments(biggest_contour)
             #get the mass center
            mc = (mu['m10']  / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))
            # draw 
            cv2.rectangle(img,(x,y),(x+w,y+h),(10,255,250),2)
            cv2.circle(img, (int(mc[0]), int(mc[1])), 4, (10,255,250), -1)

            ## I guess dy/dx and current speed can show us whether the target car is running or parking
            print("x:",(int(mc[0])))
            print("y:",(int(mc[1])))




    cv2.imshow("dilate", img)


    cv2.waitKey(1)
