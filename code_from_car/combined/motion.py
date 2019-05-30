#!/usr/bin/env python3

import numpy as np
import cv2

threshold = 10 # filter out some noise

bgSub = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=60, detectShadows=False)

def checkForMotion(img):
    img = img[200:380, 140:640] # remove the top, bottom and left parts
    motion = bgSub.apply(img)
    numMovingPixels = np.sum(motion)
    return numMovingPixels > threshold*255 # True = motion detected
