#!/usr/bin/env python3

import numpy as np
import cv2

img = np.load("img.npy")
img = img[200:480, 0:640]

blur = cv2.GaussianBlur(img, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
pts = np.array(((0, 280), (0, 170), (200, 130), (420, 135), (640, 190),
    (640, 280), (0, 280))).astype(np.int32)
cv2.fillPoly(gray, [pts], 0)

grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

img_x = cv2.convertScaleAbs(grad_x)
img = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5,
    cv2.convertScaleAbs(grad_y), 0.5, 0)

cv2.imshow("img", img)
cv2.moveWindow("img", 0, 0)
cv2.imshow("img_x", img_x)
cv2.moveWindow("img_x", 660, 0)
cv2.waitKey(0)

blur = cv2.boxFilter(img_x, -1, (18, 18))
blur = cv2.inRange(blur, 15, 255)
cv2.imshow("blur", blur)
cv2.moveWindow("blur", 0, 0)
cv2.waitKey(0)
