#!/usr/bin/env python3

import numpy as np
import cv2

def cross(img, pt, color):
    cv2.line(img, (pt[0]-5,pt[1]+5), (pt[0]+5,pt[1]-5), color)
    cv2.line(img, (pt[0]+5,pt[1]+5), (pt[0]-5,pt[1]-5), color)

def findCones(img, hsvRanges):
    cones = None
    for i in range(len(hsvRanges)):
        inRange = cv2.inRange(img, hsvRanges[i][0], hsvRanges[i][1])
        if i == 0: cones = inRange
        else:      cones = cv2.bitwise_or(cones, inRange)

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(cones, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)

    try:
        _, contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    conePos = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        conePos.append((x + int(w/2), y + h))
    conePos.sort(key=lambda pt: pt[1])
    return conePos

def addPoints(cones, w, h):
    pts = cones.copy()

    # add a vertical line at the bottom
    pts.append((pts[-1][0], h))

    # extend first line to the top edge (y=0)
    if pts[1][1] - pts[0][1] == 0:
        pts[0] = (pts[1][0], 0) # line has dy/dx=0; just draw a line to y=0
    else:
        slope = (pts[1][0] - pts[0][0]) / (pts[1][1] - pts[0][1])
        dx = slope * (-pts[0][1])
        pts[0] = (round(pts[0][0] + dx), 0)

    return pts

def calcMiddleLine(bluPts, ylwPts):
    if len(bluPts) == 0 or len(ylwPts) == 0: return []

    bluPts = np.array(bluPts)
    ylwPts = np.array(ylwPts)

    # calculate middle line points from blue points and yellow lines
    bluCorrPts = np.interp(bluPts[:,1], ylwPts[:,1], ylwPts[:,0])
    x1 = np.round((bluCorrPts + bluPts[:,0]) / 2).astype(int)
    pts1 = np.vstack((x1, bluPts[:,1])).T.tolist()

    # calculate middle line points from yellow points and blue lines
    ylwCorrPts = np.interp(ylwPts[:,1], bluPts[:,1], bluPts[:,0])
    x2 = np.round((ylwCorrPts + ylwPts[:,0]) / 2).astype(int)
    pts2 = np.vstack((x2, ylwPts[:,1])).T.tolist()

    pts = [tuple(pt) for pt in (pts1 + pts2)]
    pts = list(set(pts)) # remove duplicates
    pts.sort(key=lambda pt: pt[1])

    return pts

def distanceFromMiddle(pts, xMid, y):
    pts = np.array(pts)
    x = np.interp(y, pts[:,1], pts[:,0])
    if type(x) == np.float64: x = x.item()
    return round(x - xMid)

minOrgConesDx = 10
def classifyOrgCones(orgCones, bluCones, ylwCones):
    # assume that the largest "hole" in the x-direction between orange cones
    # is the lane and add cones to the left to the yellow cones and cones to
    # the right to the blue
    orgCones.sort(key=lambda pt: pt[0])
    maxDx = 0
    iMaxDx = 0
    for i in range(1, len(orgCones)):
        dx = orgCones[i][0] - orgCones[i-1][0]
        if dx > maxDx:
            maxDx = dx
            iMaxDx = i

    if maxDx > minOrgConesDx:
        bluCones.extend(orgCones[iMaxDx:])
        bluCones.sort(key=lambda pt: pt[1])
        ylwCones.extend(orgCones[0:iMaxDx])
        ylwCones.sort(key=lambda pt: pt[1])
