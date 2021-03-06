#!/usr/bin/env python3

import sys, time, atexit, signal, math, sysv_ipc
import numpy as np
import OD4Session
import message_set_pb2 as messages

import vision

# y-coordinate of line to check for steering direction
ySteering = 70

maxPedalPosition = 0.18

maxGroundSteering = 0.08
minGroundSteering = 0
groundSteeringMultiplier = 2
steeringOffset = 0.035 # 0 is not straight ahead
setPointXOffset = 40 # offset from the middle of the image on the x axis

# limit the pedal position linearly from max to "pedalPosition" when
# minSteer < steer < maxSteer
steerThrottle = {"minSteer": 0.6, "maxSteer": 0.7, "pedalPosition": 0.15}

lastSeen = 0
timeout = 1 # stop if no cones have been seen for this long

cid = 112


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

def distanceFromSetpoint(pts, xMid, y):
    pts = np.array(pts)
    x = np.interp(y, pts[:,1], pts[:,0])
    return round(x - xMid + setPointXOffset)

def calcPedalPosition(steer):
    steerMaxPedalPosition = maxPedalPosition
    steer = abs(steer) # don't care about the sign in this function
    if steer > steerThrottle["minSteer"]:
        if steer >= steerThrottle["maxSteer"]:
            steerMaxPedalPosition = steerThrottle["pedalPosition"]
        else:
            steerMaxPedalPosition = (maxPedalPosition -
                (maxPedalPosition - steerThrottle["pedalPosition"]) *
                ((steer - steerThrottle["minSteer"]) /
                 (steerThrottle["maxSteer"] - steerThrottle["minSteer"])))
        print('steerThrottle:', steerMaxPedalPosition)

    return steerMaxPedalPosition

def calcGroundSteering(steer):
    groundSteering = -(min(groundSteeringMultiplier * steer * maxGroundSteering, maxGroundSteering) +
            steeringOffset)
    if abs(groundSteering) < minGroundSteering:
        groundSteering = math.copysign(minGroundSteering, groundSteering)
    return groundSteering

def calcSteer(bluCones, ylwCones, xSize, ySize):
    if len(bluCones) == 0 and len(ylwCones) == 0: return None
    if len(bluCones) == 0: bluCones = [(xSize-1, 0)]
    if len(ylwCones) == 0: ylwCones = [(0, 0)]

    bluPts = addPoints(bluCones, xSize, ySize)
    ylwPts = addPoints(ylwCones, xSize, ySize)
    midPts = calcMiddleLine(bluPts, ylwPts)

    xMid = round(xSize / 2)
    dx = distanceFromSetpoint(midPts, xMid, ySteering)

    steer = dx / (xSize / 2)
    print('steer=%f' % steer)
    return steer

def stop():
    print('stopping')

    # send a short burst if some of them don't arrive
    for i in range(10):
        pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
        pedalPositionRequest.position = 0
        session.send(1086, pedalPositionRequest.SerializeToString())
        time.sleep(0.1)

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
atexit.register(stop)
session.connect()

totalTime = 0
numTime = 0
while True:
    cond.Z()

    t = time.time()

    mutex.acquire()
    shm.attach()
    buf = shm.read()
    shm.detach()
    mutex.release()

    bluCones, ylwCones, xSize, ySize = vision.findCones(buf)
    steer = calcSteer(bluCones, ylwCones, xSize, ySize)

    if steer == None:
        # stop if no cones have been since for a while
        if time.time() - lastSeen < timeout:
            continue
        print('timeout')
        pedalPosition = 0
        groundSteering = steeringOffset
    else:
        lastSeen = time.time()
        pedalPosition = calcPedalPosition(steer)
        groundSteering = calcGroundSteering(steer)

    groundSteeringRequest = messages.opendlv_proxy_GroundSteeringRequest()
    groundSteeringRequest.groundSteering = groundSteering
    session.send(1090, groundSteeringRequest.SerializeToString())

    pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
    pedalPositionRequest.position = pedalPosition
    session.send(1086, pedalPositionRequest.SerializeToString())

    totalTime += time.time() - t
    numTime += 1
    print('time:', totalTime/numTime)
