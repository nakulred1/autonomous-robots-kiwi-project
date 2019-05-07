#!/usr/bin/env python3

import sys, time, atexit, signal, math, sysv_ipc
import numpy as np
import OD4Session
import message_set_pb2 as messages

import vision


# y-coordinate of line to check for steering direction
ySteering = 100

maxPedalPosition = 0.16
pedalPositionThreshold = 0.08 # pedal position at which the car stops moving

maxGroundSteering = 0.08
minGroundSteering = 0.01
groundSteeringMultiplier = 2.5
steeringOffset = 0.04 # 0 is not straight ahead

# limit the pedal position linearly from max to the threshold when
# min < frontDistance < max
throttleDistances = {"max": 0, "min": 0}

# limit the pedal position linearly from max to "pedalPosition" when
# minSteer < steer < maxSteer
steerThrottle = {"minSteer": 0.5, "maxSteer": 0.7, "pedalPosition": 0.14}

msgTimeout = 1 # stop if no cone messages have been received for this long

cid = 112
freq = 10


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
    return round(x - xMid)

def calcPedalPosition(steer):
    distMaxPedalPosition = maxPedalPosition
    if distances["front"] < throttleDistances["max"]:
        distMaxPedalPosition = (maxPedalPosition -
            (maxPedalPosition - pedalPositionThreshold) *
            ((throttleDistances["max"] - distances["front"]) /
             (throttleDistances["max"] - throttleDistances["min"])))
        print('distanceThrottle:', distMaxPedalPosition)

    steerMaxPedalPosition = maxPedalPosition
    steer = abs(steer) # don't care about the sign in this function
    if steer > steerThrottle["minSteer"]:
        if steer >= steerThrottle["maxSteer"]:
            steerMaxPedalPosition = steerThrottle["pedalPosition"]
        else:
            steerPedalPosition = (maxPedalPosition -
                (maxPedalPosition - steerThrottle["pedalPosition"]) *
                ((steer - steerThrottle["minSteer"]) /
                 (steerThrottle["maxSteer"] - steerThrottle["minSteer"])))
        print('steerThrottle:', steerPedalPosition)

    return min(distMaxPedalPosition, steerMaxPedalPosition)

def calcGroundSteering(steer):
    groundSteering = -(min(groundSteeringMultiplier * steer * maxGroundSteering, maxGroundSteering) +
            steeringOffset)
    if abs(groundSteering) < minGroundSteering:
        groundSteering = math.copysign(minGroundSteering, groundSteering)
    return groundSteering

def calcSteer(bluCones, ylwCones, xSize, ySize):
    # Treat a message with no cones as no message at all
    if len(bluCones) == 0 and len(ylwCones) == 0: return
    if len(bluCones) == 0: bluCones = [(xSize-1, 0)]
    if len(ylwCones) == 0: ylwCones = [(0, 0)]

    bluPts = addPoints(bluCones, xSize, ySize)
    ylwPts = addPoints(ylwCones, xSize, ySize)
    midPts = calcMiddleLine(bluPts, ylwPts)

    xMid = round(xSize / 2)
    dx = distanceFromMiddle(midPts, xMid, ySteering)

    steer = dx / (xSize / 2)
    groundSteering = steer * maxGroundSteering
    print('steer=%f' % steer)
    return steer


distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };
def onDistance(msg, senderStamp, timeStamps):
    if senderStamp == 0:
        distances["front"] = msg.distance
    elif senderStamp == 1:
        distances["left"] = msg.distance
    elif senderStamp == 2:
        distances["rear"] = msg.distance
    elif senderStamp == 3:
        distances["right"] = msg.distance

def stop():
    print('stopping')

    # send a short burst if some of them don't arrive
    for i in range(10):
        pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
        pedalPositionRequest.position = 0
        session.send(1086, pedalPositionRequest.SerializeToString())
        time.sleep(1/freq)

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
session.registerMessageCallback(1039, onDistance,
        messages.opendlv_proxy_DistanceReading)
atexit.register(stop)
session.connect()

while True:
    cond.Z()

    mutex.acquire()
    shm.attach()
    buf = shm.read()
    shm.detach()
    mutex.release()

    bluCones, ylwCones, xSize, ySize = vision.findCones(buf)
    steer = calcSteer(bluCones, ylwCones, xSize, ySize)

    pedalPosition = calcPedalPosition(steer)
    groundSteering = calcGroundSteering(steer)

    groundSteeringRequest = messages.opendlv_proxy_GroundSteeringRequest()
    groundSteeringRequest.groundSteering = groundSteering
    session.send(1090, groundSteeringRequest.SerializeToString())

    # don't know why, but if this is not here the second message often gets
    # dropped...
    time.sleep(0.001)

    pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
    pedalPositionRequest.position = pedalPosition
    session.send(1086, pedalPositionRequest.SerializeToString())

    time.sleep(1/freq)
