#!/usr/bin/env python3

import sys, time, atexit, signal, math, sysv_ipc
import numpy as np
import OD4Session
import message_set_pb2 as messages

import vision, motion


# y-coordinate of line to check for steering direction
ySteering = 100

maxPedalPosition = 0.16

maxGroundSteering = 0.08
minGroundSteering = 0.01
groundSteeringMultiplier = 2.5
steeringOffset = 0.04 # 0 is not straight ahead

# limit the pedal position linearly from max to threshold (at which the car
# stops moving) when min < frontDistance < max
distanceThrottle = {"max": 0, "min": 0, "threshold": 0.08}

# limit the pedal position linearly from max to "pedalPosition" when
# minSteer < steer < maxSteer
#steerThrottle = {"minSteer": 0.5, "maxSteer": 0.7, "pedalPosition": 0.14}
steerThrottle = {"minSteer": 1, "maxSteer": 1, "pedalPosition": 0.14}

lastSeen = 0
timeout = 1 # stop if no cones have been seen for this long

state = "drive"

# used by the waitAtIntersection state
lastMotion = None
waitTimeout = 2
trafficFromRight = True # has to be changed depending on where the car starts


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

def distanceFromMiddle(pts, xMid, y):
    pts = np.array(pts)
    x = np.interp(y, pts[:,1], pts[:,0])
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

def calcPedalPosition(steer):
    distMaxPedalPosition = maxPedalPosition
    if distances["front"] < distanceThrottle["max"]:
        distMaxPedalPosition = (maxPedalPosition -
            (maxPedalPosition - distanceThrottle["threshold"]) *
            ((distanceThrottle["max"] - distances["front"]) /
             (distanceThrottle["max"] - distanceThrottle["min"])))
        print('distanceThrottle:', distMaxPedalPosition)

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

    return min(distMaxPedalPosition, steerMaxPedalPosition)

def calcGroundSteering(steer):
    groundSteering = -(min(groundSteeringMultiplier * steer * maxGroundSteering, maxGroundSteering) +
            steeringOffset)
    if abs(groundSteering) < minGroundSteering:
        groundSteering = math.copysign(minGroundSteering, groundSteering)
    return groundSteering

def calcSteer(bluCones, ylwCones, xSize, ySize):
    if len(bluCones) == 0 and len(ylwCones) == 0: return
    if len(bluCones) == 0: bluCones = [(xSize-1, 0)]
    if len(ylwCones) == 0: ylwCones = [(0, 0)]

    bluPts = addPoints(bluCones, xSize, ySize)
    ylwPts = addPoints(ylwCones, xSize, ySize)
    midPts = calcMiddleLine(bluPts, ylwPts)

    xMid = round(xSize / 2)
    dx = distanceFromMiddle(midPts, xMid, ySteering)

    steer = dx / (xSize / 2)
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

    img = np.frombuffer(buf, np.uint8).reshape(480, 640, 4)

    bluCones, ylwCones, orgCones, xSize, ySize = vision.findCones(img)
    if len(orgCones) > 2:
        classifyOrgCones(orgCones, bluCones, ylwCones)

    steer = calcSteer(bluCones, ylwCones, xSize, ySize)

    if state == "drive":
        if steer == None:
            if time.time() - lastSeen < timeout:
                continue
            print('timeout')
            pedalPosition = 0
            groundSteering = steeringOffset
        else:
            lastSeen = time.time()
            pedalPosition = calcPedalPosition(steer)
            groundSteering = calcGroundSteering(steer)

        if len(orgCones) > 0:
            if trafficFromRight:
                lastMotion = time.time()
                print('drive -> waitAtIntersection')
                state = "waitAtIntersection"
            else:
                print('drive -> continueAfterIntersection')
                state = "continueAfterIntersection"

    if state == "waitAtIntersection":
        pedalPosition = 0
        groundSteering = steeringOffset

        if motion.checkForMotion(img):
            lastMotion = time.time()

        if time.time() - lastMotion > waitTimeout:
            print('waitAtIntersection -> continueAfterIntersection')
            state = "continueAfterIntersection"

    if state == "continueAfterIntersection":
        pedalPosition = calcPedalPosition(steer)
        groundSteering - calcGroundSteering(steer)

        if len(orgCones) == 0:
            # next intersection will be the reverse from this one
            trafficFromRight = !trafficFromRight

            print('continueAfterIntersection -> drive')
            state = "drive"

    groundSteeringRequest = messages.opendlv_proxy_GroundSteeringRequest()
    groundSteeringRequest.groundSteering = groundSteering
    session.send(1090, groundSteeringRequest.SerializeToString())

    # don't know why, but if this is not here the second message often gets
    # dropped...
    time.sleep(0.001)

    pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
    pedalPositionRequest.position = pedalPosition
    session.send(1086, pedalPositionRequest.SerializeToString())
