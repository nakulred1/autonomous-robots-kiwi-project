#!/usr/bin/env python3

import sys, time, atexit, signal, math
import numpy as np
import OD4Session
import message_set_pb2 as messages


# y-coordinate of line to check for steering direction
ySteering = 100

velocity = 0.1 # constant car velocity
maxGroundSteering = 0.1
minGroundSteering = 0.02
steeringOffset = 0.07 # 0 is not straight ahead

stoppingDistance = 0.1 # stop the car when front distance is less than this

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


lastConeMsg = time.time()
noCones = False
steer = 0
def onCones(msg, senderStamp, timeStamps):
    global lastConeMsg
    global noCones
    global steer

    xSize = msg.xSize
    ySize = msg.ySize
    bluCones = np.frombuffer(msg.blueCones, dtype='uint16').reshape(-1, 2)
    bluCones = [tuple(pt) for pt in bluCones.tolist()]
    ylwCones = np.frombuffer(msg.yellowCones, dtype='uint16').reshape(-1, 2)
    ylwCones = [tuple(pt) for pt in ylwCones.tolist()]

    if len(bluCones) == 0 and len(ylwCones) == 0:
        # Stop the car if it can't see any cones
        noCones = True
        return
    if len(bluCones) == 0: bluCones = [(xSize-1, 0)]
    if len(ylwCones) == 0: ylwCones = [(0, 0)]

    bluPts = addPoints(bluCones, xSize, ySize)
    ylwPts = addPoints(ylwCones, xSize, ySize)
    midPts = calcMiddleLine(bluPts, ylwPts)

    xMid = round(xSize / 2)
    dx = distanceFromMiddle(midPts, xMid, ySteering)

    lastConeMsg = time.time()
    noCones = False
    steer = dx / (xSize / 2)
    groundSteering = steer * maxGroundSteering
    print('steer=%f' % steer)

distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };
def onDistance(msg, senderStamp, timeStamps):
    if senderStamp == 0:
        print("frontDistance=%f" % msg.distance)
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

session = OD4Session.OD4Session(cid)
session.registerMessageCallback(1500, onCones, messages.tme290_Cones)
session.registerMessageCallback(1039, onDistance,
        messages.opendlv_proxy_DistanceReading)
atexit.register(stop)
session.connect()

while True:
    # Stop the car if we haven't received any cone messages for a long time, we
    # don't see any cones, or something is close in front
    if time.time() - lastConeMsg > 1 or noCones or distances["front"] < stoppingDistance:
        pedalPosition = 0
        groundSteering = 0
    else:
        pedalPosition = velocity
        groundSteering = -(min(2.5 * steer * maxGroundSteering,
            maxGroundSteering) + steeringOffset)
        if abs(groundSteering) < minGroundSteering:
            groundSteering = math.copysign(minGroundSteering, groundSteering)

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
