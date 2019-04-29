#!/usr/bin/env python3
# Simple script to set a sequence of pedalPositions and groundSteering angles.
# Useful for experimenting with different speeds and steering angles. Always
# stops the car at the end.

import sys, getopt, time, atexit
import OD4Session
import message_set_pb2 as messages

def usage():
    print('usage:', sys.argv[0], '--cid=<cid> [--freq=<freq>] [commands...]')
    print('command syntax: <pedalPosition>,<groundSteering>,<duration>')
    sys.exit(1)

def stop():
    print('stopping')

    # send a short burst if some of them don't arrive
    for i in range(10):
        pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
        pedalPositionRequest.position = 0
        session.send(1086, pedalPositionRequest.SerializeToString())
        time.sleep(timeStep)

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h', ['cid=', 'freq='])
except getopt.GetoptError:
    usage()

cid = None
freq = 10
for opt, arg in opts:
    if opt == 'h':
        usage()
    elif opt == '--cid':
        cid = int(arg)
    elif opt == '--freq':
        freq = int(arg)
if cid == None: usage()

session = OD4Session.OD4Session(cid=cid)
session.connect()

steps = []
for arg in args:
    try:
        pedalPosition, groundSteering, duration = [float(item) for item in 
                arg.split(',')]
    except ValueError:
        print('invalid syntax:', arg)
        print('command syntax is: <pedalPosition>,<groundSteering>,<duration>')
        sys.exit(1)

    if pedalPosition < -1 or pedalPosition > 0.25:
        print('allowed pedalPosition range: [-1.0, 0.25]')
        sys.exit(1)

    if groundSteering < -0.66 or groundSteering > 0.66:
        print('allowed groundSteering range: [-0.66, 0.66]')
        sys.exit(1)

    steps.append({'pedalPosition': pedalPosition,
        'groundSteering': groundSteering,
        'duration': duration})

timeStep = 1/freq;
runtime = 0
i = 0

atexit.register(stop)

if len(steps) > 0:
    print('pedalPosition:', steps[0]['pedalPosition'], '|', 'groundSteering:',
            steps[0]['groundSteering'])
while i < len(steps):
    step = steps[i]

    groundSteeringRequest = messages.opendlv_proxy_GroundSteeringRequest()
    groundSteeringRequest.groundSteering = step['groundSteering']
    session.send(1090, groundSteeringRequest.SerializeToString())

    # don't know why, but if this is not here the second message often gets
    # dropped...
    time.sleep(0.001)

    pedalPositionRequest = messages.opendlv_proxy_PedalPositionRequest()
    pedalPositionRequest.position = step['pedalPosition']
    session.send(1086, pedalPositionRequest.SerializeToString())

    time.sleep(timeStep)
    runtime += timeStep
    if runtime >= step['duration']:
        runtime = 0
        i += 1
        if i < len(steps):
            print('pedalPosition:', steps[i]['pedalPosition'], '|',
                    'groundSteering:', steps[i]['groundSteering'])
