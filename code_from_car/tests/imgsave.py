#!/usr/bin/env python3
# Saves the next image from h264-decoder.
# Run the recording to the desired frame, pause, start this script, and then
# resume the recording to save the desired frame.

import sysv_ipc
import numpy
import cv2

name = "/tmp/img.argb"
keySharedMemory = sysv_ipc.ftok(name, 1, True)
keySemMutex = sysv_ipc.ftok(name, 2, True)
keySemCondition = sysv_ipc.ftok(name, 3, True)
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemMutex)
cond = sysv_ipc.Semaphore(keySemCondition)

cond.Z()

mutex.acquire()
shm.attach()
buf = shm.read()
shm.detach()
mutex.release()

img = numpy.frombuffer(buf, numpy.uint8).reshape(480, 640, 4)
numpy.save("img.npy", img)
