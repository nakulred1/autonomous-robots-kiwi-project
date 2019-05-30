#!/usr/bin/env python3

import sys, time, atexit, signal, math, sysv_ipc

imgPath = "/tmp/img.argb"
keySharedMemory = sysv_ipc.ftok(imgPath, 1, True)
keySemMutex = sysv_ipc.ftok(imgPath, 2, True)
keySemCondition = sysv_ipc.ftok(imgPath, 3, True)
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemMutex)
cond = sysv_ipc.Semaphore(keySemCondition)

prevTime = time.time()
while True:
    cond.Z()
    
    print(time.time() - prevTime)
    prevTime = time.time()

    mutex.acquire()
    shm.attach()
    buf = shm.read()
    shm.detach()
    mutex.release()
