#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(1920, 1080)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    def displayFrame(name, frame):
        cv2.imshow(name, frame)
    
    def capture(frame, i):
        cv2.imwrite(f'images/{i:03}.png', frame)
    
    start_time = time.time()
    i = 0
    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if frame is not None:
            displayFrame("rgb", frame)

        elapsed_time = int(time.time() - start_time)
        if elapsed_time > 10:
            capture(frame, i)
            print('captured')
            start_time = time.time()
            i += 1
            # break
        
        if cv2.waitKey(1) == ord('q'):
            break