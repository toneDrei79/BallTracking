import cv2
import depthai as dai



def main():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(1920, 1080)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        print('Connected cameras:', device.getConnectedCameraFeatures())
        print('Usb speed:', device.getUsbSpeed().name)
        if device.getBootloaderVersion() is not None:
            print('Bootloader version:', device.getBootloaderVersion())
        print('Device name:', device.getDeviceName())

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

            # Retrieve 'bgr' (opencv format) frame
            cv2.imshow("rgb", inRgb.getCvFrame())

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    main()