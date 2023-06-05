import os
import csv
import json
import depthai as dai
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='./Data/out3/saved_model', help='path of the object detection model')
    parser.add_argument('--datadir', type=str, default='./calibdata/00/', help='path of the calibration data')
    parser.add_argument('--capture', action='store_true', help='strat from capturing images')
    parser.add_argument('--confidence', type=float, default=0.5, help='detection confidence')
    parser.add_argument('--unitwidth', type=float, default=1.2, help='width of view away 1m from camera')
    parser.add_argument('--unitheight', type=float, default=0.8, help='height of view away 1m from camera')
    parser.add_argument('--resolution_x', type=int, default=1920, help='x resolution')
    parser.add_argument('--resolution_y', type=int, default=1080, help='y resolution')
    return parser.parse_args()


def capture(args, distances=[0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]):
    os.makedirs(args.datadir, exist_ok=True)

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(args.resolution_x, args.resolution_y)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        def displayFrame(name, frame):
            cv2.imshow(name, frame)
        
        def save(frame, dis):
            path = os.path.join(args.datadir, f'images/{int(dis*10):02d}.png')
            cv2.imwrite(path, frame)
            with open(os.path.join(args.datadir, 'data.csv'), 'a') as f:
                f.write(f'{path} {dis:.3f}\n')
            print(f'captured {dis:.3f} m')

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        start_time = time.time()
        i = 0
        while i < len(distances):
            inRgb = qRgb.tryGet()
            if inRgb is None:
                continue
            frame = inRgb.getCvFrame()
            displayFrame("rgb", frame)

            elapsed_time = int(time.time() - start_time)
            if elapsed_time > 5:
                os.makedirs(os.path.join(args.datadir, 'images/'), exist_ok=True)
                save(frame, distances[i])
                start_time = time.time()
                i += 1
            
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

    
# Global Variables
modelFlag = 0
detection_model = None

def loadModel(path):
    global modelFlag, detection_model
    if not modelFlag:
        detection_model = tf.saved_model.load(path)
        print("Model Loaded")
        modelFlag = 1


def detect(image):
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = image[:, :, ::-1]

    # Add a new axis to match the input size requirement of the model
    rgb_frame_expanded = np.expand_dims(rgb_frame, axis=0)

    # Run the frame through the model
    output_dict = detection_model(rgb_frame_expanded)

    # Detection scores are the detection confidence
    detection_scores = output_dict['detection_scores'][0].numpy()

    # Detection boxes are the coordinates of the detected object
    detection_boxes = output_dict['detection_boxes'][0].numpy()

    idxmax = np.argmax(detection_scores)
    score = detection_scores[idxmax]
    if score > args.confidence:
        box = detection_boxes[idxmax] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        radius = max((box[3] - box[1]) / 2, (box[2] - box[0]) / 2)

        return radius

    return -1


def least_squares(X, Y):
    '''
    Assumption:
        distance = a/size + b
    
    denote y = a/_x + b, x = 1/_x
    derive optimal a and b
    '''

    # averages
    X_ = np.mean([1/x for x in X])
    Y_ = np.mean([y for y in Y])
    X2_ = np.mean([(1/x)*(1/x) for x in X])
    XY_ = np.mean([y/x for x,y in zip(X,Y)])

    a = (len(X)*XY_ - X_*Y_) / (len(X)*X2_ - X_*X_)
    b = (X2_*Y_ - X_*XY_) / (len(X)*X2_ - X_*X_)

    return a, b


def main(args):
    if args.capture:
        capture(args)
    
    loadModel(args.modelpath)

    images = []
    distances = []
    with open(os.path.join(args.datadir, 'data.csv'), 'r') as f:
        _data = csv.reader(f, delimiter=' ')
        for row in _data:
            images.append(cv2.imread(row[0]))
            distances.append(float(row[1]))
    
    X = []
    Y = []
    for i, image in enumerate(images):
        print(i, end=' ')
        radius = detect(image)
        resolution = image.shape[:2]
        if radius > 0:
            size = (radius*2) / resolution[0] # != diameter
            print('deteted')
            X.append(size)
            Y.append(distances[i])
        else:
            print('not detected')
    
    fig = plt.figure()
    plt.plot(X, Y)
    for x,y in zip(X,Y):
        print(f'size:{x:.3f}, distance:{y:.3f}')

    a, b = least_squares(X, Y)
    print(a, b)

    sampleX = np.arange(0.05,1,0.01)
    sampleY = np.array([a/x + b for x in sampleX])
    plt.plot(sampleX, sampleY)
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(args.datadir, 'graph.png'))

    with open(os.path.join(args.datadir, 'parameters.json'), 'w') as f:
        _data = {'a': a, 'b': b,
                 'width': args.unitwidth, 'height': args.unitheight,
                 'resolution_x': args.resolution_x, 'resolution_y': args.resolution_y}
        json.dump(_data, f)


if __name__ == '__main__':
    args = get_args()
    main(args)