import json
import cv2
import tensorflow as tf
import numpy as np
import depthai as dai
import argparse
import socket



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/Users/tone/Documents/project/Ball/Assets/coordinates.csv', help='path of the coordinate.csv')
    parser.add_argument('--preview', action='store_true', help='preview video')
    parser.add_argument('--calibdata', type=str, default='./calibdata/00/parameters.json', help='path of the calibration data')
    parser.add_argument('--confidence', type=float, default=0.5, help='detection confidence')
    return parser.parse_args()


def send_data(position, ip='127.0.0.1', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = f'{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}'
    sock.sendto(message.encode(), (ip, port))
    sock.close()


# Global Variables
modelFlag = 0
detection_model = None

def loadModel():
    global modelFlag, detection_model
    if not modelFlag:
        detection_model = tf.saved_model.load('Data/out3/saved_model')
        print("Model Loaded")
        modelFlag = 1

def perspective(_position, params):
    z = _position[2]
    x = _position[0] * z * params['width']
    y = _position[1] * z * params['height']
    position = (x, y, z)
    return position

def estimate_position(_x, _y, _rad, params):
    x = -0.5 + _x / params['resolution_x']
    y = 0.5 - _y / params['resolution_y']
    size = (_rad*2) / params['resolution_y'] # != diameter

    z = params['a'] / size + params['b']
    x, y, z = perspective((x,y,z), params)
    print(f'{x:.3f} {y:.3f} {z:.3f}')

    return x, y, z


def main(args):
    with open(args.calibdata, 'r') as f:
        params = json.load(f)


    loadModel()

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutVideo.setStreamName("video")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(params['resolution_x'], params['resolution_y'])
    camRgb.setFps(60)
    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    # Linking
    camRgb.video.link(xoutVideo.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        
        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

        while True:

            videoIn = video.get()

            # Get BGR frame from NV12 encoded video frame to show with opencv
            # Visualizing the frame on slower hosts might have overhead
            frame = videoIn.getCvFrame()

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_frame = frame[:, :, ::-1]

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
                box = detection_boxes[idxmax] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])

                # Get the center of the box
                center_x = (box[1] + box[3]) / 2
                center_y = (box[0] + box[2]) / 2

                # Get the radius of the circle from the box dimensions
                radius = max((box[3] - box[1]) / 2, (box[2] - box[0]) / 2)

                # Draw a circle around the detected object
                cv2.circle(frame, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)

                x, y, z = estimate_position(center_x, center_y, radius, frame.shape[:2], args.path, params)
                
                send_data((x,y,z))

            if args.preview:
                cv2.imshow("video", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    main(args)