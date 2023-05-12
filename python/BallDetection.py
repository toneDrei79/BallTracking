import cv2
import tensorflow as tf
import numpy as np
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/Users/tone/Documents/project/Ball/Assets/coordinates.csv', help='path of the coordinate.csv')
    parser.add_argument('--preview', action='store_true', help='preview video')
    parser.add_argument('--confidence', type=float, default=0.5, help='detection confidence')
    return parser.parse_args()



# Global Variables
modelFlag = 0
detection_model = None

def loadModel():
    global modelFlag, detection_model
    if not modelFlag:
        detection_model = tf.saved_model.load('Data/out3/saved_model')
        print("Model Loaded")
        modelFlag = 1

def perspective(_position):
    z = _position[2]
    ratio = 1 / z
    x = _position[0] * ratio
    y = _position[1] * ratio
    position = (x, y, z)
    return position

def estimate_position(_x, _y, _rad, resolution, path):
    x = -0.5 + _x / resolution[0]
    y = 0.5 - _y / resolution[1]
    size = (_rad*2) / resolution[0] # != diameter

    z = 0.5 / (3*size-0.1) + 0.1
    x, y, z = perspective((x,y,z))
    
    with open(path, 'w') as f:
        f.write(f'{x:.3f} {y:.3f} {z:.3f}')


def main(args):
    loadModel()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Add a new axis to match the input size requirement of the model
        rgb_frame_expanded = np.expand_dims(rgb_frame, axis=0)

        # Run the frame through the model
        output_dict = detection_model(rgb_frame_expanded)

        # Get the number of objects detected
        num_detections = int(output_dict.pop('num_detections'))

        # Detection scores are the detection confidence
        detection_scores = output_dict['detection_scores'][0].numpy()

        # Detection classes are the id of the detected object
        detection_classes = output_dict['detection_classes'][0].numpy().astype(np.uint32)

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

            estimate_position(center_x, center_y, radius, frame.shape[:2], args.path)

        # Display the resulting frame
        if args.preview:
            cv2.imshow('frame', frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    main(args)