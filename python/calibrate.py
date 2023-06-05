import sys
import cv2
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt



# Global Variables
modelFlag = 0
detection_model = None

def loadModel():
    global modelFlag, detection_model
    if not modelFlag:
        detection_model = tf.saved_model.load('Data/out3/saved_model')
        print("Model Loaded")
        modelFlag = 1


def detect(image):
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = image[:, :, ::-1]

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
    # if score > args.confidence:
    if score > 0.5:
        box = detection_boxes[idxmax] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

        # Get the center of the box
        center_x = (box[1] + box[3]) / 2
        center_y = (box[0] + box[2]) / 2

        # Get the radius of the circle from the box dimensions
        radius = max((box[3] - box[1]) / 2, (box[2] - box[0]) / 2)

        return radius

        # Draw a circle around the detected object
        # cv2.circle(image, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)
    return -1



def main(filepath):
    loadModel()

    images = []
    distances = []
    with open(filepath, 'r') as f:
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


        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=100, param2=60, minRadius=0, maxRadius=0)
        # print(i, end=' ')
        # if circles is not None:
        #     circles = np.uint16(np.around(circles[0]))
        #     print('deteted')
        # else:
        #     print('not deteted')
        #     continue
        # resolution = image.shape[:2]
        # if len(circles) > 0:
        #     size = (circles[0][2]*2) / resolution[0] # != diameter
        #     X.append(size)
        #     Y.append(distances[i])
    
    for x,y in zip(X,Y):
        print(f's:{x:.3f}, z:{y:.3f}')

    plt.plot(X, Y)
    plt.grid()
    # plt.show()


    # least squares
    # y = a/_x + b
    # x = 1/_x
    aveX = np.mean([1/x for x in X])
    aveY = np.mean([y for y in Y])
    aveX2 = np.mean([(1/x)*(1/x) for x in X])
    aveXY = np.mean([y/x for x,y in zip(X,Y)])
    a = (len(X)*aveXY - aveX*aveY) / (len(X)*aveX2 - aveX*aveX)
    b = (aveX2*aveY - aveX*aveXY) / (len(X)*aveX2 - aveX*aveX)
    # print(a, b)
    sampleX = np.arange(0.05,1,0.01)
    sampleY = np.array([a/x + b for x in sampleX])
    plt.plot(sampleX, sampleY)
    plt.grid()
    plt.show()

    return


if __name__ == '__main__':
    args = sys.argv
    main(filepath=args[1])