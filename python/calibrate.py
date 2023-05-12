#!/usr/bin/env python3

import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt


images = []
distances = []
with open('data.csv', 'r') as f:
    _data = csv.reader(f, delimiter=' ')
    for row in _data:
        images.append(cv2.imread(row[0]))
        distances.append(float(row[1]))

X = []
Y = []
for i, image in enumerate(images):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=100, param2=60, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
    else:
        continue
    resolution = image.shape[:2]
    if len(circles) > 0:
        size = (circles[0][2]*2) / resolution[0] # != diameter
        X.append(size)
        Y.append(distances[i])

for x,y in zip(X,Y):
    print(f's:{x:.3f}, z:{y:.3f}')

plt.plot(X, Y)
plt.grid()
plt.show()