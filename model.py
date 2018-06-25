# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:22:51 2018

@author: hayesmat
"""

import csv
import cv2
import numpy as np

lines = []
with open('./data/data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        lines.append(line)
#         break

images = []
measurments = []
correction = [0.0,0.2,-0.2]
for line in lines:
    for idx in range(3):
        source_path = line[idx]
        filename = source_path.split('/')[-1]
        current_path = './data/data/IMG/' + filename
        image = cv2.imread(current_path)
        vimage=image.copy()
        vimage=cv2.flip(image,1)
        images.append(image)
        images.append(vimage)
        measurment = float(line[3]) + correction[idx]
        measurments.append(measurment)
        measurments.append(-measurment)
    
X_train = np.array(images)
y_train = np.array(measurments)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
n_epoch = 5
n_batch = 128
for i in range(n_epoch):
    model.fit(X_train, y_train, batch_size=n_batch, validation_split=0.2, shuffle=True)
model.save('model.h5')
exit()