# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:00:22 2023

@author: Tulika
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp
import cv2
import os
#loading the dataset-load directly from tensorflow,no need to process csv files
mnist= tf.keras.datasets.mnist
#split the data and label data
# x being pixel data and y being the classification
(x_train,y_train), (x_test,y_test)=mnist.load_data()

#normalize the datas
x_train= tf.keras.utils.normalize(x_train, axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)

#models
model=tf.keras.models.Sequential()
#flat layer-flat an input shape-turns into 28*28 line of pixels
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#rectifier linear unit
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
#10 digits representation-10 neurons
#softmax-for probability of each neuron
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#model compiled
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

model.save('handwritten.model')

model=tf.keras.models.load_model('handwritten.model')

loss,accuracy=model.evaluate(x_test,y_test)
print(loss)
print(accuracy*100)


for img_no in range(1,10):
    img=cv2.imread(f'digits/digit{img_no}.png')[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print(f"The number is probably {np.argmax(prediction)}")
    mp.imshow(img[0], cmap=mp.cm.binary)
    mp.show()
    
