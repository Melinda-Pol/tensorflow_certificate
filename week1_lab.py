import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
#Define and create a model to predict pricing houses
def house_model(y_new):
    xs = np.array([0,1,2,3,4,5,6],dtype=float)
    ys = np.array([0.5,1,1.5,2,2.5,3,3.5],dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd',loss='mse')
    model.fit(xs,ys,epochs=500)
    return model.predict(y_new)[0]
prediction = house_model([10.0])
print(prediction)