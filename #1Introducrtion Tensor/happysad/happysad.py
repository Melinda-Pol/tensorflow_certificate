import tensorflow as tf
import os, zipfile
from os import  path, getcwd, chdir
from  tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
path = '../happysad/happy-or-sad.zip'
zip_ref = zipfile.ZipFile(path,'r')
zip_ref.extractall('/tmp/h-or-s')
zip_ref.close()

def train_happy_sad_model():
    DESIRED_ACC = 0.99

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc')>= DESIRED_ACC):
                print('Reached 99% accuracy, so cancelling training')
                self.model.stop_training = True

    callbacks = myCallback()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')])

    model.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=['acc'])
    train_datagen = ImageDataGenerator(rescale=1./255.)
    train_generator = train_datagen.flow_from_directory('/tmp/h-or-s',
                                                        target_size=(150,150),
                                                        batch_size=10,
                                                        class_mode='binary')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=20,
        verbose=2,
        callbacks=[callbacks]
    )
    return history.history['acc'][-1]

train_happy_sad_model()