import  os, zipfile,shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
path_inception = '../horsevshumans/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
local_weights_file = path_inception
pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False
#Freeze the last output layer where I just want to start my training.
last_layer  = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

#Define a callback to stop a model training when it reaches a good training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')> 0.97):
            print("Reached 97% desired training, so cancelling the training")
            self.model.stop_training = True

#Define the continuied model from a part choose of the pre-trained model
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(pre_trained_model.input,x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

path_horsevshuman = '../horsevshumans/horse-or-human.zip'
path_validation = '../horsevshumans/validation-horse-or-human.zip'

shutil.rmtree('../tmp/')
local_zip = path_horsevshuman
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('../tmp/training')
zip_ref.close()

local_zip = path_validation
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('../tmp/validation')
zip_ref.close()

train_dir = '../tmp/training'
validation_dir = '../tmp/validation'
train_horses_dir = os.path.join(train_dir,'horses')
train_humans_dir = os.path.join(train_dir,'humans')
val_horses_dir = os.path.join(validation_dir,'horses')
val_humans_dir = os.path.join(validation_dir,'humans')

#Count filenames
train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
val_horses_fnames = os.listdir(val_horses_dir)
val_humans_fnames = os.listdir(val_humans_dir)

print(len(train_horses_fnames), len(train_humans_fnames))
print(len(val_humans_fnames), len(val_humans_fnames))

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=0.2,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255.)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150))
callbacks = myCallback()
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=50,
                              epochs=3,
                              validation_steps=50,
                              verbose=2,
                              callbacks=[callbacks])
