import csv
import  numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
import  matplotlib.pyplot as plt

#Getdata function
def get_data(filename):
    with open(filename) as training_file:
        read_file = csv.reader(training_file, delimiter=',')
        lines = True
        tmp_img = []
        tmp_labels = []
        for row in read_file:
            if lines:
                lines = False
            else:
                tmp_labels.append(row[0])
                img_data = row[1:785]
                img_array = np.array_split(img_data,28)
                tmp_img.append(img_array)
        images = np.array(tmp_img).astype('float')
        labels = np.array(tmp_labels).astype('float')

    return images, labels
path_sign_mnist_train = '../multiclassifier/archive/sign_mnist_train/sign_mnist_train.csv'
path_sign_mnist_test = '../multiclassifier/archive/sign_mnist_test/sign_mnist_test.csv'

training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)


training_images = np.expand_dims(training_images,3)
testing_images = np.expand_dims(testing_images, 3)

train_datagen = ImageDataGenerator(
    rescale= 1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255.)
print(training_images.shape, testing_images.shape)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.90):
            print("Reached desired accuracy 90%, cancelling training")
            self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(25,activation='softmax')
])

train_generator = train_datagen.flow(training_images, training_labels, batch_size=32)
validation_generator = validation_datagen.flow(testing_images, testing_labels, batch_size=32)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(training_images)/32,
                              epochs=20,
                              validation_data=validation_generator,
                              validation_steps=len(testing_images)/32,
                              verbose=1,
                              callbacks=[callbacks])
model.evaluate(testing_images,testing_labels, verbose=0)

#Plot chart

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss,'r',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

