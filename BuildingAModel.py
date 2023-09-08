import pickle
import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Defining the input shape
width, height = 224, 224
shape = (width, height, 3) # RGB images

# Number of classes
num_classes = 5

model = Sequential()

# Adding layers
model.add(Conv2D(16, (3, 3), input_shape=shape))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), input_shape=shape))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=shape))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), input_shape=shape))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

# Connecting the layers (fully)
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()

batch_size = 32
epochs = 10

train_set = 'your path to train directory'
test_set = 'your path to test directory'
val_set = 'your path to validation directory'

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_set = train_gen.flow_from_directory(train_set, target_size=(width, height), batch_size=batch_size, class_mode='categorical')
test_set = test_gen.flow_from_directory(test_set, target_size=(width, height), batch_size=batch_size, class_mode='categorical')
val_set = val_gen.flow_from_directory(val_set, target_size=(width, height), batch_size=batch_size, class_mode='categorical')

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_set, steps_per_epoch=len(train_set), epochs = epochs, validation_data=val_set, validation_steps=len(val_set), callbacks=[early_stop])

test_loss, test_acc = model.evaluate(test_set)

print("Test accuracy: ", test_acc)

model.save('AstroImage_classification_model.h5')

