# General Set of Libraries
import numpy as np
import os
import pandas as pd
import csv
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras import layers
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE, ADASYN

# Model Specific Libraries
from keras.applications.efficientnet import EfficientNetB4, preprocess_input

# Load the pre-trained model without including the top (fully connected layers)
# Already existing weights can also be loaded very easily from .h5 files

base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(200, 200, 3)) # Input_Shape can be adjusted based on the image
# loaded_model = load_model('/raid/ambarishp/PRML/weights-improvement-03-0.37.h5')

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = True

# Create your own fully connected layers on top of the model base
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)  # new FC layer, random init
predictions = Dense(5, activation='softmax')(x)  # new softmax layer
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
opt = Adam(learning_rate = 1e-5,  weight_decay = 5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Load the labels from the .csv file
df1 = pd.read_csv('/raid/ambarishp/PRML/trainLabels.csv')
df2 = pd.read_csv('/raid/ambarishp/PRML/valLabels.csv')

# Define the ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255.)
validation_datagen = ImageDataGenerator(rescale=1./255)

temp = []
for i in range(len(df2['level'])):
    temp.append([df2['level'][i]])

df2['level'] = temp
df2['image'] = df2['image']+'.jpeg'

# Load the images from the dataframe
X_train = []  # Store your training images here as numpy arrays
y_train = []  # Store your training labels here as numpy arrays

for index, row in df1.iterrows():
    img = image.load_img(os.path.join('/raid/ambarishp/PRML/preprocess', row['image'] + '.jpeg'), target_size=(200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    X_train.append(img)
    y_train.append(row['level'])

X_train = np.vstack(X_train)
y_train = pd.get_dummies(y_train).values

# Apply SMOTE/ ADASYN to balance the classes

# Code for SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(-1, 200*200*3), np.argmax(y_train, axis=1))

# Code for ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train.reshape(-1, 200*200*3), np.argmax(y_train, axis=1))

y_train_resampled = pd.get_dummies(y_train_resampled).values

# Reshape X_train_resampled back to image shape
X_train_resampled = X_train_resampled.reshape(-1, 200, 200, 3)

# Create ImageDataGenerator with resampled data
train_datagen = ImageDataGenerator(rescale=1./255.)

print(y_train_resampled)
train_generator = train_datagen.flow(
    X_train_resampled, y_train_resampled,
    batch_size=16
)

# Generate batches of validation data
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = df2,
    directory = '/raid/ambarishp/PRML/validation',
    x_col = 'image',
    y_col = 'level',
    target_size = (200, 200),  # adjust to your image size
    batch_size = 16,
    class_mode='categorical'
)

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

# train the model on the new data for a few epochs
model.fit(
    train_generator,
    batch_size = 16,
    epochs = 15,
    steps_per_epoch = len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks_list
)