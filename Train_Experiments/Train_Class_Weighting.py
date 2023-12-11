# General Set of Libraries
import numpy as np
import os
import pandas as pd
import csv
from sklearn.utils.class_weight import compute_class_weight
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras import layers, regularizers
from keras.models import Model, load_model, Sequential
from keras.layers import InputLayer, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Adamax

# Model Specific Libraries
from keras.applications.efficientnet import EfficientNetB4, preprocess_input

# Load the pre-trained model without including the top (fully connected layers)
# Already existing weights can also be loaded very easily from .h5 files

base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = True

# Create your own fully connected layers on top of the model base
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x) # Number of Nodes
output = Dense(5, activation='softmax')(x)

# Create a new model by combining the base model and custom layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Load the labels from the .csv file
df1 = pd.read_csv('/raid/ambarishp/PRML/trainLabels.csv')
df2 = pd.read_csv('/raid/ambarishp/PRML/valLabels.csv')

# Define the ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255.)
validation_datagen = ImageDataGenerator(rescale=1./255)

temp = []
for i in range(len(df1['level'])):
    temp.append([df1['level'][i]])

df1['level'] = temp
df1['image'] = df1['image']+'.jpeg'

# Assuming 'temp' is a list of lists and you want to extract the first element of each sublist
temp2 = [item[0] for item in df1['level']]  # Extracting the first element of each sublist

# Calculate class weights using compute_class_weight
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(temp2), 
    y=temp2
)
print(class_weights)

temp = []
for i in range(len(df2['level'])):
    temp.append([df2['level'][i]])

df2['level'] = temp
df2['image'] = df2['image']+'.jpeg'

# Generate batches of validation data
train_generator = train_datagen.flow_from_dataframe(
    dataframe = df1,
    directory = '/raid/ambarishp/PRML/preprocess',
    x_col = 'image',
    y_col = 'level',
    target_size = (200, 200),  # adjust to your image size
    batch_size = 48,
    class_mode='categorical'
)

# Generate batches of validation data
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = df2,
    directory = '/raid/ambarishp/PRML/validation',
    x_col = 'image',
    y_col = 'level',
    target_size = (200, 200),  # adjust to your image size
    batch_size = 48,
    class_mode='categorical'
)

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

cw = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3], 4: class_weights[4]}

# train the model on the new data for a few epochs
model.fit(
    train_generator,
    batch_size = 48,
    epochs = 15,
    steps_per_epoch = len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks_list,
    class_weight = cw
    )