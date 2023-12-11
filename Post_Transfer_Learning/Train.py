# General Set of Libraries
import numpy as np
import os
import pandas as pd
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras import layers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# Model Specific Libraries
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.applications.efficientnet import preprocess_input


# Load the pre-trained model without including the top (fully connected layers) - Choose one out of the four models
# Already existing weights can also be loaded very easily from .h5 files

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3)) # Input_Shape can be adjusted based on the image
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create your own fully connected layers on top of the model base
x = Flatten()(base_model.output)
x = Dense(16, activation='relu')(x) # Number of Nodes
output = Dense(5, activation='softmax')(x)

# Create a new model by combining the base model and custom layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the labels from the .csv file
df = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML\Diabetic_Retinopathy/trainLabels.csv')

# Define the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255.)

temp = []
for i in range(len(df['level'])):
    temp.append([df['level'][i]])

df['level'] = temp
df['image'] = df['image']+'.jpeg'

# Load the images from the dataframe
train_generator = datagen.flow_from_dataframe(
    dataframe = df,
    directory= 'C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/preprocess',
    x_col = 'image',
    y_col = 'level',
    batch_size = 32,
    class_mode = 'categorical',
    target_size=(200,200) # Image Size
)

# train the model on the new data for a few epochs
model.fit(
    train_generator,
    batch_size = 32,
    epochs = 10,
    steps_per_epoch = len(train_generator),
)

model.save('trained_model.h5')