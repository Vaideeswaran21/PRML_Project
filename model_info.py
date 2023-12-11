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
from tensorflow.keras.utils import plot_model
from IPython.display import Image

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

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3)) # Input_Shape can be adjusted based on the image
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
# base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
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

plot_model(model, to_file='Conv.png', show_shapes=True,show_layer_names=True, dpi = 72)
Image(filename='Conv.png')
