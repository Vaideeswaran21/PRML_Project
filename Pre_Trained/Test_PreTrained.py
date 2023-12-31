# General Set of Libraries
import numpy as np
import os
import csv
from keras.preprocessing import image
from keras import layers
import keras
import multiprocessing

# Model Specific Libraries
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.applications.efficientnet import preprocess_input

# We use the multiprocess library to test all the 4 pre-trained models at the same time: Test VGG16 (first), ResNet50 (second), EfficientNetB3 and B4 (third and fourth respectively).

def first():

   base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
   
   for layer in base_model.layers:
        layer.trainable = False

   model = keras.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(5, activation='softmax')
    ])

   # Reading from the image directory

   directory = os.listdir("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test")
   list_len_VGG = len(directory)

   with open('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/VGG_Labels.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        
        for y in range(list_len_VGG):
            img_path = "C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test/" + directory[y]
            img = image.load_img(img_path, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            temp = preds.argmax()
            arr = []
            arr.append(directory[y])
            arr.append(temp)
            writer.writerow(arr)

def second():

   base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
   
   for layer in base_model.layers:
        layer.trainable = False

   model = keras.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(5, activation='softmax')
    ])

   # Reading from the image directory

   directory = os.listdir("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test")
   list_len_Res = len(directory)

   with open('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/ResNet50_Labels.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        
        for y in range(list_len_Res):
            img_path = "C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test/" + directory[y]
            img = image.load_img(img_path, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            temp = preds.argmax()
            arr = []
            arr.append(directory[y])
            arr.append(temp)
            writer.writerow(arr)

def third():

   base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
   
   for layer in base_model.layers:
        layer.trainable = False

   model = keras.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(5, activation='softmax')
    ])
   
   directory = os.listdir("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test")
   list_len_B3 = len(directory)
   
   with open('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/EfficientNetB3_Labels.csv', 'w', newline = '') as f:
        writer = csv.writer(f)

        for y in range(list_len_B3):
            img_path = "C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test/" + directory[y]
            img = image.load_img(img_path, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            temp = preds.argmax()
            arr = []
            arr.append(directory[y])
            arr.append(temp)
            writer.writerow(arr)

def fourth():
    
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(5, activation='softmax')  # 5 classes
    ])

    directory = os.listdir("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test")
    list_len_B4 = len(directory)
    
    with open('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/EfficientNetB4_Labels.csv', 'w', newline = '') as f:
        writer = csv.writer(f)

        for y in range(list_len_B4):
            img_path = "C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/test/" + directory[y]
            img = image.load_img(img_path, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            temp = preds.argmax()
            arr = []
            arr.append(directory[y])
            arr.append(temp)
            writer.writerow(arr)

if __name__== "__main__":
    prc1 = multiprocessing.Process(target = first)
    prc2 = multiprocessing.Process(target = second)
    prc3 = multiprocessing.Process(target = third)
    prc4 = multiprocessing.Process(target = fourth)

    prc1.start()
    prc2.start()
    prc3.start()
    prc4.start()

    prc1.join()
    prc2.join()
    prc3.join()
    prc4.join()
            
    print("END!")
