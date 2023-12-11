# General Set of Libraries
import numpy as np
import os
import pandas as pd
import csv
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, cohen_kappa_score, balanced_accuracy_score, accuracy_score
from keras.preprocessing import image
from keras import layers
import keras
import multiprocessing
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten

# Model Specific Libraries
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.efficientnet import preprocess_input


loaded_model = load_model('/raid/ambarishp/PRML/weights-improvement-01-0.59.h5')
   
for layer in loaded_model.layers:
    layer.trainable = False

directory = sorted(os.listdir("/raid/ambarishp/PRML/test"), key = lambda x : (100*int(x.split('_')[0])) + ord(x.split('_')[1][0]))

list_len_B4 = len(directory)

with open('/raid/ambarishp/PRML/EfficientNetB4_Labels.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    arr = []
    arr.append('image')
    arr.append('level')
    writer.writerow(arr)

    for y in range(list_len_B4):
        img_path = "/raid/ambarishp/PRML/test/" + directory[y]
        img = image.load_img(img_path, target_size=(200, 200, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = loaded_model.predict(x, verbose = 1)
        temp = preds.argmax()
        arr = []
        arr.append(directory[y])
        arr.append(temp)
        writer.writerow(arr)

# Load the labels from the .csv file
df1 = pd.read_csv('/raid/ambarishp/PRML/testLabels.csv')
df2 = pd.read_csv('/raid/ambarishp/PRML/EfficientNetB4_Labels.csv')

y1 = [] # True Label
y2 = [] # Predicted Label - EfficientNetB4

for y in df1['level']:
    y1.append(y)

for y in df2['level']:
    y2.append(y)

print("Accuracy: ",accuracy_score(y1, y2))
print("Recall: ",recall_score(y1, y2, average = 'micro'))
print("Precision: ",precision_score(y1, y2, average='micro'))# General Set of Libraries
import numpy as np
import os
import pandas as pd
import csv
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, cohen_kappa_score, balanced_accuracy_score, accuracy_score
from keras.preprocessing import image
from keras import layers
import keras
import multiprocessing
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten

# Model Specific Libraries
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.efficientnet import preprocess_input


loaded_model = load_model('/raid/ambarishp/PRML/weights-improvement-01-0.59.h5')
   
for layer in loaded_model.layers:
    layer.trainable = False

directory = sorted(os.listdir("/raid/ambarishp/PRML/test"), key = lambda x : (100*int(x.split('_')[0])) + ord(x.split('_')[1][0]))

list_len_B4 = len(directory)

with open('/raid/ambarishp/PRML/EfficientNetB4_Labels.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    arr = []
    arr.append('image')
    arr.append('level')
    writer.writerow(arr)

    for y in range(list_len_B4):
        img_path = "/raid/ambarishp/PRML/test/" + directory[y]
        img = image.load_img(img_path, target_size=(200, 200, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = loaded_model.predict(x, verbose = 1)
        temp = preds.argmax()
        arr = []
        arr.append(directory[y])
        arr.append(temp)
        writer.writerow(arr)

# Load the labels from the .csv file
df1 = pd.read_csv('/raid/ambarishp/PRML/testLabels.csv')
df2 = pd.read_csv('/raid/ambarishp/PRML/EfficientNetB4_Labels.csv')

y1 = [] # True Label
y2 = [] # Predicted Label - EfficientNetB4

for y in df1['level']:
    y1.append(y)

for y in df2['level']:
    y2.append(y)

print("Accuracy: ",accuracy_score(y1, y2))
print("Recall: ",recall_score(y1, y2, average = 'micro'))
print("Precision: ",precision_score(y1, y2, average='micro'))
print("F1_Score: ",f1_score(y1, y2, average = 'micro'))
print("Kappa Score: ",cohen_kappa_score(y1, y2, weights = 'quadratic'))
print("Confusion Matrix: ",confusion_matrix(y1, y2))
print("F1_Score: ",f1_score(y1, y2, average = 'micro'))
print("Kappa Score: ",cohen_kappa_score(y1, y2, weights = 'quadratic'))
print("Confusion Matrix: ",confusion_matrix(y1, y2))