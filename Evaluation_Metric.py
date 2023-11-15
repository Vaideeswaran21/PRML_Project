# Libraries
import numpy as np
import os
import sklearn
from sklearn import metrics
import pandas as pd

# Load the labels from the .csv file
df1 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/testLabels.csv')
df2 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Models/VGG_Labels.csv')
df3 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Models/ResNet50_Labels.csv')
df4 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Models/EfficientNetB3_Labels.csv')
df5 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Models/EfficientNetB4_Labels.csv')

y1 = [] # True Label
y2 = [] # Predicted Label - VGG
y3 = [] # Predicted Label - ResNet50
y4 = [] # Predicted Label - EfficientNetB3
y5 = [] # Predicted Label - EfficientNetB4

for y in df1['level']:
    y1.append(y)

for y in df2['level']:
    y2.append(y)

for y in df3['level']:
    y3.append(y)

for y in df4['level']:
    y4.append(y)

for y in df5['level']:
    y5.append(y)

# Use the Quadratic Weighted Kappa metric - higher the value, better is the performance

print("Kappa Metric for VGG16 is: ", sklearn.metrics.cohen_kappa_score(y1, y2, weights = 'quadratic'))
print("Kappa Metric for ResNet50 is: ", sklearn.metrics.cohen_kappa_score(y1, y3, weights = 'quadratic'))
print("Kappa Metric for EfficientNetB3 is: ", sklearn.metrics.cohen_kappa_score(y1, y4, weights = 'quadratic'))
print("Kappa Metric for EfficientNetB4 is: ", sklearn.metrics.cohen_kappa_score(y1, y5, weights = 'quadratic'))
