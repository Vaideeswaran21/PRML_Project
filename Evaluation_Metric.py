# Libraries
import numpy as np
import os
import sklearn
from sklearn import metrics
import pandas as pd

# Load the labels from the .csv file
df1 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/testLabels.csv')
df2 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/VGG_Labels.csv')
df3 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/ResNet50_Labels.csv')
df4 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/EfficientNetB3_Labels.csv')
df5 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Pre_Trained/EfficientNetB4_Labels.csv')

df6 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Transfer_Learning/VGG_Labels.csv')
df7 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Transfer_Learning/ResNet50_Labels.csv')
df8 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Transfer_Learning/EfficientNetB3_Labels.csv')
df9 = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/Transfer_Learning/EfficientNetB4_Labels.csv')

y1 = [] # True Label

# Before Transfer Learning

y2 = [] # Predicted Label - VGG
y3 = [] # Predicted Label - ResNet50
y4 = [] # Predicted Label - EfficientNetB3
y5 = [] # Predicted Label - EfficientNetB4

# After Transfer Learning

y6 = [] # Predicted Label - VGG
y7 = [] # Predicted Label - ResNet50
y8 = [] # Predicted Label - EfficientNetB3
y9 = [] # Predicted Label - EfficientNetB4

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

for y in df6['level']:
    y6.append(y)

for y in df7['level']:
    y7.append(y)

for y in df8['level']:
    y8.append(y)

for y in df9['level']:
    y9.append(y)

length = len(y1)

# Use the Quadratic Weighted Kappa metric - higher the value, better is the performance

print("We use Transfer Learning")

print("Before Training")
print("Kappa Metric for VGG16 is: ", sklearn.metrics.cohen_kappa_score(y1, y2, weights = 'quadratic'))
print("Kappa Metric for ResNet50 is: ", sklearn.metrics.cohen_kappa_score(y1, y3, weights = 'quadratic'))
print("Kappa Metric for EfficientNetB3 is: ", sklearn.metrics.cohen_kappa_score(y1, y4, weights = 'quadratic'))
print("Kappa Metric for EfficientNetB4 is: ", sklearn.metrics.cohen_kappa_score(y1, y5, weights = 'quadratic'))

print("After Training")
print("Kappa Metric for VGG16 is: ", sklearn.metrics.cohen_kappa_score(y1, y6, weights = 'quadratic'))
print("Kappa Metric for ResNet50 is: ", sklearn.metrics.cohen_kappa_score(y1, y7, weights = 'quadratic'))
print("Kappa Metric for EfficientNetB3 is: ", sklearn.metrics.cohen_kappa_score(y1, y8, weights = 'quadratic'))
print("Kappa Metric for EfficientNetB4 is: ", sklearn.metrics.cohen_kappa_score(y1, y9, weights = 'quadratic'))

# Average difference between the predicted and true labels

sum0 = 0
sum1 = 0
sum2 = 0
sum3 = 0

sum4 = 0
sum5 = 0
sum6 = 0
sum7 = 0

for i in range(length):
    sum0 = sum0 + abs(y1[i] - y2[i])
    sum1 = sum1 + abs(y1[i] - y3[i])
    sum2 = sum2 + abs(y1[i] - y4[i])
    sum3 = sum3 + abs(y1[i] - y5[i])
    sum4 = sum4 + abs(y1[i] - y6[i])
    sum5 = sum5 + abs(y1[i] - y7[i])
    sum6 = sum6 + abs(y1[i] - y8[i])
    sum7 = sum7 + abs(y1[i] - y9[i])

sum0 = sum0/length
sum1 = sum1/length
sum2 = sum2/length
sum3 = sum3/length

sum4 = sum4/length
sum5 = sum5/length
sum6 = sum6/length
sum7 = sum7/length

print("Before Training")  
print("Average difference between the predicted and true labels for VGG16: ",sum0)
print("Average difference between the predicted and true labels for ResNet50: ",sum1)
print("Average difference between the predicted and true labels for EfficientNetB3: ",sum2)
print("Average difference between the predicted and true labels for EfficientNetB4: ",sum3)

print("After Training")  
print("Average difference between the predicted and true labels for VGG16: ",sum4)
print("Average difference between the predicted and true labels for ResNet50: ",sum5)
print("Average difference between the predicted and true labels for EfficientNetB3: ",sum6)
print("Average difference between the predicted and true labels for EfficientNetB4: ",sum7)

print("Improvement in the average prediction error:")
print("VGG: ", sum0 - sum4)
print("ResNet50: ", sum1 - sum5)
print("EfficientNetB3: ", sum2 - sum6)
print("EfficientNetB4: ", sum3 - sum7)
