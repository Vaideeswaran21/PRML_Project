# This code checks if the all the images present in the training directory have been labelled in the csv file.

# Libraries
import numpy as np
import os
import pandas as pd

# Load the labels from the .csv file
df = pd.read_csv('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML\Diabetic_Retinopathy/trainLabels.csv')

df['image'] = df['image']
temp = []

directory = os.listdir("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML\Diabetic_Retinopathy/preprocess")
list_len = len(directory)
counter = 0

for y in df['image']:
    y = y + '.jpeg'
    temp.append(y)


for x in range(list_len):
    if directory[x] in temp:
        counter  = counter + 1 
    else:
        print(directory[x])