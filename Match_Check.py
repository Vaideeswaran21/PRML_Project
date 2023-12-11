# This code checks if the all the images present in the training directory have been labelled in the csv file.

# Libraries
import numpy as np
import os
import pandas as pd

# Load the labels from the .csv file
df = pd.read_csv('/raid/ambarishp/PRML/trainLabels3.csv')

df['image'] = df['image']
temp = []

directory = os.listdir("/raid/ambarishp/PRML/preprocess3")
list_len = len(directory)
counter = 0

for y in df['image']:
    y = y + '.jpeg'
    temp.append(y)

elements_only_in_array1 = list(set(temp) - set(directory))
print("Elements only in array1:", elements_only_in_array1)

elements_only_in_array2 = list(set(directory) - set(temp))
print("Elements only in array2:", elements_only_in_array2)

counter0 = 0
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0

for i in range(len(df['level'])):
    if (df['level'][i] == 0):
        counter0 = counter0 + 1
    if (df['level'][i] == 1):
        counter1 = counter1 + 1
    if (df['level'][i] == 2):
        counter2 = counter2 + 1
    if (df['level'][i] == 3):
        counter3 = counter3 + 1
    if (df['level'][i] == 4):
        counter4 = counter4 + 1
print(counter0)
print(counter1)
print(counter2)
print(counter3)
print(counter4)