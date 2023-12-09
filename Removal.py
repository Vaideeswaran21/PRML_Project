# General Set of Libraries
import numpy as np
import os
import pandas as pd

counter = 0

# Load the labels from the .csv file
df1 = pd.read_csv('/raid/ambarishp/PRML/trainLabels2.csv')
directory = '/raid/ambarishp/PRML/preprocess2'
temp = []
temp2 = []
deleted_files = []

for i in range(len(df1['level'])):
    temp.append(df1['level'][i])
    temp2.append(df1['image'][i])

i = 0
while (counter < 4255):
    if (temp[i] == 0):
        file_path = '/raid/ambarishp/PRML/preprocess2/' + temp2[i] + '.jpeg'
        os.remove(file_path)
        deleted_files.append(temp2[i])
        counter = counter + 1
    i = i + 1

df1 = df1[~df1['image'].isin(deleted_files)] 
df1.to_csv('/raid/ambarishp/PRML/trainLabels2.csv', index=False)

