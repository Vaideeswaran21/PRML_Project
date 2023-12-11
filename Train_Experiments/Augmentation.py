# Libraries
import numpy as np
import os
import pandas as pd
from PIL import Image
import os

# Load the labels from the .csv file

counter = 0

# Load the labels from the .csv file
df1 = pd.read_csv('/raid/ambarishp/PRML/trainLabels.csv')
directory = '/raid/ambarishp/PRML/preprocess'
temp = []
temp2 = []
augmented_files = {}

for i in range(len(df1['level'])):
    temp.append(df1['level'][i])
    temp2.append(df1['image'][i])

i = 0
while (counter < 127):
    if (temp[i] == 4):
        file_path = '/raid/ambarishp/PRML/preprocess/' + temp2[i] + '.jpeg'
        img = Image.open(file_path)
        
        # Rotate the image by 90 degrees
        rotated_img = img.rotate(90)
        rotated_img.save('/raid/ambarishp/PRML/preprocess/' + '90deg' + temp2[i] + '.jpeg')
        augmented_files['90deg' + temp2[i]] = 4
        
        # Rotate the image by 180 degrees
        rotated_img = img.rotate(180)
        rotated_img.save('/raid/ambarishp/PRML/preprocess/' + '180deg' + temp2[i] + '.jpeg')
        augmented_files['180deg' + temp2[i]] = 4
        
        # Rotate the image by 270 degrees
        rotated_img = img.rotate(270)
        rotated_img.save('/raid/ambarishp/PRML/preprocess/' + '270deg' + temp2[i] + '.jpeg')
        augmented_files['270deg' + temp2[i]] = 4

        counter = counter + 1
    i = i + 1

# Convert the augmented_files dictionary to a list of dictionaries
new_entries = [{'image': k, 'level': v} for k, v in augmented_files.items()]
df_new_entries = pd.DataFrame(new_entries)
df_updated = pd.concat([df1, df_new_entries], ignore_index=True)
df_updated.to_csv('/raid/ambarishp/PRML/trainLabels.csv', index=False)