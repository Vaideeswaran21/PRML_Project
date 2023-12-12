# PRML_Project
This project is regarding Diabetic Retinopathy Detection and Classification using Deep Learning Models. Dataset 1 comprises the images along with their corresponding labels in a csv format.
Since the original images had extremly high resolutions, it became imperative to preprocess them. This was done by performing adaptive cropping followed by resizing down to 200x200x3 resolution (check pre-process_Adaptive_Cropping.py for more details)

Initially, a transfer learning framework was used to compare between VGG16, ResNet50, EfficientNetB3, and EfficientNetB4. 
Subsequently, it was arrived at that EfficientNetB4 was the better choice.
The performance of the models were assessed before and after transfer learning using Evaluation_Metric.py
The csv outputs can be found in Pre_Trained and Post_Transfer_Learning respectively.

# Work with EfficientNetB4

Several methods including regular hyper-parameter tuning, class-weighting, rotational augmentation (Check Augmentation.py), use of synthesizers like SMOTE and ADASYN were explored (Check Train_Experiments). The performances were assessed using Test.py.
Remove.py was used to remove images belonging to the dominant classes to ensure a more equitable class distribution.
Note that Augmented Dataset 1 comprises the rotated images along with their corresponding labels in a csv format.

# New Dataset

Owing to several challenges with the first dataset, a new dataset had to be explored. The results on that dataset were more promising and it can be found in final-code.ipynb.

# Auxiliary codes

1. Since we are performing large amounts of file handling, it becomes imperative to see if the contents in the image directory and the labels provided in the csv file actually match. This is done so using Match_Check.py. Discrepancies (if any) are reported as an output list.
2. model_info.py provides a detailed flow-chart of the model architecture, number of node and so forth.

# Reports

All the detailed analysis can be found in the reports and slides. Links to the datasets can also be found in those. There might be rendering issues when viewed via GitHub and hence is downloading is strongly recommended.
