# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import multiprocessing

# Reading from the image directory

directory = os.listdir("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML\Diabetic_Retinopathy/train")
list_len = len(directory)

# function for adaptive cropping

def adaptive_cropping(img_1):
    red = img_1[:,:,0]
    green = img_1[:,:,1]
    blue = img_1[:,:,2]

    red_T = np.transpose(red)
    green_T = np.transpose(green)
    blue_T = np.transpose(blue)

    threshold = 0.95
    (rows, columns) = np.shape(red)

    mod_image_red = []
    mod_image_green = []
    mod_image_blue = []

    for i in range(columns):
        if ((np.count_nonzero(red_T[i])/ rows) >= (1 - threshold) and (np.count_nonzero(green_T[i])/ rows) >= (1 - threshold) and (np.count_nonzero(blue_T[i])/ rows) >= (1 - threshold)):
            mod_image_red.append(red_T[i])
            mod_image_green.append(green_T[i])
            mod_image_blue.append(blue_T[i])

    mod_image_red_T = np.transpose(mod_image_red)
    mod_image_green_T = np.transpose(mod_image_green)
    mod_image_blue_T = np.transpose(mod_image_blue)

    mod_image = np.dstack((mod_image_red_T, mod_image_green_T, mod_image_blue_T))
    mod_image = cv2.resize(mod_image, dsize=(200, 200)) # Required Target Size
    return mod_image

# Multi-Processing helps to speed up the pre-processing

def first():
    for x in range(list_len//4):
        img_1 = mpimg.imread("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/train/" + directory[x])
        mod_image = adaptive_cropping(img_1)
        plt.imsave('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/preprocess/' + directory[x], mod_image)

def second():
    for x in range(list_len//4, list_len//2):
        img_1 = mpimg.imread("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/train/" + directory[x])
        mod_image = adaptive_cropping(img_1)
        plt.imsave('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/preprocess/' + directory[x], mod_image)

def third():
    for x in range(list_len//2, 3*list_len//4):
        img_1 = mpimg.imread("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/train/" + directory[x])
        mod_image = adaptive_cropping(img_1)
        plt.imsave('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/preprocess/' + directory[x], mod_image)

def fourth():
    for x in range(3*list_len//4, list_len):
        img_1 = mpimg.imread("C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/train/" + directory[x])
        mod_image = adaptive_cropping(img_1)
        plt.imsave('C:/Users/vaide/OneDrive/Documents/IIT Hyderabad/Semester 5/Full Semester Courses/PRML/Diabetic_Retinopathy/preprocess/' + directory[x], mod_image)

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