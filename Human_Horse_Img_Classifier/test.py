from PIL import Image
import matplotlib.pyplot as plt
import glob
from matplotlib import image
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


horse_list = []
for filename in glob.glob('/home/dilshaad/Desktop/ML/horses/*.png'):
    data=image.imread(filename)
    horse_list.append(data)
horse_target=[0 for x in range(len(horse_list))]

human_list = []
for filename in glob.glob('/home/dilshaad/Desktop/ML/humans/*.png'):
    data=image.imread(filename)
    human_list.append(data)    

human_target=[1 for x in range(len(human_list))]

print(horse_list[0].shape)

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data



#%%
horse_train,horse_train_target,horse_test,horse_test_target=train_test_split(horse_list[0:5],horse_target[0:5],test_size=0.33,random_state=42)
human_train,human_train_target,human_test,human_test_target=train_test_split(human_list[0:5],human_target[0:5],test_size=0.33,random_state=42)

X_Train=horse_train+human_train
Y_Train=horse_train_target+human_train_target

X_Test=horse_test+human_test
Y_Test=horse_test_target+human_test_target
 
model=keras.Sequential([
        keras.layers.Flatten(input_shape=(300,300,4)),
        keras.layers.Dense(300,activation=tf.nn.relu),
        keras.layers.Dense(2,activation=tf.nn.softmax)
        ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_Train,Y_Train)
