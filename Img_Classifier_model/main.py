#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:10:45 2019

@author: dilshaad
"""

from matplotlib import image
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

curr_dr=os.getcwd()

train_dir = os.path.join(curr_dr,'train')

train_horse_dir = os.path.join(train_dir, 'horses')  
train_human_dir = os.path.join(train_dir, 'humans')


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example(im):
    img = load_image(im)
    return img

train_horse=os.listdir(train_horse_dir)
train_human=os.listdir(train_human_dir)
train_horse_data=[]
train_human_data=[]

for x in train_horse:
    path=os.path.join(train_horse_dir,x)
    train_horse_data.append(run_example(path))

for x in train_human:
    path=os.path.join(train_human_dir,x)
    train_human_data.append(run_example(path))

#%%
    
x_train0,x_test0=train_test_split(train_horse_data,test_size=0.33,random_state=2)
x_train1,x_test1=train_test_split(train_human_data,test_size=0.33,random_state=2)

horse_target_train=[[0] for x in x_train0]
human_target_train=[[1] for x in x_train1]


horse_target_test=[[0] for x in x_test0]
human_target_test=[[1] for x in x_test1]

x_train=np.array(x_train0+x_train1)
#x_train=x_train/255.0
y_train=np.array(horse_target_train+human_target_train)
#y_train=y_train/255.0
x_test=np.array(x_test0+x_test1)
#x_test=x_test/255.0
y_test=np.array(horse_target_test+human_target_test)
#y_test=y_test/255.0


#%%


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%

from sklearn.metrics import accuracy_score
#new=np.arange(1,20)
#new2=[]
#for i in new:
  #  model.fit(x_train,y_train,epochs=i)
 #   pred=model.predict(x_test)

model.fit(x_train,y_train,epochs=21)
pred=model.predict(x_test)

tar=[]
for x in pred:
    tar.append(np.argmax(x))

print(accuracy_score(y_test,tar))
score=model.evaluate(x_test,y_test,verbose=0)
print("score",score[1])

