from matplotlib import image
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

curr_dr=os.getcwd()
PATH=os.path.join(curr_dr,'horse_human_filtered')

train_dir = os.path.join(PATH,'train')
validation_dir = os.path.join(PATH, 'validation')

print(train_dir,validation_dir)

train_horse_dir = os.path.join(train_dir, 'horse')  # directory with our training cat pictures
train_human_dir = os.path.join(train_dir, 'human')  # directory with our training dog pictures
validation_horse_dir = os.path.join(validation_dir, 'horse')  # directory with our validation cat pictures
validation_human_dir = os.path.join(validation_dir, 'human')

num_horse_tr = len(os.listdir(train_horse_dir))
num_human_tr = len(os.listdir(train_human_dir))

num_horse_val = len(os.listdir(validation_horse_dir))
num_human_val = len(os.listdir(validation_human_dir))

total_train = num_horse_tr + num_human_tr
total_val = num_horse_val + num_human_val

print(os.listdir(train_horse_dir))

batch_size = 128
epochs = 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
#%%%
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

#%%
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

#%%

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#%%
from skimage.transform import resize
data=plt.imread('horse01-0.png')
plt.imshow(data)
plt.show()

new_image=resize(data,(32,32,3))
plt.imshow(new_image)
plt.show()