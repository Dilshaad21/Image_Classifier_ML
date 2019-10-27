import matplotlib.pyplot as plt
from skimage.transform import resize
import os

curr_dr=os.getcwd()
PATH=os.path.join(curr_dr,'horse_human_filtered')

train_dir = os.path.join(PATH,'train')
validation_dir = os.path.join(PATH, 'validation')

print(train_dir,validation_dir)

train_horse_dir = os.path.join(train_dir, 'horse')  # directory with our training cat pictures
train_human_dir = os.path.join(train_dir, 'human')  # directory with our training dog pictures
validation_horse_dir = os.path.join(validation_dir, 'horse')  # directory with our validation cat pictures
validation_human_dir = os.path.join(validation_dir, 'human')

#%%

import random

train_horse=os.listdir(validation_human_dir)
for x in train_horse:
    path=os.path.join(validation_human_dir,x)
    data=plt.imread(path)
    new_image=resize(data,(32,32,3))
    new_file=os.path.join(validation_human_dir,str(random.randint(1,10000)))    
    plt.imsave(new_file,new_image)
