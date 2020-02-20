import cv2
from scipy.io import wavfile
import numpy as np
import glob
from sklearn.utils import shuffle
import pickle

folder = "C:/Project1/.idea/one_super/"

nonBees = glob.glob(folder+'validation/no_bee/*/*.png')
bees = glob.glob(folder+'validation/bee/*/*.png')

imagePaths = []

for f in nonBees:
    imagePaths.append((f,(0,1)))
for f in bees:
    imagePaths.append((f,(1,0)))

newimg  = shuffle(imagePaths)
print(len(newimg))

def save(arr, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(arr, file)


filename2 ="C:/Project1/.idea/valid_pre_processed_data_one_super_CNN.pck"

test_d = []

#with open(filename2,'rb') as pck:
#    test_d = pickle.load(pck)

for x in range(len(newimg)):
    img = cv2.imread(newimg[x][0])
    scaled_gray_image = img/255.0
    test_d.append((cv2.resize(scaled_gray_image,(90,90)),newimg[x][1]))

test_d=np.array(test_d)

save(test_d,filename2)