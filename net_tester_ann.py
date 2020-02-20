import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np
import pickle
from scipy import stats
import cv2
from scipy.io import wavfile
import glob
from sklearn.utils import shuffle

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def load_image_ann_bee1(path):
    input_layer = input_data(shape=[None,32,32,1])
    fc_layer_1 = fully_connected(input_layer, 90,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(fc_layer_3)
    model.load(path)
    return model

def load_image_ann_one_super(path):
    input_layer = input_data(shape=[None,90,90,1])
    fc_layer_1 = fully_connected(input_layer, 80,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 60,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_4 = fully_connected(fc_layer_3, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_4, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.5)
    model = tflearn.DNN(fc_layer_4)
    model.load(path)
    return model

def load_image_ann_two_super(path):
    input_layer = input_data(shape=[None,90,90,1])
    fc_layer_1 = fully_connected(input_layer, 60,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.05)
    model = tflearn.DNN(fc_layer_3)
    model.load(path)
    return model

def load_audio_ann_buzz1(path):
    input_layer = input_data(shape=[None,150,100,1])
    fc_layer_1 = fully_connected(input_layer, 250,
                                 activation='relu',
                                 regularizer='L2',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 150,
                                 activation='relu',
                                 regularizer='L2',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(fc_layer_3)
    model.load(path)
    return model

def load_audio_ann_buzz2(path):
    input_layer = input_data(shape=[None,440,100,1])
    fc_layer_1 = fully_connected(input_layer, 100,
                                 activation='relu',
                                 regularizer='L2',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 70,
                                 activation='relu',
                                 regularizer='L2',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(fc_layer_3)
    model.load(path)
    return model

def fit_image_ann_bee1(ann,image_path):
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    input = scaled_gray_image

    prediction = ann.predict(input.reshape([-1,32,32,1]))
    result = [0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

# *********THE FIT FUNCTION FOR BEE2-1S AND BEE2-2S IS THE SAME**********
def fit_image_ann_bee2(ann,image_path):
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    input = cv2.resize(scaled_gray_image,(90,90))

    prediction = ann.predict(input.reshape([-1,90,90,1]))
    result = [0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

def fit_audio_ann_buzz1(ann,image_path):
    samplerate, audio = wavfile.read(image_path)
    audio = audio/float(np.max(audio))
    audio = audio[45000:60000]
    input = audio

    prediction = ann.predict(input.reshape([-1, 150, 100, 1]))
    result =[0,0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

def fit_audio_ann_buzz2(ann,image_path):
    samplerate, audio = wavfile.read(image_path)
    audio = audio/float(np.max(audio))
    audio = audio[22000:66000]
    input = audio

    prediction = ann.predict(input.reshape([-1, 440, 100, 1]))
    result =[0,0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

ann_model = load_image_ann_bee1('C:/Users/A02290684/Desktop/Intelligent systems/Project1/.idea/Done and Dusted/BEE1_ANN/BEE1_ANN2.tfl')
result = fit_image_ann_bee1(ann_model,"C:/Users/A02290684/Desktop/Intelligent systems/Project1/.idea/BEE1/bees/bee_test/img0/70_298_yb.png")