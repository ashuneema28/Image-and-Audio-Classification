import cv2
from scipy.io import wavfile
import numpy as np
import glob
from sklearn.utils import shuffle
import pickle
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def load_image_convnet_bee1(path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

def load_image_convnet_one_super(path):
    input_layer = input_data(shape=[None, 90, 90, 3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=60,
                            filter_size=5,
                            activation='tanh',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='tanh',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.05)
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

def load_image_convnet_two_super(path):
    input_layer = input_data(shape=[None, 90, 90, 3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=60,
                            filter_size=5,
                            activation='tanh',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='tanh',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.05)
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

def load_audio_convnet_buzz1(path):
    input_layer = input_data(shape=[None, 150, 100, 1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=50,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3  = conv_2d(pool_layer_2,
                            nb_filter=40,
                            filter_size=3,
                            activation='relu',
                            name='conv_layer_3')
    pool_layer_3  = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1  = fully_connected(pool_layer_3, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

def load_audio_convnet_buzz2(path):
    input_layer = input_data(shape=[None, 150, 100, 1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=50,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3  = conv_2d(pool_layer_2,
                            nb_filter=40,
                            filter_size=3,
                            activation='relu',
                            name='conv_layer_3')
    pool_layer_3  = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1  = fully_connected(pool_layer_3, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

def fit_image_convnet_bee1(cnn, image_path):
    img = cv2.imread(image_path)
    scaled_gray_image = img/255.0
    input = scaled_gray_image

    prediction = cnn.predict(input.reshape([-1, 32, 32, 3]))
    result =[0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

# *********THE FIT FUNCTION FOR BEE2-1S AND BEE2-2S CONVNETS IS THE SAME**********
def fit_image_convnet_bee2(cnn, image_path):
    img = cv2.imread(image_path)
    scaled_gray_image = img/255.0
    input = cv2.resize(scaled_gray_image,(90,90))

    prediction = cnn.predict(input.reshape([-1, 90, 90, 3]))
    result =[0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

# *********THE FIT FUNCTION FOR BUZZ1 AND BUZZ2 CONVNETS IS THE SAME**********
def fit_audio_convnet(cnn, image_path):
    samplerate, audio = wavfile.read(image_path)
    audio = audio/float(np.max(audio))
    audio = audio[45000:60000]
    input = audio

    prediction = cnn.predict(input.reshape([-1, 150, 100, 1]))
    result =[0,0,0]
    output_position = np.argmax(prediction, axis=1)[0]
    result[output_position]=1
    print(result)
    return result

cnn_model = load_audio_convnet_buzz1("C:/Project1/.idea/DONE and DUSTED/BUZZ2_CNN/BUZZ2_CNN.tfl")
result = fit_audio_convnet(cnn_model, "C:/Project1/.idea/AUDIO/BUZZ2/out_of_sample_data_for_validation/noise_test/192_168_4_5-2018-05-12_19-45-01_3.wav")