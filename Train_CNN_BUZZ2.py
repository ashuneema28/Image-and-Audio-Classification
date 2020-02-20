import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import pickle
import numpy as np

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

def load(filename):
    with open(path + filename, 'rb') as fp:
        input_array = pickle.load(fp)
    return input_array

path= "C:/Project1/.idea/AUDIO/BUZZ2_PICKLES/"

train_d = load("train_pre_processed_data_BUZZ2.pck")
test_d = load("test_pre_processed_data_BUZZ2.pck")

def get_input_data(data):
    input_dX =[]
    input_dY =[]
    for x in range(len(data)):
        input_dX.append(data[x][0])
        input_dY.append(data[x][1])
    return np.array(input_dX), input_dY

input_train_d, label_train_d = get_input_data(train_d)
input_test_d, label_test_d = get_input_data(test_d)

input_train_d = input_train_d.reshape([-1,150,100,1])
input_test_d = input_test_d.reshape([-1,150,100,1])

def build_CNN_BUZZ1():
    input_layer = input_data(shape=[None, 150, 100, 1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=70,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=50,
                           filter_size=5,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3  = conv_2d(pool_layer_2,
                            nb_filter=50,
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
                         learning_rate=0.05)
    model = tflearn.DNN(network)
    return model

NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = build_CNN_BUZZ1()
MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(input_test_d, label_test_d),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='Train_CNN_BUZZ2')


SAVE_BUZZ2_CNN_PATH = 'C:/Project1/.idea/AUDIO/BUZZ2_CNN_ADI/BUZZ2_CNN_ADI.tfl'
MODEL.save(SAVE_BUZZ2_CNN_PATH)