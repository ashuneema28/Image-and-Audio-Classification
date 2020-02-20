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

path= "C:/Users/A02290684/Desktop/Intelligent systems/Project1/.idea/"

train_d = load("train_pre_processed_data_one_super_CNN.pck")
test_d = load("test_pre_processed_data_one_super_CNN.pck")
valid_d = load("valid_pre_processed_data_one_super_CNN.pck")

def get_input_data(data):
    input_dX =[]
    input_dY =[]
    for x in range(len(data)):
        input_dX.append(data[x][0])
        input_dY.append(data[x][1])
    return np.array(input_dX), input_dY

input_train_d, label_train_d = get_input_data(train_d)
input_test_d, label_test_d = get_input_data(test_d)
input_valid_d, label_valid_d = get_input_data(valid_d)

input_train_d = input_train_d.reshape([-1,90,90,3])
input_test_d = input_test_d.reshape([-1,90,90,3])

def build_CNN_one_super():
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
    model = tflearn.DNN(network)
    return model

NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = build_CNN_one_super()
MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(input_test_d, label_test_d),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='Train_One_Super_CNN')


SAVE_BEE1_ANN_PATH = 'C:/Users/A02290684/Desktop/Intelligent systems/Project1/.idea/ONE_SUPER_CNN2/ONE_SUPER_CNN2.tfl'
MODEL.save(SAVE_BEE1_ANN_PATH)


