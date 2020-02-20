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

path= "C:/Project1/.idea/AUDIO/ANN_PICKLES_BUZZ1/"

train_d = load("train_pre_processed_data_BUZZ1_2.pck")
test_d = load("test_pre_processed_data_BUZZ1_2.pck")

def get_input_data(data):
    input_dX =[]
    input_dY =[]
    for x in range(len(data)):
        input_dX.append(data[x][0])
        input_dY.append(data[x][1])
    return np.array(input_dX), input_dY

input_train_d, label_train_d = get_input_data(train_d)
input_test_d, label_test_d = get_input_data(test_d)

print(input_train_d.shape)
input_train_d = input_train_d.reshape([-1,150,100,1])
input_test_d = input_test_d.reshape([-1,150,100,1])

def build_ann_BUZZ1():
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
    model = tflearn.DNN(network)
    return model

NUM_EPOCHS = 50
BATCH_SIZE = 10
MODEL = build_ann_BUZZ1()
MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(input_test_d, label_test_d),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='Train_BUZZ1_ANN')


SAVE_BUZZ1_ANN_PATH = 'C:/Project1/.idea/AUDIO/BUZZ1_ANN2/BUZZ1_ANN2.tfl'
MODEL.save(SAVE_BUZZ1_ANN_PATH)