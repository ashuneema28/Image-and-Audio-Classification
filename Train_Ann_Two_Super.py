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

path= "C:/Project1/.idea/"

train_d = load("train_pre_processed_data_two_super_ANN.pck")
test_d = load("test_pre_processed_data_two_super_ANN.pck")
valid_d = load("valid_pre_processed_data_two_super_ANN.pck")

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
print(input_train_d.shape)

input_train_d = input_train_d.reshape([-1,90,90,1])
input_test_d = input_test_d.reshape([-1,90,90,1])

def build_ann_Two_Super():
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
    model = tflearn.DNN(network)
    return model

NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = build_ann_Two_Super()
MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(input_test_d, label_test_d),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='Train_ANN_Two_Super')


SAVE_BEE1_ANN_PATH = 'C:/Project1/.idea/TWO_SUPER_ANN/TWO_SUPER_ANN.tfl'
MODEL.save(SAVE_BEE1_ANN_PATH)