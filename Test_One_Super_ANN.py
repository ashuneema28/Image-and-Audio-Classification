import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np
import pickle
from scipy import stats

directory_path = 'C:/Project1/.idea/DONE and DUSTED/ONE_SUPER_ANN/ONE_SUPER_ANN2.tfl'

def load_ann_One_Super(path):
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

one_super_ann_model = load_ann_One_Super(directory_path)

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def test_tflearn_ann_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 90, 90, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

valid_d = load("C:/Project1/.idea/DONE and DUSTED/ONE_SUPER_ANN/test_pre_processed_data_one_super.pck")

def get_input_data(data):
    input_dX =[]
    input_dY =[]
    for x in range(len(data)):
        input_dX.append(data[x][0])
        input_dY.append(data[x][1])
    return np.array(input_dX), input_dY

input_valid_d, label_valid_d = get_input_data(valid_d)

if __name__ == '__main__':
    print('One Super ANN accuracy = {}'.format(test_tflearn_ann_model(one_super_ann_model, input_valid_d, label_valid_d)))
