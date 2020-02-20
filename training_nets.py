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

def get_training_testing_validation_data(train_path,test_path,valid_path):
    train_d = load(train_path)
    test_d = load(test_path)
    valid_d = load(valid_path)
    return train_d,test_d,valid_d

def get_input_data(data):
    input_dX =[]
    input_dY =[]
    for x in range(len(data)):
        input_dX.append(data[x][0])
        input_dY.append(data[x][1])
    return np.array(input_dX), input_dY

##############################################################################################################

'''
BEE1 ANN Architecture:
    Layer 1 = input layer of 32*32 dimension
    Layer 2 = 90 neurons with relu as activation function 
    Layer 3 = 60 neurons with relu as activation function
    Layer 4 = output layer with 2 neurons and softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.01
    optimizer = sgd
'''

def build_ann_BEE1():
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
    model = tflearn.DNN(network)
    return model

def Train_Ann_BEE1(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,32,32,1])
    input_test_d = input_test_d.reshape([-1,32,32,1])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_ann_BEE1()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_Bee1_ANN')

    MODEL.save(save_path)

##############################################################################################################

'''
Two Super ANN Architecture:
    Layer 1 = input layer of 90*90 dimension
    Layer 2 = 60 neurons with relu as activation function 
    Layer 3 = 60 neurons with relu as activation function
    Layer 4 = output layer with 2 neurons and softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.05
    optimizer = sgd
'''


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

def Train_Ann_Two_Super(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,90,90,1])
    input_test_d = input_test_d.reshape([-1,90,90,1])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_ann_Two_Super()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_ANN_Two_Super')

    MODEL.save(save_path)

#############################################################################################################

'''
One Super ANN Architecture:
    Layer 1 = input layer of 90*90 dimension
    Layer 2 = 80 neurons with relu as activation function 
    Layer 3 = 60 neurons with relu as activation function
    Layer 4 = 60 neurons with relu as activation function
    Layer 5 = output layer with 2 neurons and softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.5
    optimizer = sgd
'''

def build_ann_One_Super():
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
    model = tflearn.DNN(network)
    return model

def Train_Ann_One_Super(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,90,90,1])
    input_test_d = input_test_d.reshape([-1,90,90,1])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_ann_One_Super()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_ANN_One_Super')

    MODEL.save(save_path)

#############################################################################################################

'''
BUZZ1 ANN Architecture:
    Layer 1 = input layer of 150*100 dimension
    Layer 2 = 250 neurons with relu as activation function 
    Layer 3 = 150 neurons with relu as activation function
    Layer 4 = output layer with 3 neurons and softmax as activation function
    
Parameters :
    number of epochs = 50
    batch size = 10
    trained using tflearn
    learning rate = 0.01
    optimizer = sgd
    regularizer = L2
'''

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

def Train_Ann_BUZZ1(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,150,100,1])
    input_test_d = input_test_d.reshape([-1,150,100,1])

    NUM_EPOCHS = 50
    BATCH_SIZE = 10
    MODEL = build_ann_BUZZ1()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_BUZZ1_ANN')

    MODEL.save(save_path)

#############################################################################################################

'''
BUZZ2 ANN Architecture:
    Layer 1 = input layer of 440*100 dimension
    Layer 2 = 100 neurons with relu as activation function 
    Layer 3 = 70 neurons with relu as activation function
    Layer 4 = output layer with 3 neurons and softmax as activation function
    
Parameters :
    number of epochs = 50
    batch size = 10
    trained using tflearn
    learning rate = 0.01
    optimizer = sgd
    regularizer = L2
'''

def build_ann_BUZZ2():
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
    model = tflearn.DNN(network)
    return model

def Train_Ann_BUZZ2(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,440,100,1])
    input_test_d = input_test_d.reshape([-1,440,100,1])

    NUM_EPOCHS = 50
    BATCH_SIZE = 10
    MODEL = build_ann_BUZZ1()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_BUZZ1_ANN')

    MODEL.save(save_path)

#############################################################################################################