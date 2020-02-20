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
    with open(filename, 'rb') as fp:
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
BEE1 CNN Architecture:
    Layer 1 = input layer of 32*32*3 dimension
    Layer 2 = Convolution layer with 20 filters and filter size of 5 , with relu as activation function 
    Layer 3 = pooling layer with down sampling of 2
    Layer 4 = Convolution layer with 40 filters and filter size of 3 , with relu as activation function
    Layer 5 = pooling layer with down sampling of 2
    Layer 6 = Fully connected Layer with 100 neurons, with relu as activation function
    Layer 7 = Output Fully connected Layer with 2 neurons, with softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.01
    optimizer = sgd
'''

def build_CNN_BEE1():
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
    model = tflearn.DNN(network)
    return model

def Train_CNN_BEE1(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,32,32,3])
    input_test_d = input_test_d.reshape([-1,32,32,3])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_CNN_BEE1()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_Bee1_CNN')

    MODEL.save(save_path)

##############################################################################################################

'''
BEE2 One Super CNN Architecture:
    Layer 1 = input layer of 90*90*3 dimension
    Layer 2 = Convolution layer with 60 filters and filter size of 5 , with tanh as activation function 
    Layer 3 = pooling layer with down sampling of 2
    Layer 4 = Convolution layer with 40 filters and filter size of 3 , with tanh as activation function
    Layer 5 = pooling layer with down sampling of 2
    Layer 6 = Fully connected Layer with 100 neurons, with relu as activation function
    Layer 7 = Output Fully connected Layer with 2 neurons, with softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.05
    optimizer = sgd
'''

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

def Train_CNN_ONE_SUPER(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,90,90,3])
    input_test_d = input_test_d.reshape([-1,90,90,3])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_CNN_one_super()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_One_Super_CNN')

    MODEL.save(save_path)

##############################################################################################################

'''
BEE2 Two Super CNN Architecture:
    Layer 1 = input layer of 90*90*3 dimension
    Layer 2 = Convolution layer with 60 filters and filter size of 5 , with tanh as activation function 
    Layer 3 = pooling layer with down sampling of 2
    Layer 4 = Convolution layer with 40 filters and filter size of 3 , with tanh as activation function
    Layer 5 = pooling layer with down sampling of 2
    Layer 6 = Fully connected Layer with 100 neurons, with relu as activation function
    Layer 7 = Output Fully connected Layer with 2 neurons, with softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.05
    optimizer = sgd
'''

def build_CNN_two_super():
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

def Train_CNN_TWO_SUPER(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,90,90,3])
    input_test_d = input_test_d.reshape([-1,90,90,3])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_CNN_two_super()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_Two_Super_CNN')

    MODEL.save(save_path)

##############################################################################################################

'''
BUZZ1 CNN Architecture:
    Layer 1 = input layer of 150*100*1 dimension
    Layer 2 = Convolution layer with 50 filters and filter size of 5 , with relu as activation function 
    Layer 3 = pooling layer with down sampling of 2
    Layer 4 = Convolution layer with 40 filters and filter size of 3 , with relu as activation function
    Layer 5 = pooling layer with down sampling of 2
    Layer 6 = Convolution layer with 40 filters and filter size of 3 , with relu as activation function
    Layer 7 = pooling layer with down sampling of 2
    Layer 8 = Fully connected Layer with 100 neurons, with relu as activation function
    Layer 9 = Output Fully connected Layer with 3 neurons, with softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.01
    optimizer = sgd
'''

def build_CNN_BUZZ1():
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
    model = tflearn.DNN(network)
    return model

def Train_CNN_BUZZ1(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,150,100,1])
    input_test_d = input_test_d.reshape([-1,150,100,1])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_CNN_BUZZ1()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_CNN_BUZZ1')

    MODEL.save(save_path)

##############################################################################################################

'''
BUZZ2 CNN Architecture:
    Layer 1 = input layer of 150*100*1 dimension
    Layer 2 = Convolution layer with 70 filters and filter size of 5 , with relu as activation function 
    Layer 3 = pooling layer with down sampling of 2
    Layer 4 = Convolution layer with 50 filters and filter size of 5 , with relu as activation function
    Layer 5 = pooling layer with down sampling of 2
    Layer 6 = Convolution layer with 50 filters and filter size of 3 , with relu as activation function
    Layer 7 = pooling layer with down sampling of 2
    Layer 8 = Fully connected Layer with 100 neurons, with relu as activation function
    Layer 9 = Output Fully connected Layer with 3 neurons, with softmax as activation function
    
Parameters :
    number of epochs = 30
    batch size = 10
    trained using tflearn
    learning rate = 0.05
    optimizer = sgd
'''

def build_CNN_BUZZ2():
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

def Train_CNN_BUZZ2(train_path,test_path,valid_path,save_path):
    train_d, test_d, valid_d = get_training_testing_validation_data(train_path,test_path,valid_path)

    input_train_d, label_train_d = get_input_data(train_d)
    input_test_d, label_test_d = get_input_data(test_d)
    input_valid_d, label_valid_d = get_input_data(valid_d)

    input_train_d = input_train_d.reshape([-1,150,100,1])
    input_test_d = input_test_d.reshape([-1,150,100,1])

    NUM_EPOCHS = 30
    BATCH_SIZE = 10
    MODEL = build_CNN_BUZZ1()
    MODEL.fit(input_train_d, label_train_d, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(input_test_d, label_test_d),
              show_metric=True,
              batch_size=BATCH_SIZE,
              run_id='Train_CNN_BUZZ2')

    MODEL.save(save_path)

##############################################################################################################