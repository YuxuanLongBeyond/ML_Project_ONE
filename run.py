#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:48:20 2019

@author: YuxuanLong
"""
import numpy as np
import implementations as imp
import feature_processing as feat
import fast_simple_net as sim

SEED = 0
np.random.seed(SEED)

def extract_train_data(index, data):
    """
    Read the training data.
    Crop the redundant feature (-999s) from original data, by using the index.
    
    Return:
        x - the features
        y - labels
    """
    out = np.delete(data, index, axis = 1)
    y = out[:, -1]
    x = out[:, 0:-1]
    dim = x.shape[1]
    return x, y, dim

def data_whitening(x, epsilon = 1e-9):
    """
    Perform ZCA for data whitening on the features.
    
    Return:
        M       the linear transformation matrix
        mean    the mean of the feature
    """
    mean = np.mean(x, axis = 0)
    x_norm = x - mean
    sigma = np.dot(x_norm.T, x_norm) / x.shape[0]
    u, V = np.linalg.eig(sigma)
    M = np.dot(V / np.sqrt(u + epsilon), V.T)
    return M, mean

def build_poly(x, degree = 1):
    X = np.ones((x.shape[0], 1))
    for i in range(degree):
        X = np.concatenate((X, x ** (i + 1)), axis = 1)
    return X


def train_test(data_list, test_interval, val_num, test_list, whitening = True,
               method = 'ls', name_list = ['A', 'B', 'AB', 'BC', 'ABC', 'D'],
               max_iters = 1000, gamma = 0.01, lambda_ = 0.001, epsilon = 1e-9, 
               fan_out_list = [25, 10], out_dim = 2, lr = 0.001, lam = 0.0005, 
               batch_size = 100, num_epoch = 100):
    """
    Train the model and test the accuracy.
    Note that there are 6 models to be trained that deal with 6 types of data.
    Return: 
        accuracy_list: list of accuracies for the models
        loss_list: collection of final losses. 
                    (for neural network, we collect all the losses)
                    
        recall_list: collection of recalls (only for neural net)
        precision_list: collection of precision (only for neural net)
    
    
    #######Parameters######
    data_list       collect different type of data in a list
    test_interval   collect indices which indicate the range of data being trained
    val_num         the number of the data subset in k-fold, note 0 <= val_num <= k - 1
    test_list       collect indices of redundant feature of all data types
    whitening       a boolean value for data whitening
    method          a string that is either 'log', 'ls' or 'dl' 
                        - 'log': logistic regression
                        - 'ls': least squares (or ridge regression if lambda_ > 0)
                        - 'dl': deep learning method (neural network)
    name_list       collect all names of the data type
    max_iters       maximum iterations for logistic regression
    gamma           step size for each iteration in logistic regression
    lambda_         parameter for l2 regularization in least squares and logistic regression
    epsilon         parameter for ZCA data whitening, should be a small positive
    fan_out_list    the list that collects the number of neurons in hidden layer of the neural network
    out_dim         output dimension at last layer in the neural network (before softmax)
    lr              learning rate when optimizing the neural network
    lam             parameter for weight decay (l2 regularization)
    batch_size      batch size for stochastic gradient descent in neural network
    num_epoch       number of epochs when optimizing the neural network
    """
    
    # collect all parameters such as data whitening and weights
    W_collection = []
    b_collection = []
    M_list = []
    mean_list = []
    accuracy_list = []
    
    w_list = []
    loss_list = []
    precision_list = []
    recall_list = []
    # iterate through all data types, train and test each model on a specific method
    for i in range(len(name_list)):
        
        x, y, dim = extract_train_data(test_list[i], data_list[i])
        
        # only use some part of data for training, rest is for testing
        i1 = test_interval[i][0]
        i2 = test_interval[i][1]
        index = list(range(0, i1)) + list(range(i2, len(y)))
        x_tr = x[index, :]
        y_tr = y[index]
        x_tst = x[i1 : i2, :]
        y_tst = y[i1 : i2]     
        
        
        
        # print('Dummy accuracy is ', np.sum(y_tst == -1) / len(y_tst))
        
        # we use training data to obtain transformation
        if whitening:
            M, mean = data_whitening(x, epsilon) 
            x_tr = np.dot(x_tr - mean, M)
            x_tst = np.dot(x_tst - mean, M)
            
            M_list.append(M)
            mean_list.append(mean)
        
        x_tr = build_poly(x_tr)
        x_tst = build_poly(x_tst)
        print(x_tr.shape[1])
          
        if method == 'ls':
            # least squares / ridge regression
            w, loss = imp.ridge_regression(y_tr, x_tr, lambda_)
            
#            w = np.dot(x_tr.T, y_tr) / lambda_
#            loss = 0
            
            accuracy = imp.evaluate(w, x_tst, y_tst)
            accuracy_list.append(accuracy)
            w_list.append(w)
            
        elif method == 'log':
            # logistic regression
            initial_w = np.random.rand(dim + 1)
            w, loss = imp.reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
            accuracy = imp.evaluate(w, x_tst, y_tst)
            accuracy_list.append(accuracy)
            w_list.append(w)
        elif method == 'dl':
            # deep learning method
            y_tr = y_tr.astype(np.int8)
            y_tr[y_tr == -1] = 0
            y_tst = y_tst.astype(np.int8)
            y_tst[y_tst == -1] = 0
            inst = sim.SimNet(fan_out_list, x_tr[:, 1:].T, y_tr, 
                              out_dim, lr, lam, batch_size, num_epoch)
            loss = inst.optimize()
            accuracy, precision, recall= inst.test(x_tst[:, 1:].T, y_tst)
            recall_list.append(recall)
            precision_list.append(precision)
            W_collection.append(inst.W_list)
            b_collection.append(inst.b_list)
            accuracy_list.append(accuracy)
        else:
            raise ValueError
        loss_list.append(loss)
        
        print('For training data ', name_list[i], ', the average accuracy is: ', accuracy, '\n')
    
    # Save all parameters
    if whitening:
        np.save('./parameters/data_whitening/mean_list_val' + str(val_num), np.array(mean_list))
        np.save('./parameters/data_whitening/M_list_val' + str(val_num), np.array(M_list))
    if method == 'dl':
        np.save('./parameters/neural_net/W_collection_dl_val' + str(val_num), np.array(W_collection))
        np.save('./parameters/neural_net/b_collection_dl_val' + str(val_num), np.array(b_collection))
    elif method == 'ls':
        np.save('./parameters/ridge/w_' + method+ '_val' + str(val_num), np.array(w_list))
    elif method == 'log':
        np.save('./parameters/logistic/w_' + method+ '_val' + str(val_num), np.array(w_list))
    return accuracy_list, precision_list, recall_list, loss_list


if __name__ == '__main__':
    
    train_validate = False # False
    final_test = True # True
    
    # Select a ML method to build models, e.g. logistic regression
    method = 'dl' # 'ls', 'log', 'dl'
    
    ### Set up hyper-parameters
    ### Those parameters are explained in 'train_test' function above
    k_fold = 5 # for k-fold cross validation
    
    # data whitening
    whitening = True
    epsilon = 1e-9 # 1e-9
    
    # parameters for logistic regression
    max_iters = 100
    gamma = 1.0
    lambda_log = 0.00001
    
    # parameter for ridge regression 
    lambda_ls = 0.001 # 0.001

    # parameters for neural network
    fan_out_list = [25, 10] # [25, 10]
    lr = 0.0001 # 0.0001
    lam = 0.001 # 0.001
    batch_size = 100 # 100
    num_epoch = 300 # 300
    out_dim = 2 # 2 n (binary classifier)
    
    
    
    
    if method == 'ls':
        lambda_ = lambda_ls
    elif method == 'log':
        lambda_ = lambda_log
    elif method == 'dl':
        # set a dummy lambda
        lambda_ = 0.001
    else:
        print('The method is not available')
        raise ValueError
    
    # collect the number of training data points in each data type
    data_num_list = np.array([4429, 69982, 7562, 73790, 26123, 68114])    
    index_set = []
    for num in data_num_list:
        l = np.round(np.linspace(0, num, k_fold + 1))
        index = np.array([[l[i], l[i + 1]] for i in range(k_fold)])
        index_set.append(index)
    index_set = np.array(index_set).astype(np.int32)

    # we roughly divide data into 6 types
    # A, B, C all refer to feature ids having 999s
    # note id starts from 0
    A = [0]
    B = [4, 5, 6, 12, 26, 27, 28]
    C = [23, 24, 25]
    AB = A + B
    BC = B + C
    ABC = A + B + C
    D = []
    test_list = [A, B, AB, BC, ABC, D]
    name_list = ['A', 'B', 'AB', 'BC', 'ABC', 'D']

   
    
    train = False

    if train or train_validate:
        data_A = np.load('./train_data/data_A.npy')
        data_B = np.load('./train_data/data_B.npy')
        data_AB = np.load('./train_data/data_AB.npy')
        data_BC = np.load('./train_data/data_BC.npy')
        data_ABC = np.load('./train_data/data_ABC.npy')
        data_D = np.load('./train_data/data_D.npy')
        data_list = [data_A, data_B, data_AB, data_BC, data_ABC, data_D]     
    
    # select the first interval for simple run of training, just for test
    val_num = 0
    test_interval = index_set[:, val_num, :]
    if train:
        accuracy_list, precision_list, recall_list, loss_list = train_test(data_list, test_interval, val_num, test_list, whitening,
                                   method, name_list, max_iters, 
                                   gamma, lambda_ , epsilon, 
                                   fan_out_list, out_dim, lr, lam, 
                                   batch_size, num_epoch)

    # perform k-fold cross-validation
    # meanwhile produce k models for each data type
    # this means we have k * 6 models to do classification
    validate_set = [] # collect accuracy for each validation set
    if train_validate:
        for val_num in range(k_fold):
            test_interval = index_set[:, val_num, :]
            accuracy_list, precision_list, recall_list, loss_list= train_test(data_list, test_interval, val_num, test_list, whitening,
                                       method, name_list, max_iters, 
                                       gamma, lambda_ , epsilon, 
                                       fan_out_list, out_dim, lr, lam, 
                                       batch_size, num_epoch)
            validate_set.append(accuracy_list)
    
    # test and produce the submission file
    if final_test:
        test_file = './data/test.csv'
        test_data = np.genfromtxt(test_file, delimiter = ',', dtype = 'U')
        
        data = test_data[1:]
        ids = data[:, 0]
        N = len(ids)
        features = np.array(list(data[:, 2:]), dtype = float)
        
        mask_999 = (features == -999.0)

        # collect prediction from each model
        prediction_set = []
        for val_num in range(k_fold):
            M_list = np.load('./parameters/data_whitening/M_list_val' + str(val_num) + '.npy')
            mean_list = np.load('./parameters/data_whitening/mean_list_val' + str(val_num) + '.npy')
            if method == 'dl':
                W_collection = np.load('./parameters/neural_net/W_collection_dl_val' + str(val_num) + '.npy')
                b_collection = np.load('./parameters/neural_net/b_collection_dl_val' + str(val_num) + '.npy')  
            elif method == 'ls':
                w_list = np.load('./parameters/ridge/w_' + method+ '_val' + str(val_num) + '.npy')
            elif method == 'log':
                w_list = np.load('./parameters/logistic/w_' + method+ '_val' + str(val_num) + '.npy')
            
            
            prediction = np.zeros(N, np.int8)
            for i in range(6):
                mask, _ = feat.cal_mask(mask_999, test_list[i])
                x = np.delete(features, test_list[i], axis = 1)
                x = x[mask, :]
                if whitening:
                    x = np.dot(x - mean_list[i], M_list[i])
    
#                x = np.concatenate((np.ones((x.shape[0],1)), x), axis = 1)
            
                x = build_poly(x)

                if method == 'dl':
                    # pass a dummy x and y to SimNet
                    dummy = np.zeros((2, 2))
                    inst = sim.SimNet(fan_out_list, dummy, dummy, 
                                      out_dim, lr, lam, batch_size, num_epoch)
                    inst.W_list = W_collection[i]
                    inst.b_list = b_collection[i]
                    y = np.argmax(inst.forward(x[:, 1:].T), axis = 0).astype(np.int8)
                    y[y == 0] = -1
                else:
                    y = np.dot(x, w_list[i])
                    y[y > 0] = 1
                    y[y < 0] = -1
                prediction[mask] = y
            prediction_set.append(prediction)
            
        # vote for the final prediction among all models
        prediction_set = np.array(prediction_set)
        pred_mask = np.sum(prediction_set, axis = 0) > 0
        final_prediction = pred_mask.astype(np.int8)
        final_prediction[final_prediction == 0] = -1
        
        submit = np.array([['Id', 'Prediction']])
        tem = np.stack((ids, final_prediction)).T
        submit = np.concatenate((submit, tem), axis = 0)
        np.savetxt("submission.csv", submit, delimiter=",", fmt='%s')
        
