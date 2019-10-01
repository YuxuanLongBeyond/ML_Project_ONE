#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:48:20 2019

@author: YuxuanLong
"""
import numpy as np
import implementations as imp
import fast_simple_net as sim

# Preprocess (reduce dimension of feature) and normalize data

# Select ML model to train, e.g. logistic regression
np.random.seed(0)

def cal_mask(mask_999, index):
    mask = np.all(mask_999[:, index], axis = 1) & (np.sum(mask_999, axis = 1) == len(index))
    return mask, np.sum(mask)

def extract_train_data(index, data):
    out = np.delete(data, index, axis = 1)
    y = out[:, -1]
    x = out[:, 0:-1]
    dim = x.shape[1]
    return x, y, dim

def data_whitening(x, epsilon = 1e-9):
    # method of ZCA
    mean = np.mean(x, axis = 0)
    x_norm = x - mean
    sigma = np.dot(x_norm.T, x_norm) / x.shape[0]
    u, V = np.linalg.eig(sigma)
    M = np.dot(V / np.sqrt(u + epsilon), V.T)
    return M, mean

def train_test(data_list, test_interval, val_num, test_list, whitening = True,
               method = 'ls', name_list = ['A', 'B', 'AB', 'BC', 'ABC', 'D'],
               max_iters = 1000, gamma = 0.01, lambda_ = 0.001, epsilon = 1e-9, 
               fan_out_list = [25, 10], out_dim = 2, lr = 0.001, lam = 0.0005, 
               batch_size = 100, num_epoch = 100):
        
    W_collection = []
    b_collection = []
    M_list = []
    mean_list = []
    accuracy_list = []
    
    w_list = []

    for i in range(len(name_list)):
        
        x, y, dim = extract_train_data(test_list[i], data_list[i])
        
        i1 = test_interval[i][0]
        i2 = test_interval[i][1]
        index = list(range(0, i1)) + list(range(i2, len(y)))
        x_tr = x[index, :]
        y_tr = y[index]
        x_tst = x[i1 : i2, :]
        y_tst = y[i1 : i2]        
        
        # we use training data to obtain normalization matrix
    
        if whitening:
            M, mean = data_whitening(x, epsilon) 
            
#            x_tr -= mean
#            x_tst -= mean
            x_tr = np.dot(x_tr - mean, M)
            x_tst = np.dot(x_tst - mean, M)
            
            M_list.append(M)
            mean_list.append(mean)
        
        x_tr = np.concatenate((np.ones((x_tr.shape[0],1)), x_tr), axis = 1)
        x_tst = np.concatenate((np.ones((x_tst.shape[0],1)), x_tst), axis = 1)
          
        if method == 'ls':
            w, loss = imp.ridge_regression(y_tr, x_tr, lambda_)
            accuracy = imp.evaluate(w, x_tst, y_tst)
            accuracy_list.append(accuracy)
            w_list.append(w)
            
        elif method == 'log':
            initial_w = np.random.rand(dim + 1)
            w, loss = imp.reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
            accuracy = imp.evaluate(w, x_tst, y_tst)
            accuracy_list.append(accuracy)
            w_list.append(w)
        elif method == 'dl':
            y_tr = y_tr.astype(np.int8)
            y_tr[y_tr == -1] = 0
            y_tst = y_tst.astype(np.int8)
            y_tst[y_tst == -1] = 0

            inst = sim.SimNet(fan_out_list, x_tr[:, 1:].T, y_tr, 
                              out_dim, lr, lam, batch_size, num_epoch)
            loss = inst.optimize()
            accuracy = inst.test(x_tst[:, 1:].T, y_tst)
            W_collection.append(inst.W_list)
            b_collection.append(inst.b_list)
            accuracy_list.append(accuracy)
        else:
            raise ValueError

        print('For training data ', name_list[i], ', the average accuracy is: ', accuracy, '\n')
    
    if whitening:
        np.save('./parameters/mean_list_val' + str(val_num), np.array(mean_list))
        np.save('./parameters/M_list_val' + str(val_num), np.array(M_list))
    if method == 'dl':
        np.save('./parameters/W_collection_dl_val' + str(val_num), np.array(W_collection))
        np.save('./parameters/b_collection_dl_val' + str(val_num), np.array(b_collection))
    else:
        np.save('./parameters/w_' + method+ '_val' + str(val_num), np.array(w_list))
    return accuracy_list


if __name__ == '__main__':
    # we roughly divide data into 8 types
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
    validate = True
    final_test = False
    
    whitening = True
    method = 'ls' # 'ls', 'log', 'dl'
    
    max_iters = 1000
    gamma = 0.1
    lambda_ = 0.0001
    epsilon = 1e-9 # 1e-9
    data_num_list = np.array([4429, 69982, 7562, 73790, 26123, 68114])
    
    fan_out_list = [25, 10] # [25, 10]
    lr = 0.0001 # 0.0001
    lam = 0.001 # 0.001
    batch_size = 100 # 100
    num_epoch = 300 # 300
    out_dim = 2 #2
    
    k_fold = 5
    index_set = []
    for num in data_num_list:
        l = np.round(np.linspace(0, num, k_fold + 1))
        index = np.array([[l[i], l[i + 1]] for i in range(k_fold)])
        index_set.append(index)
    index_set = np.array(index_set).astype(np.int32)
    
    # we select the first interval for simple test
    val_num = 0
    test_interval = index_set[:, val_num, :]
    
    data_A = np.load('./train_data/data_A.npy')
    data_B = np.load('./train_data/data_B.npy')
    data_AB = np.load('./train_data/data_AB.npy')
    data_BC = np.load('./train_data/data_BC.npy')
    data_ABC = np.load('./train_data/data_ABC.npy')
    data_D = np.load('./train_data/data_D.npy')
    data_list = [data_A, data_B, data_AB, data_BC, data_ABC, data_D]    
    
    if train:
        accuracy_list = train_test(data_list, test_interval, val_num, test_list, whitening,
                                   method, name_list, max_iters, 
                                   gamma, lambda_ , epsilon, 
                                   fan_out_list, out_dim, lr, lam, 
                                   batch_size, num_epoch)

    validate_set = []
    if validate:
        for val_num in range(k_fold):
            test_interval = index_set[:, val_num, :]
            accuracy_list = train_test(data_list, test_interval, val_num, test_list, whitening,
                                       method, name_list, max_iters, 
                                       gamma, lambda_ , epsilon, 
                                       fan_out_list, out_dim, lr, lam, 
                                       batch_size, num_epoch)
            validate_set.append(accuracy_list)
    
    if final_test:
        test_file = '../project1_data/test.csv'
        test_data = np.genfromtxt(test_file, delimiter = ',', dtype = 'U')
        
        data = test_data[1:]
        ids = data[:, 0]
        N = len(ids)
        features = np.array(list(data[:, 2:]), dtype = float)
        
        mask_999 = (features == -999.0)
#        num_999 = np.sum(mask_999, axis = 0) / mask_999.shape[0]



        prediction_set = []
        for val_num in range(k_fold):
            M_list = np.load('./parameters/M_list_val' + str(val_num) + '.npy')
            mean_list = np.load('./parameters/mean_list_val' + str(val_num) + '.npy')
            if method == 'dl':
                W_collection = np.load('./parameters/W_collection_dl_val' + str(val_num) + '.npy')
                b_collection = np.load('./parameters/b_collection_dl_val' + str(val_num) + '.npy')  
            else:
                w_list = np.load('./parameters/w_' + method+ '_val' + str(val_num) + '.npy')
            
            
            prediction = np.zeros(N, np.int8)
            for i in range(6):
                mask, _ = cal_mask(mask_999, test_list[i])
                x = np.delete(features, test_list[i], axis = 1)
                x = x[mask, :]
                if whitening:
                    x = np.dot(x - mean_list[i], M_list[i])
    
                x = np.concatenate((np.ones((x.shape[0],1)), x), axis = 1)
    

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
        prediction_set = np.array(prediction_set)
        pred_mask = np.sum(prediction_set, axis = 0) > 0
        final_prediction = pred_mask.astype(np.int8)
        final_prediction[final_prediction == 0] = -1
        # Id, Prediction (-1, 1)
        submit = np.array([['Id', 'Prediction']])
        tem = np.stack((ids, final_prediction)).T
        submit = np.concatenate((submit, tem), axis = 0)
        np.savetxt("submission.csv", submit, delimiter=",", fmt='%s')
        