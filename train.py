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
    mean = np.mean(x, axis = 0)
    x_norm = x - mean
    sigma = np.dot(x_norm.T, x_norm) / x.shape[0]
    u, V = np.linalg.eig(sigma)
    M = V / np.sqrt(u + epsilon)
    return M, mean



def ensemble_learning(max_iters, gamma, lambda_, dim, num_ensem, x_tr, y_tr, method = 'ls'):
    mask_pos = (y_tr == 1.0)
    mask_neg = ~mask_pos
    x_tr_pos = x_tr[mask_pos, :]
    x_tr_neg = x_tr[mask_neg, :]
    num_pos = np.sum(mask_pos)
    num_neg = np.sum(mask_neg)
    
    y_new = np.concatenate((np.ones(num_pos), -np.ones(num_pos)), axis = 0)
    
    w_mat = np.zeros((dim + 1, num_ensem))
    loss_list = []
    for i in range(num_ensem):
        index = np.random.randint(0, num_neg, num_pos)
        
        x_new = np.concatenate((x_tr_pos, x_tr_neg[index, :]), axis = 0)
    
        if method == 'ls':
            w, loss = imp.ridge_regression(y_new, x_new, lambda_)
        elif method == 'log':
            initial_w = np.random.rand(dim + 1)
            w, loss = imp.reg_logistic_regression(y_new, x_new, lambda_, initial_w, max_iters, gamma)
        else:
            raise ValueError
        w_mat[:, i] = w
        loss_list.append(loss)
    return w_mat, loss_list

def mode(y):
    mask = y > 0
    compare = (np.sum(mask, axis = 1) >= np.sum(~mask, axis = 1))

    result = np.zeros(y.shape[0])
    result[compare] = 1.0
    result[~compare] = -1.0
    return result

def test_ensemble(x_tst, y_tst, w_mat, num_ensem):
    y_hat = np.dot(x_tst, w_mat)
    y_hat = mode(y_hat)
    accuracy = np.sum(y_tst == y_hat) / len(y_tst)
    return accuracy

def train_test(test_list, whitening = True, ensemble = False, method = 'ls', 
               ratio = 0.8, num_ensem_list = [10, 2, 6, 2, 9, 2],
               name_list = ['A', 'B', 'AB', 'BC', 'ABC', 'D'],
               data_num_list = [4429, 69982, 7562, 73790, 26123, 68114],
               max_iters = 10000, gamma = 0.01, lambda_ = 0.001, epsilon = 1e-9, 
               fan_out_list = [30], out_dim = 2, lr = 0.001, lam = 0.0005, 
               batch_size = 100, num_epoch = 100):
        
    W_collection = []
    b_collection = []
    M_list = []
    mean_list = []
    data_A = np.load('./train_data/data_A.npy')
    data_B = np.load('./train_data/data_B.npy')
    data_AB = np.load('./train_data/data_AB.npy')
    data_BC = np.load('./train_data/data_BC.npy')
    data_ABC = np.load('./train_data/data_ABC.npy')
    data_D = np.load('./train_data/data_D.npy')
    data_list = [data_A, data_B, data_AB, data_BC, data_ABC, data_D]
    for i in range(6):
        
        train_num = int(data_num_list[i] * ratio)
        x, y, dim = extract_train_data(test_list[i], data_list[i])
        num_ensem = num_ensem_list[i]
        
        x_tr = x[0:train_num, :]
        y_tr = y[0:train_num]
        x_tst = x[train_num:, :]
        y_tst = y[train_num:]        
        
        # we use training data to obtain normalization matrix
    
        if whitening:
            M, mean = data_whitening(x, epsilon) 
            
            x_tr -= mean
            x_tst -= mean
#            x_tr = np.dot(x_tr - mean, M)
#            x_tst = np.dot(x_tst - mean, M)
            
            M_list.append(M)
            mean_list.append(mean)
        
        test_num = len(y_tst)
        x_tr = np.concatenate((np.ones((train_num,1)), x_tr), axis = 1)
        x_tst = np.concatenate((np.ones((test_num,1)), x_tst), axis = 1)
          
        if ensemble:

            w_mat, loss_list = ensemble_learning(max_iters, gamma, lambda_, dim, num_ensem, x_tr, y_tr, method)
            accuracy = test_ensemble(x_tst, y_tst, w_mat, num_ensem)
            
            np.save('./parameters/w_mat_' + name_list[i], w_mat)
        else:
            if method == 'ls':
                w, loss = imp.ridge_regression(y_tr, x_tr, lambda_)
                accuracy = imp.evaluate(w, x_tst, y_tst)
                np.save('./parameters/w_' + name_list[i], w)
            elif method == 'log':
                initial_w = np.random.rand(dim + 1)
                w, loss = imp.reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
                accuracy = imp.evaluate(w, x_tst, y_tst)
                np.save('./parameters/w_' + name_list[i], w)
            elif method == 'dl':
                
                y_tr = y_tr.astype(np.int8)
                y_tr[y_tr == -1] = 0
                y_tst = y_tst.astype(np.int8)
                y_tst[y_tst == -1] = 0

                inst = sim.SimNet(fan_out_list, x_tr[:, 1:].T, y_tr, 
                                  out_dim, lr, lam, batch_size, num_epoch)
                inst.optimize()
                accuracy = inst.test(x_tst[:, 1:].T, y_tst)
                W_collection.append(inst.W_list)
                b_collection.append(inst.b_list)
            else:
                raise ValueError

        print('For training data ', name_list[i], ', the average accuracy is: ', accuracy, '\n')
    return W_collection, b_collection, M_list, mean_list

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
    
    train = True
    final_test = True
    
    whitening = True
    ensemble = False
    method = 'dl' # 'ls', 'log', 'dl'
    ratio = 0.9 # 0.999
    max_iters = 1000
    gamma = 0.001
    lambda_ = 0.0001
    epsilon = 1e-9 # 1e-9
    num_ensem_list = np.array([10, 2, 6, 2, 9, 2]) * 10
    data_num_list = [4429, 69982, 7562, 73790, 26123, 68114]
    
    fan_out_list = [30, 15] # [30, 15]
    lr = 0.001 # 0.001
    lam = 0.0005 # 0.0005
    batch_size = 100 # 100
    num_epoch = 200 # 200
    out_dim = 2 #2
    
    if train:
        W_collection, b_collection, M_list, mean_list = train_test(test_list, whitening, ensemble, 
                                                           method, ratio, num_ensem_list, 
                                                           name_list, data_num_list, max_iters, 
                                                           gamma, lambda_ , epsilon, 
                                                           fan_out_list, out_dim, lr, lam, 
                                                batch_size, num_epoch)
        np.save('./parameters/W_collection', np.array(W_collection))
        np.save('./parameters/b_collection', np.array(b_collection))

    
    if final_test:
        test_file = '../project1_data/test.csv'
        test_data = np.genfromtxt(test_file, delimiter = ',', dtype = 'U')
        
        data = test_data[1:]
        ids = data[:, 0]
        N = len(ids)
        features = np.array(list(data[:, 2:]), dtype = float)
        
        mask_999 = (features == -999.0)
#        num_999 = np.sum(mask_999, axis = 0) / mask_999.shape[0]

        if not train:
            W_collection = np.load('./parameters/W_collection.npy')
            b_collection = np.load('./parameters/b_collection.npy')

        
        prediction = np.zeros(N, np.int8)
        for i in range(6):
            mask, _ = cal_mask(mask_999, test_list[i])
            x = np.delete(features, test_list[i], axis = 1)
            x = x[mask, :]
            if whitening:
#                M, mean = data_whitening(x, epsilon) 
#                x = np.dot(x - mean, M)
#                x = np.dot(x - mean_list[i], M_list[i])
                x = x - mean_list[i]

            x = np.concatenate((np.ones((x.shape[0],1)), x), axis = 1)

            
            if ensemble:
                # for ensemble learning, no dl method
                w = np.load('./parameters/w_mat_' + name_list[i] + '.npy')
                y = np.dot(x, w)
                prediction[mask] = mode(y)
            else:
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
                    w = np.load('./parameters/w_' + name_list[i] + '.npy')
                    y = np.dot(x, w)
                    y[y > 0] = 1
                    y[y < 0] = -1
                prediction[mask] = y

        # Id, Prediction (-1, 1)
        submit = np.array([['Id', 'Prediction']])
        tem = np.stack((ids, prediction)).T
        submit = np.concatenate((submit, tem), axis = 0)
        np.savetxt("submission.csv", submit, delimiter=",", fmt='%s')