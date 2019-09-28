#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 08:16:56 2019

@author: YuxuanLong
"""
import numpy as np
import implementations as imp


if __name__ == '__main__':
#    A = [0]
#    B = [4, 5, 6, 12, 26, 27, 28]
#    C = [23, 24, 25]
#    AB = A + B
#    AC = A + C
#    BC = B + C
#    ABC = A + B + C
#    D = []
#    
#    index = ABC
#    w = np.load('./w_ABC.npy')
#    data = np.load('./train_data/data_B.npy')
#    
#    out = np.delete(data, index, axis = 1)
#    y = out[:, -1]
#    x = out[:, 0:-1]
#    dim = x.shape[1]
#    x = np.dot(x - mean, M)
#    x = np.concatenate((np.ones((x.shape[0],1)), x), axis = 1)
#    
#    accuracy = imp.evaluate(w, x, y)
#    print(accuracy)
    x = features
    y = labels
    features[mask_999] = 0.0
    train_num = 200000
    
    x_tr = x[0:train_num, :]
    y_tr = y[0:train_num]
    x_tst = x[train_num:, :]
    y_tst = y[train_num:]     
    
    fan_out_list = [30, 10]
    lr = 0.001
    lam = 0.0005
    batch_size = 200
    num_epoch = 200
    out_dim = 2 
    
    
    y_tr = y_tr.astype(np.int8)
    y_tr[y_tr == -1] = 0
    y_tst = y_tst.astype(np.int8)
    y_tst[y_tst == -1] = 0

    inst = sim.SimNet(fan_out_list, x_tr[:, 1:].T, y_tr, 
                      out_dim, lr, lam, batch_size, num_epoch)
    inst.optimize()
    accuracy = inst.test(x_tst[:, 1:].T, y_tst)    