#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:08:41 2019

@author: YuxuanLong
"""
import numpy as np

"""
# feature engineering
# data filtering and selection
"""

def cal_mask(mask_999, index):
    """
    Generate mask for croping the redundant feature (-999s)
    """
    mask = np.all(mask_999[:, index], axis = 1) & (np.sum(mask_999, axis = 1) == len(index))
    return mask, np.sum(mask)

def save_data(features, labels, mask, file_name):
    """
    Given the mask, we crop the feature from some data of the same category
    """
    label = labels[mask]
    label = label.reshape((len(label), 1))
    data = np.concatenate((features[mask, :], label), axis = 1)
    np.save(file_name, data)

def processing():
    print('Loading the training data...')
    train_file = './data/train.csv'
    train_data = np.genfromtxt(train_file, delimiter = ',', dtype = 'U')
    print('Separate the data into 6 types and save them...')
    data = train_data[1:]
    
    labels = data[:, 1]
    labels[labels == 'b'] = -1.0
    labels[labels == 's'] = 1.0
    labels = np.array(list(labels), dtype = float)
    
    features = np.array(list(data[:, 2:]), dtype = float)
    
    mask_999 = (features == -999.0)
    num_999 = np.sum(mask_999, axis = 0) / mask_999.shape[0]
    
    # we roughly divide data into 8 types
    # A, B, C all refer to feature ids having some 999s
    # note id starts from 0
    A = [0]
    B = [4, 5, 6, 12, 26, 27, 28]
    C = [23, 24, 25]
    AB = A + B
    AC = A + C
    BC = B + C
    ABC = A + B + C
    # D type: no any 999s
    
    mask_A, A_num = cal_mask(mask_999, A) # 4429 ~ 571
    mask_B, B_num = cal_mask(mask_999, B) # 69982 ~ 27005
    mask_C, C_num = cal_mask(mask_999, C) # 0
    mask_AB, AB_num = cal_mask(mask_999, AB) # 7562 ~ 705
    mask_AC, AC_num = cal_mask(mask_999, AC) # 0
    mask_BC, BC_num = cal_mask(mask_999, BC) # 73790 ~ 23933
    mask_ABC, ABC_num = cal_mask(mask_999, ABC) # 26123 ~ 1559
    
    mask_D = np.all(~mask_999, axis = 1) # 68114 ~ 31894
    D_num = np.sum(mask_D)
    
#    print(A_num + B_num + C_num + AB_num + AC_num + BC_num + ABC_num + D_num)
    # There are two types that do not exist
    # In summary, 6 types of feature
    
    
    # we separate data and record it into 6 files
    # the numpy file is save as [features, labels]
    save_data(features, labels, mask_A, './train_data/data_A')
    save_data(features, labels, mask_B, './train_data/data_B')
    save_data(features, labels, mask_AB, './train_data/data_AB')
    save_data(features, labels, mask_BC, './train_data/data_BC')
    save_data(features, labels, mask_ABC, './train_data/data_ABC')
    save_data(features, labels, mask_D, './train_data/data_D')


    
if __name__ == '__main__':
    processing()
 