 # -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:53:43 2019

@author: Yuxuan Long
"""

import struct
import matplotlib.pyplot as plt
import numpy as np

def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((image_size, num_images))
    for i in range(num_images):
        images[:, i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def createDataFiles():
    train_label_file = './mnist/train-labels.idx1-ubyte'
    train_image_file = './mnist/train-images.idx3-ubyte'
    test_label_file = './mnist/t10k-labels.idx1-ubyte'
    test_image_file = './mnist/t10k-images.idx3-ubyte'
    
    train_images = decode_idx3_ubyte(train_image_file) / 255.0
    train_labels = np.uint64(decode_idx1_ubyte(train_label_file))
    
    test_images = decode_idx3_ubyte(test_image_file) / 255.0
    test_labels = np.uint64(decode_idx1_ubyte(test_label_file))
    
    np.save('train_images', train_images)
    np.save('train_labels', train_labels)
    np.save('test_images', test_images)
    np.save('test_labels', test_labels)
    
def dataPreprocessing(train_images, test_images):
    # data pre-processing
    u = np.mean(train_images, axis = 1).reshape((train_images.shape[0], 1))
    return (train_images - u), (test_images - u)

class SimNet:
    def __init__(self, fan_out_list, num_layers, data_X, data_Y, test_images, 
                 test_labels, out_dim, lr, lam, batch_size, num_epoch):
        self.fan_out_list = fan_out_list
        self.num_layers = num_layers
        self.data_X = data_X
        self.data_Y = data_Y
        self.test_images = test_images
        self.test_labels = test_labels
        self.out_dim = out_dim
        self.x_dim = data_X.shape[0]
        self.N = data_X.shape[1]
        self.lr = lr
        self.lam = lam
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        
        
        dim_list = [self.x_dim] + fan_out_list + [out_dim]
        W_list = []
        b_list = []
        for i in range(num_layers + 1):
            # He initialization
            W_list.append(np.random.randn(dim_list[i], dim_list[i + 1]) / np.sqrt(dim_list[i]) / 2)
            b_list.append(np.zeros(dim_list[i + 1]))

        self.W_list = np.array(W_list)
        self.b_list = np.array(b_list)
    
    
    def ReLU(self, X):
        # parallel operation
        tem = X > 0
        return tem * X
    
    def gradReLU(self, Z, grad):
        return Z * grad
    
    def Sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def gradSigmoid(self, Z, grad):
        return (1 - Z) * Z * grad
    
    def Linear(self, X, num):
        # parallel operation
        W = self.W_list[num]
        b = self.b_list[num]
        Y = np.dot(W.T, X) + b.reshape((W.shape[1], 1))
        return Y
    
    def softmax(self, X):
        # parallel operation
        tem = np.exp(X)
        return tem / np.sum(tem, axis = 0)
    
    def sample(self):
        # sample batches of data for SGD
        
        v = np.random.rand(self.N)
        
        index = np.argsort(v)
        
        return index  
    
    def forward(self,X):
        # parallel operation
        Z_list = [X]
        for i in range(self.num_layers + 1):
            Z = self.Linear(Z_list[i], i)
            if i < self.num_layers:
                Z = self.ReLU(Z)
#            Z = self.ReLU(Z)
            Z_list.append(Z)
        P = self.softmax(Z)
        self.P = P
        self.Z_list = Z_list
        return P
    
    def backward(self, label, data_num):
        # parallel operation
        W_grad_list = []
        b_grad_list = []
        
        grad = self.P
        grad[label, range(data_num)] -= 1.0  # gradient for softmax
        # equivalent to subtracting the ground truth label matrix
        all_b_grad = grad
        
        for i in range(self.num_layers + 1):
            Z = self.Z_list[-i - 1]
            X = self.Z_list[-i - 2]
            
            if i > 0:
                all_b_grad = self.gradReLU(Z, grad)  # dL / dz
            
            W_grad = np.dot(X, all_b_grad.T)
            b_grad = np.sum(all_b_grad, axis = 1)
            if i < self.num_layers:
                grad = np.dot(self.W_list[-i - 1], all_b_grad)  # w.r.t input
                
            
            W_grad_list = [W_grad] + W_grad_list
            b_grad_list = [b_grad] + b_grad_list
        return np.array(W_grad_list), np.array(b_grad_list)

    def update(self):
        # compute final gradient and update
        # collect loss
        
        m_W = 0.0
        v_W = 0.0
        m_b = 0.0
        v_b = 0.0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        beta1_t = beta1
        beta2_t = beta2
        
        loss_list = []
        for i in range(self.num_epoch):
            print('Epoch ' + str(i) + '\n')
            index = self.sample()
            for j in range(0, self.N, self.batch_size): 
                if (j + self.batch_size) <= self.N:
                    batch_x = self.data_X[:, index[j : (j + self.batch_size)]]
                    batch_label = self.data_Y[index[j : (j + self.batch_size)]]
                    num = self.batch_size
                else:
                    batch_x = self.data_X[:, index[j : self.N]]
                    batch_label = self.data_Y[index[j : self.N]]
                    num = self.N - j
                
                
#                print('Batch ' + str(j) + '\n')
                

                P = self.forward(batch_x)
                 # we do not display loss that contains regularization term
                loss = -np.sum(np.log(P[batch_label, range(num)])) / num
                              
                W_grad, b_grad = self.backward(batch_label, num)
                    
                W_grad = W_grad / num + self.lam * self.W_list
                b_grad /= num
                
                # update with l2 regularization (weight decay)
                # Adam update rule
                
                m_W, v_W, W_grad = self.adamUpdate(m_W, v_W, beta1, beta2, epsilon, W_grad, beta1_t, beta2_t)
                m_b, v_b, b_grad = self.adamUpdate(m_b, v_b, beta1, beta2, epsilon, b_grad, beta1_t, beta2_t)
                beta1_t *= beta1
                beta2_t *= beta2
                self.W_list -= self.lr * W_grad
                self.b_list -= self.lr * b_grad
                
#                print(loss)
                loss_list.append(loss)
                
#            # learning rate decay
#            self.lr *= 0.95
        return loss_list
        
    def adamUpdate(self, m, v, beta1, beta2, epsilon, grad, beta1_t, beta2_t):
        new_m = beta1 * m + (1 - beta1) * grad
        new_v = beta2 * v + (1 - beta2) * (grad ** 2)
        new_grad = new_m / (1 - beta1_t) / ((new_v / (1 - beta2_t)) ** 0.5 + epsilon)
        return new_m, new_v, new_grad
    
    def test(self):
        # test the model
        P = self.forward(self.test_images)
        return np.sum(np.argmax(P, axis = 0) == self.test_labels) / float(len(self.test_labels))

if __name__ == '__main__':
    "simple neural network"
#    createDataFiles()

    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_labels.npy')
    
    train_images, test_images = dataPreprocessing(train_images, test_images)
#    
    fan_out_list = [300]
    lr = 0.001
    lam = 0.0005
    batch_size = 100
    num_epoch = 10
    
    num_layers = len(fan_out_list)
    out_dim = 10
    
    inst = SimNet(fan_out_list, num_layers, train_images, train_labels, test_images, 
                  test_labels, out_dim, lr, lam, batch_size, num_epoch)
    loss_list = inst.update()
    accuracy = inst.test()
    print(accuracy)