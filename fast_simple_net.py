 # -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:53:43 2019

@author: Yuxuan Long
"""

import numpy as np


class SimNet:
    def __init__(self, fan_out_list, data_X, data_Y, 
                 out_dim, lr, lam, batch_size, num_epoch):
        
        """
        A super simple neural network model, fully connected
        Only include linear layer, activations and softmax

        #######Parameters######
        fan_out_list    the list that collects the number of neurons in hidden layer of the neural network
        data_X          training data, a matrix with feature as its columns
        data_Y          labels for training data
        out_dim         output dimension at last layer in the neural network (before softmax)
        lr              learning rate when optimizing the neural network
        lam             parameter for weight decay (l2 regularization)
        batch_size      batch size for stochastic gradient descent in neural network
        num_epoch       number of epochs when optimizing the neural network
        """        
        
        self.fan_out_list = fan_out_list
        self.num_layers = len(fan_out_list)
        self.data_X = data_X
        self.data_Y = data_Y
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
        for i in range(self.num_layers + 1):
            # He's initialization
            W_list.append(np.random.randn(dim_list[i], dim_list[i + 1]) * np.sqrt(2.0 / dim_list[i]))
            b_list.append(np.zeros(dim_list[i + 1]))
        self.W_list = np.array(W_list)
        self.b_list = np.array(b_list)
        
    
    
    def ReLU(self, X):
        """
        ReLU activation function
        """
        tem = X > 0
        return tem * X
    
    def gradReLU(self, Z, grad):
        return (Z > 0) * grad
    
    def Sigmoid(self, X):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-X))
    
    def gradSigmoid(self, Z, grad):
        return (1 - Z) * Z * grad
    
    def Linear(self, X, num):
        """
        Linear layer
        """
        W = self.W_list[num]
        b = self.b_list[num]
        Y = np.dot(W.T, X) + b.reshape((W.shape[1], 1))
        return Y
    
    def softmax(self, X):
        tem = np.exp(X)
        return tem / np.sum(tem, axis = 0)
    
    def sample(self):
        """
        Generate random integers (not repetitive) for sample batch
        """
        
        v = np.random.rand(self.N)
        
        index = np.argsort(v)
        
        return index  
    
    def forward(self,X):
        """
        Forward the input batch through the network
        """
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
        """
        Back propagation and collect all gradients 
        """
        
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

    def optimize(self):
        """
        Iterate to update the weights by stochastic gradient descent
        Softmax loss is also computed
        """
        
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
                    
                # update with l2 regularization (weight decay)
                W_grad = W_grad / num + self.lam * self.W_list
                b_grad /= num + + self.lam * self.b_list
                
                
                # Adam update rule
                m_W, v_W, W_grad = self.adamUpdate(m_W, v_W, beta1, beta2, epsilon, W_grad, beta1_t, beta2_t)
                m_b, v_b, b_grad = self.adamUpdate(m_b, v_b, beta1, beta2, epsilon, b_grad, beta1_t, beta2_t)
                beta1_t *= beta1
                beta2_t *= beta2
                self.W_list -= self.lr * W_grad
                self.b_list -= self.lr * b_grad
                
#                print(loss)
                loss_list.append(loss)
                
        return loss_list
        
    def adamUpdate(self, m, v, beta1, beta2, epsilon, grad, beta1_t, beta2_t):
        """
        Adam updatem include first and second momentum
        """
        new_m = beta1 * m + (1 - beta1) * grad
        new_v = beta2 * v + (1 - beta2) * (grad ** 2)
        new_grad = new_m / (1 - beta1_t) / ((new_v / (1 - beta2_t)) ** 0.5 + epsilon)
        return new_m, new_v, new_grad
    
    def test(self, test_images, test_labels):
        """
        test the model if test data is given
        """
        P = self.forward(test_images)
        return np.sum(np.argmax(P, axis = 0) == test_labels) / float(len(test_labels))
