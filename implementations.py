#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:41:40 2019

@author: YuxuanLong
"""

import numpy as np


def solve(A, b, D, theta = 0.0001):
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.solve(A + np.eye(D) * theta, b)
    return w

def compute_ls_loss(y, tx, w, N):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.dot(tx, w)
    return np.dot(e, e) / (2 * N)
    # return np.linalg.norm(e, 1) / N
    
    
def compute_ls_gradient(tx, e, N):
    """Compute the gradient."""
    g = - np.dot(tx.T, e) / N
    return g
    # return - np.dot(tx.T, np.sign(e)) / N

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    N, D = tx.shape
    w = initial_w
    for i in range(max_iters):
        e = y - np.dot(tx, w)
        g = compute_ls_gradient(tx, e, N)
        w -= gamma * g
    
    loss = compute_ls_loss(y, tx, w, N)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    N, D = tx.shape
    w = initial_w
    for i in range(max_iters):
        n = np.random.randint(N)
        x = tx[n]
        y_ = y[n]
        e = y_ - np.dot(x, w)
        g = - e * x
        w -= gamma * g
        
    loss = compute_ls_loss(y, tx, w, N)
    return w, loss

    
def least_squares(y, tx):
    N, D = tx.shape
    
    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    
    w = solve(A, b, D)
    
    loss = compute_ls_loss(y, tx, w, N)
    return w, loss
    

def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    
    A = np.dot(tx.T, tx) + np.eye(D) * (2 * lambda_ * N)
    b = np.dot(tx.T, y)
    w = solve(A, b, D)
    
    loss = compute_ls_loss(y, tx, w, N) + lambda_ * np.dot(w, w)
    return w, loss


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_log_loss(y, tx, w):
    s = sigmoid(np.dot(tx, w))
    loss = -np.dot(y, np.dot(tx, w)) - np.sum(np.log(1 - s))
    return loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    N, D = tx.shape
    w = initial_w
    for i in range(max_iters):
        s = sigmoid(np.dot(tx, w))
        g = np.dot(tx.T, s - y)
        H = np.dot(tx.T * ((1 - s) * s), tx)
        d = solve(H, g, D)
        w -= gamma * d
    loss = compute_log_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    N, D = tx.shape
    w = initial_w
    for i in range(max_iters):
        s = sigmoid(np.dot(tx, w))
        g = np.dot(tx.T, s - y) + lambda_ * w
        H = np.dot(tx.T * ((1 - s) * s), tx) + lambda_ * np.eye(D)
        d = solve(H, g, D)
        w -= gamma * d
    loss = compute_log_loss(y, tx, w) - lambda_ * np.dot(w, w) / 2.0
    return w, loss

def evaluate(w, tx_test, y_test):
    y_hat = np.dot(tx_test, w)
    
    y_hat[y_hat > 0] = 1
    y_hat[y_hat < 0] = -1
    
    accuracy = np.sum(y_test == y_hat) / len(y_test)
    return accuracy
    

if __name__ == '__main__':
    N = 600
    D = 10
    
    max_iters = 500
    gamma = 0.001
    lambda_ = 0.001
    
    
    # random simple train and test
    X = np.random.rand(N, D)
    X[0:(N // 2), :] += 1.0
    tx = np.concatenate((np.ones((N,1)), X), axis = 1)
    y = np.concatenate((np.ones(N // 2), -np.ones(N // 2)), axis = 0)
    
    tx_test = np.random.rand(1000, 10)
    tx_test[0:500, :] += 1.0
    tx_test = np.concatenate((np.ones((1000,1)), tx_test), axis = 1)
    y_test = np.concatenate((np.ones(500), -np.ones(500)), axis = 0)
    
    
    initial_w = np.random.rand(D + 1)
    
    
    w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    print(loss)

#    w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
#    print(loss)
#
#    w, loss = least_squares(y, tx)
#    print(loss)
#
#    w, loss = ridge_regression(y, tx, lambda_)
#    print(loss)
#    
#    w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
#    print(loss)
#    
#    w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
#    print(loss)
    
    
    accuracy = evaluate(w, tx_test, y_test)
    print(accuracy)