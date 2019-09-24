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
#        d = np.linalg.solve(H, g)
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


if __name__ == '__main__':
    N = 200
    D = 10
    X = np.random.rand(N, D)
    max_iters = 200
    gamma = 0.001
    lambda_ = 0.001
    
    X[0:100, :] += 1.0
    
    y = np.concatenate((np.ones(100), -np.ones(100)), axis = 0)
    
    initial_w = np.random.rand(D + 1)
    tx = np.concatenate((np.ones((N,1)), X), axis = 1)

    # w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
#
#    w, loss = least_squares(y, tx)
#    print(loss)
#
#
#    w, loss = ridge_regression(y, tx, lambda_)
#    print(loss)
    
#    w, residuals, rank, s = np.linalg.lstsq(tx, y, rcond = None)
#    loss = compute_loss(y, tx, w, N)
#    print(loss)
#    inv = np.linalg.inv(np.dot(tx.T, tx))
#    w = np.dot(inv, np.dot(tx.T, y))
#    loss = compute_loss(y, tx, w, N)
#    print(loss)
#    w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
#    print(loss)
#    
    w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    print(loss)