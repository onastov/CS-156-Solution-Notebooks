#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:34:13 2019

@author: Ognen
"""

import numpy as np
import matplotlib.pyplot as plt

# dimension of input space d = 2
# number of data points = N
# data is shape (N, d+1) i.e. (N, 3)
def make_data_set(N):
    data = np.zeros([N,3])
    data[:,0] = np.ones([N,])
    data[:,1] = np.random.uniform(-1,1,N)
    data[:,2] = np.random.uniform(-1,1,N)
    return (data)

# the line defines the target function
# line is shape (2,)
def make_line():    
    x_1,y_1 = np.random.uniform(-1,1,2)
    x_2,y_2 = np.random.uniform(-1,1,2)
    slope = (y_1 - y_2) / (x_1 - x_2)
    b = y_2 - slope * x_2
    return (slope, b)

# y is shape (N,)    
def classify_input(line, data):
    slope, b = line
    y = np.sign(data[:,1]*slope + b - data[:,2])
    return (y)

def pla(data, y, maxits):  
    i = 0
    w = np.zeros(data.shape[1])
    while (i < maxits):
        y_c = np.sign(np.dot(data, w))
        if np.count_nonzero(y_c != y) == 0:
            break
        else:  # update perceptron with a misclassified point
            misclassified_indices = (y_c != y).nonzero()[0]
            ix = np.random.choice(misclassified_indices)
            w = w + data[ix,:]*y[ix]
        i += 1
    return (w, i)

# N_test should be sufficiently large (e.g. 1e6)
def classification_error(w, N_test, line):
    data = make_data_set(N_test)
    y = classify_input(line, data)
    y_c = np.sign(np.dot(data, w)) 
    error = np.count_nonzero(y_c != y) / N_test
    return (error)

def problems_7_to_10(N, trials=1000, maxits=1000, N_test=100000):
    iters = np.zeros(trials)
    errors = np.zeros(trials)
    trial = 1
    while (trial < trials):
        line = make_line()
        data = make_data_set(N)
        y = classify_input(line, data)
        w, i = pla(data, y, maxits)
        error = classification_error(w, N_test, line)
        iters[trial] = i
        errors[trial] = error
        trial += 1
    iter_mean = np.mean(iters)
    iter_std = np.std(iters)
    iter_median = np.median(iters)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_median = np.median(errors)
    print(f"iter: mean={iter_mean:.2f}, std={iter_std:.2f}, \
median={iter_median:.2f}")
    print(f"error: mean={error_mean:.2f}, std={error_std:.2f}, \
median={error_median:.2f}")
    return
    
# plot data, target fn, and final hypothesis
def plot_data(data, line, y, w):
    fig,ax = plt.subplots()
    plt.axis([-1,1,-1,1])
    plt.grid()
    slope,b = line
    for i in range(data.shape[0]):
        if y[i] == 1: 
            plt.plot(data[i,1], data[i,2], 'ro')
        else:
            plt.plot(data[i,1], data[i,2], 'bo')

    plt.plot([-1,1], [-slope+b, slope+b], 'k')
    plt.plot([-1,1], [(w[1] - w[0])/w[2], (-w[1] - w[0])/w[2]], 'g')
    ax.set_aspect('equal','box')
    return
