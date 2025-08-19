from typing import Collection
import numpy as np 

def sigmoid(x: float)->float:
    return 1 / (1 + np.exp(-x))


def softmax_pure(x: Collection) -> Collection:
    '''
    纯正的softmax函数可能出现数值溢出的问题，如e的10次方，会超出数值上限
    '''
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def softmax(x: Collection)->Collection:
    '''进行恒等变形，消除数值溢出问题'''
    max_a = np.max(x)
    exp_a = np.exp(x - max_a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def mean_squared_error(y: "numpy.ndarray", t:"numpy.ndarray") -> float:
    return 0.5 * sum((y - t) ** 2)