from typing import Collection, Callable
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


def cross_entropy_error(y: "numpy.ndarray", t: "numpy.ndarray") -> float:
    delta = 1e-7 # 添加微小值，防止log(0)出现，趋于无穷小
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_v2(y: "numpy.ndarray", t: "numpy.ndarray") -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_v3(y: "numpy.ndarray", t: "numpy.ndarray") -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



def numerical_diff_pure(func: Callable, x0) -> float:
    '''
        函数func, 在x0这一点的导数
    '''

    h = 10e-50
    y_h = func(x0 + h)
    y = func(x0)
    diff = (y_h - y) / h
    return diff


def numerical_diff(f: Callable, x: float) -> float:
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f: Callable, x: "numpy.ndarray"):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad


def squard(x:float) -> float:
    return x ** 2


def two_squard(x1: float, x2: float)->float:
    return x1 ** 2 + x2 ** 2


def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    '''
        grad = (2x[0], 2x[1])
    '''
    return x[0]**2 + x[1]**2