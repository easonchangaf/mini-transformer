from typing import Callable
import numpy as np


def numerical_gradient_old(f: Callable, x: "numpy.ndarray"):
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


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad


import numpy as np
from numba import njit, prange

@njit(parallel=True)  # 开启并行编译
def numerical_gradient_parallel(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    x_flat = x.ravel()  # 展平为1D数组，方便并行迭代
    grad_flat = grad.ravel()
    n = x_flat.size

    # 并行遍历所有元素（prange自动分配到多个线程）
    for i in prange(n):
        # 还原多维索引
        multi_idx = np.unravel_index(i, x.shape)
        tmp_val = x[multi_idx]

        # 计算f(x+h)
        x[multi_idx] = tmp_val + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[multi_idx] = tmp_val - h
        fxh2 = f(x)

        # 存储梯度
        grad_flat[i] = (fxh1 - fxh2) / (2 * h)

        # 还原值
        x[multi_idx] = tmp_val

    return grad.reshape(x.shape)  # 恢复原形状