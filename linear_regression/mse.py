import numpy as np

def MSE(y, Y):
    return np.mean((y - Y) ** 2)