import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt


class IVP():
    def __init__(self, t0: np.number, T: np.number, y0: np.array):
        self.t0 = t0
        self.T = T
        self.y0 = y0


    def f(self, t: np.number, y: np.array):
        raise NotImplementedError
