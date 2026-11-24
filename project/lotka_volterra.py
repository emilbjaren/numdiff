import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from ivp import IVP

class LotkaVolterra(IVP):
    def __init__(self, t0, T, y0, a, b, c, d):
        super().__init__(t0, T, y0)

        self.a = a 
        self.b = b
        self.c = c
        self.d = d

    def f(self, tn: np.number, un: np.array) -> np.array:
        return np.array([self.a * un[0] - self.b * un[0] * un[1], 
                           self.c * un[0] * un[1] - self.d * un[1]])

    def H(self, x, y):
        return self.c * x + self.b * y - self.d * np.log(x) - self.a * np.log(y)