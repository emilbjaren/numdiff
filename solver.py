import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from ivp import IVP

class Solver():
    def __init__(self, ivp: IVP, N: int):
        self.ivp = ivp
        self.N = N
        self.h = (ivp.T - ivp.t0) / N
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)


    def step(tn: np.number, un: np.array):
        raise NotImplementedError

    def integrate(self) -> None:
        # Skapar array med rätt antal platser från början
        self.us = np.empty(self.N + 1, float)
        self.us.put(0, self.ivp.y0)
        for i, tn in enumerate(self.grid[1:], 1):
            un = self.step(tn, self.us[i-1])
            self.us.put(i, un)

    