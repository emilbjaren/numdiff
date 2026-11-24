import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from solver import Solver

class ImplicitEuler(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        M = np.eye(self.ivp.A.shape[0]) - self.h * self.ivp.A
        return np.linalg.solve(M, un)