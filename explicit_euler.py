import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from solver import Solver

class ExplicitEuler(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        return un + self.h * self.ivp.f(tn, un)