import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from solver import Solver

class RK34(Solver):
    def __init__(self, ivp, N, tol=0.01, adaptive=False):
        super().__init__(ivp, N, tol, adaptive)

    def step(self, tn, un):
        Y1 = self.ivp.f(tn, un)
        Y2 = self.ivp.f(tn + self.h / 2, un + self.h * Y1 / 2)
        Y3 = self.ivp.f(tn + self.h / 2, un + self.h * Y2 / 2)
        Z3 = self.ivp.f(tn + self.h, un - self.h * Y1 + 2 * self.h + Y2)
        Y4 = self.ivp.f(tn + self.h, un + self.h * Y3)

        self.errestold = self.errest
        self.errest = self.h / 6 * (2 * Y2 + Z3 - 2*Y3 - Y4)

        return un + self. h / 6 * (Y1 + 2 * Y2 + 2 * Y3 + Y4)