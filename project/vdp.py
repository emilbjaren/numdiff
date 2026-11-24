import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from ivp import IVP

class VDP(IVP):
    def __init__(self, t0, T, y0, mu):
        super().__init__(t0, T, y0)
        self.mu = mu

    def f(self, tn, un) -> np.array:
        return np.array(
            [
                un[1],
                self.mu * (1 - un[0]**2) * un[1] - un[0],
            ]
        )
    