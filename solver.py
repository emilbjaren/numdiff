import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from ivp import IVP

class Solver():
    def __init__(self, ivp: IVP, N: int, tol=0.01, adaptive=False):
        self.ivp = ivp
        self.N = N
        self.tol = tol
        self.k = 1
        self.h = (ivp.T - ivp.t0) / N if not adaptive else self._calc_h0() # special shit
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1) if not adaptive else []

        self.errest = self.tol
        self.errestold = self.tol
        self.adaptive = adaptive

    def step(tn: np.number, un: np.array):
        raise NotImplementedError

    def newstep(self) -> None:
        term1 = (self.tol / self.errest)**(2 / self.k)
        term2 = (self.tol / self.errestold)**(-1 / (3 * self.k))

        self.h = term1 * term2 * self.h

    def integrate(self) -> None:
        if self.adaptive:
            self.integrate_adaptive()
        else:
            self.integrate_nonadaptive()

    def integrate_nonadaptive(self) -> None:
        # Skapar array med rätt antal platser från början
        # Lite knas kod eftersom vi måste göra den tomma arrayen på samma form som y0
        state_shape = np.shape(self.ivp.y0)
        self.us = np.empty((self.N + 1,) + state_shape, dtype=float)
        self.us[0] = self.ivp.y0
        for i, tn in enumerate(self.grid[1:], 1):
            un = self.step(tn, self.us[i-1])
            self.us[i] = un

    def integrate_adaptive(self) -> None:
        t = self.ivp.t0
        u = self.ivp.y0
        self.grid = [t]
        self.us = [u]

        while t < self.ivp.T:
            self.h = min(self.h, self.ivp.T - t) # Se till att vi inte går förbi T
            un = self.step(t, u)
            tn = min(t + self.h, self.ivp.T) # TODO: Behövs detta

            self.grid.append(tn)
            self.us.append(un)
            t, u = tn, un

            if t < self.ivp.T:
                self.newstep()
      

    def _calc_h0(self) -> float:
        top = np.abs(self.ivp.T - self.ivp.t0) * self.tol**(1 / self.k)
        bottom = 100 * (1 + scipy.linalg.norm(self.ivp.f(self.ivp.t0, self.ivp.y0)))

        return top / bottom

    