import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from ivp import IVP
from solver import Solver

class Evaluator():
    def __init__(self, ivp: IVP, solver_cls: type[Solver]):
        self.ivp = ivp
        self.solver_cls = solver_cls

    def errvstime(self, N: int) -> tuple[np.array, np.array]:
        """Function to calculate error over time. 

        Returns: two np arrays, one with the grid (x) and one with errors (y).
        
        """
        solver = self.solver_cls(self.ivp, N)
        solver.integrate()

        us = solver.us
        ys = np.array([self.ivp.exact(tn) for tn in solver.grid])

        err_list = list()
        for un, yn in zip(us, ys):
            err_list.append(scipy.linalg.norm(un - yn))

        errs = np.array(err_list)

        return solver.grid, errs

    def errvsh(self, Ns: np.array) -> tuple[np.array, np.array]:
        t = self.ivp.T - self.ivp.t0
        y_last = self.ivp.exact(self.ivp.T)
        errs = []
        hs = []

        for N in Ns:
            h = t / N
            solver = self.solver_cls(self.ivp, N)
            solver.integrate()
            u_last = solver.us[len(solver.us) - 1]
            err = scipy.linalg.norm(y_last - u_last)
            errs.append(err)
            hs.append(h)

        return hs, errs
            


