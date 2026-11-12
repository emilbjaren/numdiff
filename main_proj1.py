import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from evaluator import Evaluator
from lte import LTE
from rk34 import RK34
from rk4 import RK4

def verify_rk4():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
    solver = RK4(ivp, N = 100, adaptive=True)
    solver.integrate()
    plt.plot(solver.grid, solver.us)
    plt.show()

def error_rk4():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-2]]))
    evaluator = Evaluator(ivp, RK4)

    # Compute the errors over the grid:
    grid, errs = evaluator.errvstime(100)

    # Plot these errors with log-scale on the y-axis:
    plt.figure()
    plt.semilogy(grid, errs)
    plt.title("Error vs time (semilog)")

    # Compute the error at t = T for different h
    # corresponding to N = 2^3, 2^4, ..., 2^10:
    Ns = 2 ** np.arange(3, 11)
    hs, errs = evaluator.errvsh(Ns)

    # Plot these errors in a loglog-scale:
    plt.figure()
    plt.loglog(hs, errs)
    plt.title("Error vs h (log-log)") # should have lutning 4

    # Plot a comparison line which is O(h):
    plt.loglog(hs, hs * 2 * errs[-1] / hs[-1])
    plt.show()

def main():
    verify_rk4()
    error_rk4()

if __name__ == "__main__":
    main()