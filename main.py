import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from evaluator import Evaluator
from explicit_euler import ExplicitEuler
from implicit_euler import ImplicitEuler
from lte import LTE

def first_test():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
    solver = ExplicitEuler(ivp, N = 100)
    solver.integrate()
    # Plot the approximations { u_n } vs. the temporal
    # grid points { t_n }:
    plt.plot(solver.grid, solver.us)
    plt.show()

def evaluator_test():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
    evaluator = Evaluator(ivp, ExplicitEuler)

    # Varying time
    grid, errs = evaluator.errvstime(10, relative=True)
    plt.plot(grid, errs)
    plt.show()

    # Varying step size h
    grid, errs = evaluator.errvsh(np.array([10, 20, 25]), relative=True)
    plt.plot(grid, errs)
    plt.show()

def task71_72():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-2]]))
    evaluator = Evaluator(ivp, ExplicitEuler)

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
    plt.title("Error vs h (log-log)")

    # Plot a comparison line which is O(h):
    plt.loglog(hs, hs * 2 * errs[-1] / hs[-1])
    plt.show()

def task73():
    ivp = LTE(t0=0, T=1, y0=np.array([1, 1]), A=np.array([[1, 10],
                                                          [0, -3]]))
    evaluator = Evaluator(ivp, ExplicitEuler)

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
    plt.title("Error vs h (log-log)")

    # Plot a comparison line which is O(h):
    plt.loglog(hs, hs * 2 * errs[-1] / hs[-1])
    plt.show()

def test_implicit():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
    evaluator = Evaluator(ivp, ImplicitEuler)
    solver = ImplicitEuler(ivp, N = 100)
    solver.integrate()
    # Plot the approximations { u_n } vs. the temporal
    # grid points { t_n }:
    plt.plot(solver.grid, solver.us)
    plt.show()

    # Varying time
    grid, errs = evaluator.errvstime(10, relative=True)
    plt.plot(grid, errs)
    plt.show()

    # Varying step size h
    grid, errs = evaluator.errvsh(np.array([10, 20, 25]), relative=True)
    plt.plot(grid, errs)
    plt.show()

def implicit_evaluator():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-2]]))
    evaluator = Evaluator(ivp, ImplicitEuler)

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
    plt.title("Error vs h (log-log)")

    # Plot a comparison line which is O(h):
    plt.loglog(hs, hs * 2 * errs[-1] / hs[-1])
    plt.show()

def implicit_matrix():
    ivp = LTE(t0=0, T=1, y0=np.array([1, 1]), A=np.array([[1, 10],
                                                          [0, -3]]))
    evaluator = Evaluator(ivp, ImplicitEuler)

    # Compute the errors over the grid:
    grid, errs = evaluator.errvstime(1000)

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
    plt.title("Error vs h (log-log) (matrix)")

    # Plot a comparison line which is O(h):
    plt.loglog(hs, hs * 2 * errs[-1] / hs[-1])
    plt.show()


def main():
    # first_test()
    # evaluator_test()
    # task71_72()
    # task73()
    # test_implicit()
    # implicit_evaluator()
    implicit_matrix()

if __name__ == "__main__":
    main()

