import numpy as np
import scipy as sp
import scipy.linalg, scipy.integrate
import matplotlib.pyplot as plt

from evaluator import Evaluator
from lotka_volterra import LotkaVolterra
from lte import LTE
from rk34 import RK34
from rk4 import RK4
from vdp import VDP

def verify_rk4():
    ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
    solver = RK4(ivp, N = 100, adaptive=False)
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
    plt.loglog(hs, errs, label="Actual Errs")
    plt.title("Error vs h (log-log)") # should have lutning 4
    plt.xlabel("h (log)")
    plt.ylabel("err (log)")

    # Plot a comparison line which is O(h):
    plt.loglog(hs, hs**4 * 2 * errs[-1] / hs[-1], label="Comparison line O(h^4)")
    plt.legend()
    plt.show()

def test_lotka():
    ivp = LotkaVolterra(0, 100, np.array([1, 1]), 3, 9, 15, 15)
    solver = RK34(ivp, 1000, 1E-6, adaptive=True)

    solver.integrate()

    plt.title("Lotka-volterra")
    plt.xlabel("t")
    plt.ylabel("Population")
    plt.plot(solver.grid, solver.us, label=("y1 (hare)", "y2 (wolf)"))
    plt.legend()
    plt.show()

    plt.plot(solver.us[:,0], solver.us[:,1])
    plt.show()

    H0 = ivp.H(ivp.y0[0], ivp.y0[1]) # H at t0
    H = [ivp.H(xn, yn) for xn, yn in solver.us]

    H_drift = [np.abs(H0 / h - 1) for h in H]
    plt.semilogy(solver.grid, H_drift) # linlog due to time / error - plot
    plt.xlabel("time")
    plt.ylabel("H / H0 - 1")
    plt.title("H-drift over time (semilogy scale)")
    plt.show()

def test_vdp():
    mu = 100
    initial_conds = [[2, 0], [0, 2], [-2, 0], [1, 1]]
    for y1, y2 in initial_conds:
        ivp = VDP(0, 2*mu, [y1, y2], mu)
        solver = RK34(ivp, 10000, tol=0.0000001, adaptive=True)
        solver.integrate()
        # plt.plot(solver.grid, solver.us[:,0], label = str(y1) + ", " + str(y2))
        plt.plot(solver.us[:,0], solver.us[:,1], label = str(y1) + ", " + str(y2)) # potato
    plt.legend()
    plt.title("Phase diagram when varying initial conditions for VDP")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.show()


    ivp = VDP(0, 2*mu, [2, 0], mu)
    solver = RK34(ivp, 10000, tol=0.0000001, adaptive=True)
    solver.integrate()

    plt.title(f"VDP with mu={mu}")
    plt.plot(solver.grid, solver.us[:,1], label="y2")
    plt.legend()
    plt.show()
    plt.title(f"VDP phase diagram with mu={mu}")
    plt.plot(solver.us[:,0], solver.us[:,1])
    plt.show()

    plt.semilogy(solver.grid[:-1], np.diff(solver.grid)) # Plotting step sizes over time
    plt.show()

    # Vad fan händer när vi ändrar step size?

def task_43():
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330]
    steps = []
    for mu in mus:
        ivp = VDP(0, 0.7 * mu, [2, 0], mu)
        solver = RK34(ivp, 10000, tol=1E-7, adaptive=True) # Varför kan mindre tolerans vara snabbare?
        solver.integrate()
        steps.append(len(solver.grid))
        print(f"mu={mu} with {len(solver.grid)} steps.")

        # plt.plot(solver.grid, solver.us[:,0])
        # plt.show()

        # plt.plot(solver.us[:,0], solver.us[:,1])
        # plt.show()

    plt.title("Steps as a function of µ (loglog)")
    plt.loglog(mus, steps, label="Actual nbr steps")
    plt.xlabel("µ")
    plt.ylabel("# steps")
    ref = [mu**2 for mu in mus]
    plt.loglog(mus, ref, label="Reference of µ^2")
    plt.legend()
    plt.show()

    # mu**2 och steps växer med samma lutning, alltså beror steps kvadratiskt på mu och stiffness beror kvadratiskt på mu

def test_builtin():
    mu = 100
    ivp = VDP(0, 2*mu, [2, 0], mu)
    result = scipy.integrate.solve_ivp(ivp.f, (0, 2*mu), (2, 0), method="BDF")
    plt.plot(result.t, result.y[0], result.y[1])
    plt.show()

def task_74():
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 10_000]
    steps = []
    for mu in mus:
        ivp = VDP(0, 0.7 * mu, [2, 0], mu)
        result = scipy.integrate.solve_ivp(ivp.f, (0, 0.7*mu), (2, 0), method="BDF", atol = 1E-7, rtol = 0.001)
        # plt.plot(result.t, result.y[0], result.y[1])
        # plt.show()

        print(f"mu={mu} with {len(result.t)} steps.")
        steps.append(len(result.t))

        # plt.plot(solver.grid, solver.us[:,0])
        # plt.show()

        # plt.plot(solver.us[:,0], solver.us[:,1])
        # plt.show()

    plt.loglog(mus, steps)
    ref = [mu**2 for mu in mus]
    plt.loglog(mus, ref)
    plt.show()

    # Seems like wall-clock time is independent of mu

def task_75():
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330]
    steps = []
    for mu in mus:
        ivp = VDP(0, 0.7 * mu, [2, 0], mu)
        result = scipy.integrate.solve_ivp(ivp.f, (0, 0.7*mu), (2, 0), method="RK45", atol = 1E-7, rtol = 0.001)
        # plt.plot(result.t, result.y[0], result.y[1])
        # plt.show()

        print(f"mu={mu} with {len(result.t)} steps.")
        steps.append(len(result.t))

        # plt.plot(solver.grid, solver.us[:,0])
        # plt.show()

        # plt.plot(solver.us[:,0], solver.us[:,1])
        # plt.show()

    plt.loglog(mus, steps)
    ref = [mu**2 for mu in mus]
    plt.loglog(mus, ref)
    plt.show()

    # Similar performance to own method!


def main():
    # verify_rk4()
    # error_rk4()
    # test_lotka()
    # test_vdp()
    task_43()
    # test_builtin()
    # task_74()
    # task_75()

if __name__ == "__main__":
    main()