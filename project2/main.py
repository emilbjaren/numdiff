import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from bc import BC, Dirichlet, Neumann
from fdm import FDMSturmLiouville, FDMd2dx2
from solver import SESolver


def prepare_plot_data(
    grid: np.ndarray,
    values: np.ndarray,
    L: float,
    bc_left: BC,
    bc_right: BC,
) -> tuple[np.ndarray, np.ndarray]:
    """Include known Dirichlet boundary values in data prepared for plotting."""
    plot_grid = np.asarray(grid)
    plot_values = np.asarray(values)

    if plot_grid.ndim != 1 or plot_values.ndim != 1:
        raise ValueError("grid and values must be one-dimensional")
    if plot_grid.size == 0 or plot_grid.size != plot_values.size:
        raise ValueError("grid and values must have the same non-zero length")

    plot_values = plot_values.astype(np.result_type(plot_values.dtype, np.float64), copy=True)

    if isinstance(bc_left, Dirichlet):
        if np.isclose(plot_grid[0], 0.0, rtol=0.0, atol=1e-12):
            plot_values[0] = bc_left.value
        else:
            plot_grid = np.concatenate(([0.0], plot_grid))
            plot_values = np.concatenate(([bc_left.value], plot_values))

    if isinstance(bc_right, Dirichlet):
        if np.isclose(plot_grid[-1], L, rtol=0.0, atol=1e-12):
            plot_values[-1] = bc_right.value
        else:
            plot_grid = np.concatenate((plot_grid, [L]))
            plot_values = np.concatenate((plot_values, [bc_right.value]))

    return plot_grid, plot_values


def test_err(N):
    # Test med sinus
    k = 5
    f = lambda x: -(k * np.pi)**2 * np.sin(k * np.pi * x)
    y_exact = lambda x: np.sin(k*np.pi * x)

    (dx, grid, A, BCcorr) = FDMd2dx2(1,
    Dirichlet(0),
    Dirichlet(0),
    N)
    RHS = f(grid) - BCcorr
    # print(RHS)
    u = scipy.sparse.linalg.spsolve(A, RHS)
    u_exact = y_exact(grid)
    err = u - u_exact
    e_rms = np.linalg.norm(err) / np.sqrt(len(err))

    return dx, e_rms

def test_neumann():
    # Test med sinus
    k = 5
    f = lambda x: -(k * np.pi)**2 * np.sin(k * np.pi * x)

    N = 100
    (dx, grid, A, BCcorr) = FDMd2dx2(1,
        Dirichlet(-1),
        Neumann(0),
        N)
    RHS = f(grid) - BCcorr
    print(RHS)
    u = scipy.sparse.linalg.spsolve(A, RHS)
    plot_grid, plot_u = prepare_plot_data(
        grid, u, 1, Dirichlet(-1), Neumann(0)
    )
    plt.plot(plot_grid, plot_u)
    plt.show()


def test_1():
    FDMd2dx2(1, Dirichlet(0), Dirichlet(0), 20)

    # Test med sinus
    k = 5
    f = lambda x: -(k * np.pi)**2 * np.sin(k * np.pi * x)
    y_exact = lambda x: np.sin(k*np.pi * x)

    bc_left = Dirichlet(0)
    bc_right = Dirichlet(0)
    (dx, grid, A, BCcorr) = FDMd2dx2(1, bc_left, bc_right, 100)
    RHS = f(grid) - BCcorr
    # print(RHS)
    u = scipy.sparse.linalg.spsolve(A, RHS)
    u_exact = y_exact(grid)

    plot_grid, plot_u = prepare_plot_data(grid, u, 1, bc_left, bc_right)
    plt.plot(plot_grid, plot_u, label='Approximate')
    plt.plot(plot_grid, y_exact(plot_grid), label='Exact')
    plt.legend()
    plt.show()

    err = u - u_exact
    e_rms = np.linalg.norm(err) / np.sqrt(len(err))

    # Testar med många olika steglängder
    dxs = []
    errs = []
    for N in range(200, 2000, 5):
        dx, err = test_err(N)
        dxs.append(dx)
        errs.append(err)

    compare = lambda x: 13 * np.asarray(x)**2
    compare_errs = compare(dxs)
    plt.loglog(dxs, compare_errs)
    plt.loglog(dxs, errs)

    plt.show()

def test_beam():
    L = 10 # m
    xx = np.linspace(0, L, 1001)
    x = xx[1:-1]    
    E = 1.9e11 # N / m^2
    I = 1e-3 * (3 - 2*(np.cos(np.pi * x / L)**12)) # mystery unit
    q = lambda x: -50e3 # kN / m

    N = 1001 # 999 interior + 2

    # Solve the first equation separately
    (dx, grid, A, BCCorr) = FDMd2dx2(L, Dirichlet(0), Dirichlet(0), len(x))
    solver = BVPSolver(dx, grid, A, BCCorr, q)
    solver.solve()
    Ms = solver.u
    plot_grid, plot_Ms = prepare_plot_data(
        grid, Ms, L, Dirichlet(0), Dirichlet(0)
    )
    plt.plot(plot_grid, plot_Ms, label='M')
    plt.legend()
    plt.show()

    (dx, grid, A, BCCorr) = FDMd2dx2(L, Dirichlet(0), Dirichlet(0), len(x))
    RHS = Ms / (E*I) - BCCorr
    u = scipy.sparse.linalg.spsolve(A, RHS)
    print(u[500])
    plot_grid, plot_u = prepare_plot_data(
        grid, u, L, Dirichlet(0), Dirichlet(0)
    )
    plt.plot(plot_grid, plot_u, label='u')
    plt.legend()
    plt.show()

def test_sl():
    """Solving for Neumann to the right."""
    L = 1
    xx = np.linspace(0, L, 1001)
    x = xx[1:-1]
    q = lambda r: 0
    bc_left = Dirichlet(0)
    bc_right = Neumann(0)
    (dx, grid, A, BCCorr) = FDMSturmLiouville(L, bc_left, bc_right, len(x), q)
    vals, vecs = scipy.sparse.linalg.eigs(A, 3, which='SM')

    # Magi för att sortera
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]

    print(vals)
    exakt = np.array([-((1/2 + k)* np.pi)**2 for k in range(3)])
    print(exakt)
    diffs = np.abs(vals - exakt)
    print(diffs)

    for i in range(vecs.shape[1]):
        plot_grid, plot_vec = prepare_plot_data(
            grid, vecs[:, i].real, L, bc_left, bc_right
        )
        plt.plot(plot_grid, plot_vec, label=f"Mode {i + 1}")
    plt.legend()
    plt.show()

def err_sl():
    Ns = np.array([i for i in range(199, 1999, 100)])

    errs = []
    for N in Ns:
        err = solve_sl(N)
        errs.append(err)
        if N == 499:
            print(f"Errors for N = 499: {err}")

    plt.loglog(Ns, errs, label = ["Eigenvalue 1", "Eigenvalue 2", "Eigenvalue 3"]) # Vilken lutning har dessa?
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Absolute err")
    plt.show()

def solve_sl(N: int) -> np.array:
    """Solving for Neumann to the right."""
    L = 1
    xx = np.linspace(0, L, N+2)
    x = xx[1:-1]
    q = lambda r: 0
    (dx, grid, A, BCCorr) = FDMSturmLiouville(L, Dirichlet(0), Neumann(0), len(x), q)
    vals, vecs = scipy.sparse.linalg.eigsh(A, 3, which='SM')

    # Magi för att sortera
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]

    # print(vals)
    exakt = np.array([-((1/2 + k)* np.pi)**2 for k in range(3)])
    # print(exakt)
    diffs = np.abs(vals - exakt)
    # print(diffs)
    return diffs

def test_SE():
    """Particle in a potential box."""
    V = lambda x: x-x
    solver = SESolver(400, 6, V)
    solver.solve()

def test_SE2():
    """Triangle in middle potential"""
    V = lambda x: 700*(0.5-np.abs(x-0.5))
    solver = SESolver(400, 6, V)
    solver.solve()

def test_SE3():
    """Strong sine potential."""
    V = lambda x: 800*(np.sin(np.pi * x)**2)
    solver = SESolver(400, 6, V)
    solver.solve()

def test_SE4():
    """Strong sine potential."""
    V = lambda x: 800*(np.sin(2 * np.pi * x)**2)
    solver = SESolver(400, 6, V)
    solver.solve()



def main():
    # test_1()
    # test_neumann()
    # test_beam()
    # test_sl()
    # err_sl()
    # test_SE()
    # test_SE2()
    # test_SE3()
    test_SE4()

if __name__ == "__main__":
    main()