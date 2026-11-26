import numpy as np
import scipy
from bc import BC, Dirichlet, Neumann

def dx_dirichlet(L, N):
    return L / (N + 1)

def dx_neumann(L, N):
    """Implemented using 'approach 1'."""
    return L / N

def grid_dirichlet(L, N, dx):
    return np.linspace(0 + dx, L - dx, N)

def grid_neumann(L, N, dx):
    """Implemented using 'approach 1'."""
    return np.linspace(0 + dx, L, N)

def FDMd2dx2(L: float, bc_left: BC, bc_right: BC, N: int) -> tuple[float, np.array, np.array, np.array] :
    """
    Compute the FDM discretization

    $$
    u \mapsto Au + BCcorr
    $$

    of the differential operator

    $$
    y''
    $$

    on

    $$
    x \in [0 , L],
    $$

    given the boundary conditions BCleft at x = 0 and
    BCright at x = L, using N computational points .

    Inputs
    ------
    L: float
    BCleft : BC object (e.g. Dirichlet , Neumann )
    BCright : BC object (e.g. Dirichlet , Neumann )
    N: integer > 0

    Outputs
    -------
    dx: float , grid spacing
    grid : np. array with shape (N ,) , the computational
        points
    A: np. array or scipy . sparse matrix with shape (N,N)
    BCcorr : np. array with shape (N ,1) or (N ,) ,
        containing boundary condition correction terms
    """
    dx = None
    grid = None
    BCCorr = np.zeros(N)

    diag_sup = [1 for _ in range(N-1)]
    diag_main = [-2 for _ in range(N)]
    diag_sub = [1 for _ in range(N-1)] 
    diags = (diag_sup, diag_main, diag_sub)

    if type(bc_right) == Dirichlet:
        dx = dx_dirichlet(L, N)
        grid = grid_dirichlet(L, N, dx)
        BCCorr[-1] = bc_right.value / dx
    elif type(bc_right) == Neumann:
        dx = dx_neumann(L, N)
        grid = grid_neumann(L, N, dx)
        BCCorr[-1] = 2 * bc_right.value / dx
        diag_sub[-1] = 2 # Adjusts for the additional u_N-1

    BCCorr[0] = bc_left.value / dx**2 # Always D to the left

    A = scipy.sparse.diags(diags, (1, 0, -1)) / dx**2
    return dx, grid, A, BCCorr

def FDMSturmLiouville(L: float, bc_left: BC, bc_right: BC, N: int, q):
    dx = None
    grid = None
    BCCorr = np.zeros(N)

    diag_sup = [1 for _ in range(N-1)]
    diag_main = [-2.0 for x in range(N)]
    diag_sub = [1 for _ in range(N-1)] 

    if type(bc_right) == Dirichlet:
        dx = dx_dirichlet(L, N)
        grid = grid_dirichlet(L, N, dx)
        BCCorr[-1] = bc_right.value / dx
    elif type(bc_right) == Neumann:
        dx = dx_neumann(L, N)
        grid = grid_neumann(L, N, dx)
        BCCorr[-1] = 2 * bc_right.value / dx
        diag_sub[-1] = 2 # Adjusts for the additional u_N-1
    BCCorr[0] = bc_left.value

    diag_main = np.array(diag_main)
    diag_main += q(grid) * dx**2
    diags = (diag_sup, diag_main, diag_sub)

    A = scipy.sparse.diags(diags, (1, 0, -1)) / dx**2
    return dx, grid, A, BCCorr
