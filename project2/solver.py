import numpy as np
import scipy
import matplotlib.pyplot as plt

from bc import Dirichlet
from fdm import FDMSturmLiouville

class SESolver():
    def __init__(self, N: int, num_ev: int, V):
        self.bc_left = Dirichlet(0)
        self.bc_right = Dirichlet(0)
        self.num_ev = num_ev
        self.V = lambda x: V(x)

        self.N = N
        self.L = 1
        self.xx = np.linspace(0, self.L, N + 2)
        self.x = self.xx[1:-1]

    def solve(self):
        (dx, grid, A, BCCorr) = FDMSturmLiouville(self.L, self.bc_left, self.bc_right, len(self.x), lambda x: -self.V(x))
        # Normera egenvektorer!
        self.vals, self.vecs = scipy.sparse.linalg.eigsh(A, self.num_ev, which='LA')

        # Sortera ev
        order = np.argsort(-self.vals)
        self.vals = self.vals[order]
        self.vecs = self.vecs[:, order]

        self._normalize_vecs(dx)

        print(self.vecs_squared.shape)

        plt.xlabel("x")
        plt.ylabel("V(x)")
        plt.plot(grid, self.V(grid))
        plt.show()

        plt.xlabel("x")
        plt.ylabel("$\Psi (x)$")
        plt.plot(grid, self.vecs.transpose())
        plt.show()

        plt.xlabel("x")
        plt.ylabel("$|\Psi (x)|^2$")
        plt.plot(grid, self.vecs_squared.transpose())
        plt.show()

        plt.xlabel("x")
        plt.ylabel("$|\Psi (x)|^2 + E_k / 40$")
        for val, vec in zip(self.vals, self.vecs_squared):
            plt.plot(grid, -val/40 + vec)

        plt.show()

        print(self.vals)

        # plot egenfunk
        # plot |egenfunk|^2

    def _normalize_vecs(self, dx):
        vecs = []
        for k in range(self.num_ev):
            vec = np.array(self.vecs[:,k])
            norm = np.sqrt(dx * np.sum(np.abs(vec)**2))
            vec = vec / norm
            vecs.append(vec)

        self.vecs = np.array(vecs)
        self.vecs_squared = np.abs(self.vecs)**2
