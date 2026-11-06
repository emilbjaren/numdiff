import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from ivp import IVP

class LTE(IVP):
    def __init__(self, t0: np.number, T: np.number, y0: np.array, A: np.array):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t: np.number, y: np.array):
        if len(self.A) == 1: # Operatorn @ funkar inte när A är 1x1
            return self.A * y

        return self.A @ y

    def exact(self, t):
        return scipy.linalg.expm(t * self.A) @ self.y0 

if __name__ == "__main__":
    # Test to see that the "exact" function works. 
    lte = LTE(0, 10, [1, 1], np.eye(2))
    print(lte.exact(5))

    ts = np.linspace(0,1)
    ys = [lte.exact(t) for t in ts]

    plt.plot(ts, ys)
    plt.show()