from typing import Callable
import numpy as np


class Hartree_1D:
    def __init__(self, kernel_func : Callable, coords : np.ndarray, equidistant : bool):
        self.kernel_func = kernel_func
        if equidistant:
            d_1 = coords[1] - coords[0]
            n = len(coords)
            self._make_eq_kernel(n, d_1)
        else:
            raise NotImplementedError("Only equidistant grid is implemented.")

    def _make_eq_kernel(self, n, d_1):
        self.kernel = self.kernel_func(np.arange(-n, n + 1) * d_1)

    def __call__(self):
        return self.kernel
