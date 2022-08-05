import numpy as np

from ._FDM import FDM_base
from .tools import set_class_attributes
from scipy.special import factorial
import scipy.sparse as ssp

BC_dict = {0 : 'Dirichlet', 1 : 'Neumann', 2 : 'Robin', 3 : 'Periodic',
           'periodic' : 'Periodic', 'dirichlet' : 'Dirichlet', 'neumann' : 'Neumann', 'robin' : 'Robin',
           'Periodic' : 'Periodic', 'Dirichlet' : 'Dirichlet', 'Neumann' : 'Neumann', 'Robin' : 'Robin'}

class FDM_1D(FDM_base):
    del_1 = None
    del_2 = None
    partial = [None, None, None]
    order = 2

    def __init__(self, equidistant = True, BC = 'Dirichlet', **kwargs):
        assert isinstance(equidistant, bool)
        assert BC in BC_dict

        self.equidistant = equidistant
        self.BC = BC_dict[BC]

        if equidistant:
            set_class_attributes(self, ['n_1', 'min_1', 'max_1'], **kwargs)
            if self.BC == 'Dirichlet':
                self.d_1 = (self.max_1 - self.min_1) / (self.n_1 + 1)
                self.coords = np.linspace(self.min_1 + self.d_1, self.max_1 - self.d_1, self.n_1)
            elif self.BC == 'Periodic':
                self.d_1 = (self.max_1 - self.min_1) / self.n_1
                self.coords = np.linspace(self.min_1, self.max_1, self.n_1, endpoint = False)
            else:
                raise NotImplementedError("Other than 'Periodic' and 'Dirichlet' boundary conditions are not implemented yet.")
        else:
            raise NotImplementedError('The non equidistant grid option is not yet available.')

        self.size = len(self.coords)

    def _deriv_custom(self, n, order):
        pass

    def _deriv_eq(self, n, order, sparse):
        mat        = np.power.outer(np.arange(-order, order + 1), np.arange(0, 2 * order + 1)).transpose()
        vec        = np.zeros(2 * order + 1)
        vec[n]     = factorial(n, exact = True)
        coeffs    = list(np.linalg.solve(mat, vec) / self.d_1**n)
        if sparse:
            if self.BC == 'Periodic':
                coeffs = coeffs[order+1:] + coeffs + coeffs[:order]
                offsets = [-self.n_1 + i for i in range(1, order + 1)] + [i for i in range(-order, order + 1)] + [self.n_1 - order + i for i in range(order)]
                self.partial[n] = ssp.diags(coeffs, offsets = offsets, shape = (self.size, self.size))
            elif self.BC == 'Dirichlet':
                self.partial[n] = ssp.diags(coeffs, offsets = range(-order, order + 1), shape = (self.size, self.size))
            else:
                pass

    def deriv(self, n, order = 2, sparse = True):
        if self.partial[n] is not None and self.order == order:
            return self.partial[n]
        else:
            if self.equidistant:
                self._deriv_eq(n, order, sparse)
            else:
                self._deriv_custom(n, order, sparse)
        return self.partial[n]

    def integrate(self, ftab):
        if self.equidistant:
            return np.sum(ftab) * self.d_1
        else:
            pass




