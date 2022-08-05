from typing import Callable
import scipy.sparse as ssp
import numpy as np

from ._Hamiltonian import Hamiltonian_base
from . import FDM_classes

method_dict = {'fdm' : 'FDM', 'FDM' : 'FDM', 'fem' : 'FEM', 'FEM' : 'FEM'}
dim_dict = {1 : {'FDM' : 'FDM_1D'}, 2 : {'FDM' : 'FDM_2D'}, 3 : {'FDM' : 'FDM_3D'}}

class Hamiltonian(Hamiltonian_base):
    def __init__(self, method, dim, **kwargs):
        assert method in method_dict
        self.method = method_dict[method]
        if self.method == 'FDM':
            self.num_meth = self._wrap_FDM(dim, **kwargs)

        self.V_pot = np.zeros_like(self.num_meth.coords)


    def _wrap_FDM(self, dim, **kwargs):
        FDM_class = getattr(FDM_classes, dim_dict[dim]['FDM'])
        return FDM_class(**kwargs)

    def couple_Vpot(self, func : Callable):
        self.V_pot = func(self.num_meth.coords)

    def kinetic(self, order = 2, sparse = True):
        return -self.num_meth.deriv(2, order = order, sparse = sparse) / 2

    def H_0(self, order = 2, sparse = True):
        return self.kinetic(order = order, sparse = sparse) + ssp.diags(self.V_pot)


