import numpy as np
import scipy.sparse.linalg as ssla
import scipy.sparse as ssp
from scipy.ndimage import convolve
import scipy.linalg as sla

from ._DFT import DFT_base
from .Functionals import LDA
from . import Hamiltonian
from .Hartree_kernels import Hartree_1D
from .tools import soft_coulomb

class DFT_1D(DFT_base):
    dim = 1
    psi = None
    tol = 1e-6
    def __init__(self, N_elec, Temp = 0, method = 'FDM', e_corr = 'LDA'):
        self.method = method
        self.N_elec = N_elec
        if e_corr == 'LDA':
            self.corr = LDA(self.dim)
        else:
            raise NotImplementedError('Other than LDA is not implemented.')
        if Temp == 0:
            self.f_occ = np.ones(N_elec)
        else:
            raise NotImplementedError('Finite temperature is not supported yet.')

    def setup_system(self, *args, **kwargs):
        self.ham_0 = Hamiltonian(self.method, self.dim, **kwargs)
        self.v_H_kernel = Hartree_1D(soft_coulomb, self.ham_0.num_meth.coords, self.ham_0.num_meth.equidistant)

    def calculate_prob_dens(self):
        self.prob_dens = np.einsum('j,ij', self.f_occ, np.abs(self.psi[:, :len(self.f_occ)])**2)

    def get_H(self):
        if self.psi is not None:
            ## TODO: implement correct integration for non equidistant grids
            if self.ham_0.num_meth.BC == 'Dirichlet':
                v_H = self.ham_0.num_meth.d_1 * convolve(self.prob_dens, self.v_H_kernel(), mode = 'constant', cval = 0)
            elif self.ham_0.num_meth.BC == 'Periodic':
                v_H = self.ham_0.num_meth.d_1 * convolve(self.prob_dens, self.v_H_kernel(), mode = 'wrap')
            v_C = self.corr.v_c[0](1 / (2 * self.prob_dens))
            return self.ham_0.H_0() + ssp.diags(v_H + v_C)
        return self.ham_0.H_0()

    def solve(self, num = None):
        if self.psi is None:
            v0 = None
        else:
            v0 = self.psi[:, 0]
        if num is None:
            self.energies, self.psi = ssla.eigsh(self.get_H(), k = self.N_elec, which = 'SA', v0 = v0)
        else:
            self.energies, self.psi = ssla.eigsh(self.get_H(), k = num, which = 'SA', v0 = v0)
        
        self.energies = self.energies - self.energies[0]
        if self.ham_0.num_meth.equidistant:
            self.psi = self.psi / np.sqrt(self.ham_0.num_meth.d_1)
        else:
            pass
        self.calculate_prob_dens()

    def SCF(self):
        if self.psi is None:
            self.solve()
        f_i_norm = 1.0
        SC_scheme = Pulay(self.psi.shape[0])
        while f_i_norm > self.tol:
            dens_old = self.prob_dens.copy()
            self.solve()
            f_i = self.prob_dens - dens_old
            f_i_norm = sla.norm(f_i)
            self.prob_dens = SC_scheme.cycle(f_i, dens_old)
        print('# of iterations {}'.format(SC_scheme.iter))
        
class Pulay:
    def __init__(self, size, n = 5, k = 3, alpha = 0.05):
        self.size  = size
        self.n     = n
        self.k     = k
        self.alpha = alpha
        self.R = np.empty((size, n), dtype = np.float64)
        self.F = np.empty((size, n), dtype = np.float64)
        self.iter = 0
        self.f_old = np.zeros(size, dtype = np.float64)
        self.rho_old = np.zeros(size, dtype = np.float64)

    def cycle(self, f_rho, rho):
        if self.iter < self.n:
            rho_new = rho + self.alpha * f_rho
            self.F[:, self.iter] = f_rho - self.f_old
            self.R[:, self.iter] = rho_new - self.rho_old
        elif (self.iter + 1) % self.k != 0:
            rho_new = rho + self.alpha * f_rho
            self.F[:, :-1] = self.F[:, 1:]
            self.R[:, :-1] = self.R[:, 1:]
            self.F[:, -1] = f_rho - self.f_old
            self.R[:, -1] = rho_new - self.rho_old
        else:
            F_T = self.F.T
            C_nlin = (self.R + self.alpha * self.F).dot(sla.inv(F_T.dot(self.F))).dot(F_T)
            rho_new = rho + self.alpha * f_rho - C_nlin.dot(f_rho)
            self.F[:, :-1] = self.F[:, 1:]
            self.R[:, :-1] = self.R[:, 1:]
            self.F[:, -1] = f_rho - self.f_old
            self.R[:, -1] = rho_new - self.rho_old

        self.iter += 1
        print(np.linalg.norm(f_rho))

        self.f_old = f_rho
        self.rho_old = rho_new
        return rho_new









