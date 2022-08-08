import numpy as np
import scipy.sparse.linalg as ssla
import scipy.sparse as ssp
from scipy.ndimage import convolve
import scipy.linalg as sla
from scipy.optimize import root

from ._DFT import DFT_base
from .Functionals import LDA
from . import Hamiltonian
from .Hartree_kernels import Hartree_1D
from .tools import soft_coulomb, fermi_dirac

class DFT_1D(DFT_base):
    dim = 1
    psi = None
    tol = 1e-6
    mu  = None
    def __init__(self, N_elec, Temp = 0, method = 'FDM', e_corr = 'LDA'):
        self.method = method
        self.N_elec = N_elec
        self.kBT = Temp * 3.167e-6
        if e_corr == 'LDA':
            self.corr = LDA(self.dim)
        else:
            raise NotImplementedError('Other than LDA is not implemented.')
        if Temp == 0:
            self.f_occ = np.ones(N_elec)
        else:
            pass
            #raise NotImplementedError('Finite temperature is not supported yet.')

    def setup_system(self, *args, **kwargs):
        self.ham_0      = Hamiltonian(self.method, self.dim, **kwargs)
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
            v_C = self.corr.v_c[0](np.abs(1 / (2 * self.prob_dens)))
            return self.ham_0.H_0() + ssp.diags(v_H + v_C)
        return self.ham_0.H_0()

    def solve(self, num = None):
        if self.psi is None:
            v0 = None
        else:
            v0 = self.psi[:, 0]
        if num is None:
            self.energies, self.psi = ssla.eigsh(self.get_H(), k = self.N_elec + 10, which = 'SA')
        else:
            self.energies, self.psi = ssla.eigsh(self.get_H(), k = num, which = 'SA')
        
        self.energies = self.energies - self.energies[0]

        if self.kBT != 0:
            self.mu = root(self.get_mu, self.energies[self.N_elec - 1], method = 'broyden2', tol = self.tol).x
            self.f_occ = [fermi_dirac(energy, self.mu, self.kBT) for energy in self.energies]

        #print(self.f_occ)

        if self.ham_0.num_meth.equidistant:
            self.psi = self.psi / np.sqrt(self.ham_0.num_meth.d_1)
        else:
            pass
        self.calculate_prob_dens()

    def get_mu(self, mu):
        total = -self.N_elec
        for i in range(self.N_elec + 10):
            total += fermi_dirac(self.energies[i], mu, self.kBT)
        return total

    def SCF(self):
        if self.psi is None:
            self.solve()
        N_it = root(self.iteration, self.prob_dens, method = 'broyden2', tol = self.tol).nit
        print("{} iterations were performed for convergence.".format(N_it))

    def iteration(self, rho):
        self.prob_dens = rho
        self.solve()
        return self.prob_dens - rho
        








