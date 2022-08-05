from classes import DFT_1D
import numpy as np
import matplotlib.pyplot as plt

dim = 1


L = 100
n_1 = 499
kwargs = {'BC' : 'dirichlet', 'n_1' : n_1, 'min_1' : -L / 2, 'max_1' : L / 2}
dft = DFT_1D(10)
dft.setup_system(**kwargs)

def V_pot(x):
    return np.zeros_like(x)#2 * np.cos(12 * np.pi / L * x)#-10 / np.sqrt(1 + x**2)

dft.ham_0.couple_Vpot(V_pot)
dft.SCF()
#dft.solve(num = 20)

print((dft.energies - dft.energies[0]) * 27.211)
plt.plot(dft.ham_0.num_meth.coords, dft.prob_dens)
#for i in range(dft.psi.shape[1]):
#    plt.plot(dft.ham_0.num_meth.coords, dft.psi[:, i])
plt.show()


