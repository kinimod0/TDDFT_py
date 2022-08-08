import numpy as np
from scipy.special import logsumexp

def set_class_attributes(cls, attributes, **kwargs):
    if hasattr(attributes, '__iter__'):
        for atr in attributes:
            try:
                setattr(cls, atr, kwargs[atr])
            except KeyError:
                raise KeyError("The key {} is not in the keyword arguments but was expected for the chosen method.".format(atr))

    else:
        try:
            setattr(cls, atr, kwargs[atr])
        except KeyError:
            raise KeyError("The key {} is not in the keyword arguments but was expected for the chosen method.".format(atr))


def soft_coulomb(x):
    return 1 / np.sqrt(1 + x**2)

for_sum = np.zeros(2)
def fermi_dirac(eps, mu, kBT):
    for_sum[1] = (eps - mu) / kBT
    sol = -logsumexp(for_sum)
    return np.exp(sol)



