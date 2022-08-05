import numpy as np

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




