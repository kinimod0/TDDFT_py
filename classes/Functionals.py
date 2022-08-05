from sympy import lambdify, symbols, log, pprint, Subs
import numpy as np

r_s = symbols('r_s')

lda_1D_params_0 = {'A' : 18.40, 'B' : 0.0, 'C' : 7.501, 'D' : 0.10185, 'E' : 0.012827, 'alpha' : 1.511 , 'beta' : 0.258  , 'm' : 4.424}
lda_1D_params_1 = {'A' :  5.24, 'B' : 0.0, 'C' : 1.568, 'D' : 0.1286 , 'E' : 0.00320 , 'alpha' : 0.0538, 'beta' : 1.56e-5, 'm' : 2.958}

def LDA_1D_C():
    A, B, C, D, E, alpha, beta, m = symbols('A B C D E alpha beta m')
    fraction = (r_s + E * r_s**2) / (A + B * r_s + C * r_s**2 + D * r_s**3)
    logarithm = log(1 + alpha * r_s + beta * r_s**m)
    e_c_expr = -fraction * logarithm / 2
    e_c_0 = lambdify(r_s, e_c_expr.subs(lda_1D_params_0), modules = ['numpy'])
    e_c_1 = lambdify(r_s, e_c_expr.subs(lda_1D_params_1), modules = ['numpy'])
    v_c_0 = lambdify(r_s, (e_c_expr - r_s * e_c_expr.diff(r_s)).subs(lda_1D_params_0), modules = ['numpy'])
    v_c_1 = lambdify(r_s, (e_c_expr - r_s * e_c_expr.diff(r_s)).subs(lda_1D_params_1), modules = ['numpy'])
    return [e_c_0, e_c_1], [v_c_0, v_c_1]

class LDA:
    def __init__(self, dim, which = 0):
        if dim == 1:
            self.e_c, self.v_c = LDA_1D_C()
        else:
            raise NotImplementedError("Other than dimension 1 is not implemented.")


