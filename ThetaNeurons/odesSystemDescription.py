# ------------------------------------------------------------------------------- #
import numpy as np;
# ------------------------------------------------------------------------------- #
"""
This file contains a representation of  the right-hand side function of a system of
ordinary differential equations (ODEs)  as a class. A right-hand side function with
parameters can be handled by using a class where the parameters are attributes, and
one of the methods defines the considered ODEs system.
"""
# ------------------------------------------------------------------------------- #
class thetaNeurons:
    def __init__(self, tau, eta, kappa, m, epsilon):
        self.tau, self.eta, self.kappa, self.m, self.eps = tau, eta, kappa, m, epsilon;
# ------------------------------------------------------------------------------- #
    def rhsFunction(self, qpi, t, *args):
        N = len(qpi[:-1]);
        am = 2**self.m * (np.math.factorial(self.m))**2 / np.math.factorial(2 * self.m);
        signal = np.sum(am * (1. - np.cos(qpi[:-1]))**self.m) / N;
        dqpdt = (1. - np.cos(qpi[:-1])) + \
                (1. + np.cos(qpi[:-1])) * (self.eta + self.kappa * qpi[-1]);
        dqpdt += self.eps * np.cos(2 * qpi[:-1]) * (self.eta + self.kappa * qpi[-1]);
        didt = (signal - qpi[-1]) / self.tau;
        rhs = np.append(dqpdt, didt);
        return rhs;

# ------------------------------------------------------------------------------- #