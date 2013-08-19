#!/usr/bin/env python
"""
Implements the addition of asymmetric errors on a variable.
"""
import math
import numpy as np

class PiecewiseLinearModel:
    """
    Implemented as in http://arxiv.org/pdf/physics/0306138v1.pdf, Model 1
    """
    @classmethod
    def S(cls, sigp, sigm):
        """
        the S variable for root finding.

        Args:
            sigp: the sigma-plus variation
            sigm: the sigma-minus variation
        """
        return np.power(sigp, 2)+np.power(sigm, 2)

    @classmethod
    def D(cls, sigp, sigm):
        """
        The D variable for root finding.

        Args:
            sigp: the sigma-plus variation
            sigm: the sigma-minus variation
        """
        return sigp-sigm

    @classmethod
    def sig(cls, _s, _d, sign=+1):
        """
        Calculates sigma-plus and sigma-minus from the S and D variables.

        Args:
            _s: the S variable
            _d: the D variable
            sign: whether you want the plus or the minus sigma

        """
        return sign * 0.5*(_d + sign*math.sqrt(2.0*_s-np.power(_d, 2)))

    @classmethod
    def s(cls, _v, _d):
        """
        Evaluates the new S variable from the D variable and the variance

        Args:
            _v: the value of the variance
            _d: the value of the D variable
        """
        return 2*_v + _d**2/math.pi

    @classmethod
    def d(cls, _s, _g, _d):
        """
        Evaluates the D variable with the updated S variable, the skew(gamma) and the previous D variable.

        Args:
            _s: the S variable at N
            _g: the skew (constant)
            _d: the D variable at N-1
        Returns:
            the _s variable at N
        """
        return 2.0/(3.0 * _s) * (np.sqrt(2.0 * math.pi) * _g - np.power(_d, 3) * (1.0 / math.pi - 1.0))

    @classmethod
    def Mu(cls, sig1, sig2, x0):
        """
        Evaluates the mean of the asymmetric gaussian.

        Args:
            sig1: the sigma-plus (positive uncertainty)
            sig2: the sigma-minus (negative uncertainty)
            x0: the mode of the distribution
        Returns:
            The statistical mean of the distribution.
        """
        return x0 + (1.0/(math.sqrt(2.0*math.pi)))*(sig1 - sig2)

    @classmethod
    def V(cls, sig1, sig2, x0):
        """
        Evaluates the variance
        """
        return 0.5*(sig1**2 + sig2**2) - 1.0/(2.0*math.pi)*(sig1-sig2)**2 

    @classmethod
    def Gamma(cls, sig1, sig2, x0):
        """
        Evaluates the skew
        """
        return (
            1.0/math.sqrt(2.0*math.pi)*
            (
                2*(sig1**3 - sig2**3) - 3.0/2.0*(sig1 - sig2)*
                (sig1**2 + sig2**2) + 1.0/math.pi * (sig1 - sig2)**3
            )
        )

def total_cumulants(sigmas, x0=0, model=PiecewiseLinearModel):
    """
    Calculates the total cumulants of the convolved distribution of
    asymmetric gaussians.

    Returns:
        (mean, variance, skew) of the distribution
    """
    mu, v, gamma = 0, 0, 0
    for _s in sigmas:
        args = _s[0], _s[1], x0
        _mu, _v, _gamma = model.Mu(*args), model.V(*args), model.Gamma(*args)
        mu += _mu
        v += _v
        gamma += _gamma
    return mu, v, gamma

def solve_iteratively(mu, v, gamma, x0=0, model=PiecewiseLinearModel):
    """
    Iteratively solves the equations for S and D to find the sigma values corresponding
    to the total cumulants mu, v, gamma.
    """
    d_prev = float('inf')
    d_cur = 0.0
    s_prev = float('inf')
    s_cur = 0.0

    max_err = 0.0000000001
    N_it = 100
    n_it = 0

    mu_calc = float('inf')
    v_calc = float('inf')
    g_calc = float('inf')
    delta = float('inf')

    while (np.abs(mu_calc-mu)>max_err or np.abs(v_calc-v)>max_err or np.abs(g_calc-gamma)>max_err) and n_it<N_it:
        n_it += 1

        d_prev = d_cur
        s_prev = s_cur

        s_new = model.s(v, d_prev)
        d_new = model.d(s_new, gamma, d_prev)

        sig1 = model.sig(s_new, d_new, +1)
        sig2 = model.sig(s_new, d_new, -1)

        d_cur = d_new
        s_cur = s_new

        mu_calc = model.Mu(sig1, sig2, x0)
        v_calc = model.V(sig1, sig2, x0)
        g_calc = model.Gamma(sig1, sig2, x0)

        #Calculate the shift in the mean
        delta = mu - d_new/math.sqrt(2*math.pi)

    return sig1, sig2, delta

def add_errors(sigmas):
    mu, v, gamma = total_cumulants(sigmas)
    sig1, sig2, delta = solve_iteratively(mu, v, gamma)
    return sig1, sig2, delta

if __name__=="__main__":

    #Inputs
    sigmas = [

        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
        (1.5, 0.5),
    ]
    r = add_errors(sigmas)
    print r[0], r[1], r[2]
