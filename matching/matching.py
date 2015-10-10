import numpy as _np
import scipy.optimize as _scyopt


def matching(acc, objectives=None, constraints=None,variables=None, covariables=None):
    """ Performs the matching of optical functions using least squares.


    variables : Must be a list of dictionaries with keys:
        'fam_name': family name of the ring element to be varied;
        'atribute': name of the attribute to be varied.
        'index'   : Optional. In case the attribute is a vector or a matrix,
            this key defines which index of the attribute must be varied. It
            must be an integer for 1D or tuple for multidimensional arrays.
        'min' : lower bound for the attribute value. Define None for unbounded (default).
        'max' : upper bound for the attribute value. Define None for unbounded (default).
    objectives: List of dictionaries defining the objectives (or penalties) of
        the optimization. Each dictionary must have the keys:


    """




res = _scyopt.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
            bounds=None, constraints=(), tol=None, callback=None, options=None)
