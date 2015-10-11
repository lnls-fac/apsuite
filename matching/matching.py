import numpy as _np
import scipy.optimize as _scyopt


def matching(acc, objectives=None, constraints=None,variables=None, covariables=None):
    """ Performs the matching of optical functions using least squares.

    variables : Must be a list of dictionaries with keys:
        'elements': family name of the ring element to be varied or list of
            indices of the elements in the lattice.
        'separated': Optional, Boolean. True if each element is to be treated as
            independent knobs or False if they should be seen as a family.
            Default is False.
        'atribute': name of the attribute to be varied.
        'index'   : Optional. In case the attribute is a vector or a matrix,
            this key defines which index of the attribute must be varied. It
            must be an integer for 1D or tuple for multidimensional arrays.
        'min' : lower bound for the attribute value. Define None for unbounded (default).
        'max' : upper bound for the attribute value. Define None for unbounded (default).
    objectives: List of dictionaries defining the objectives (or penalties) of
        the optimization. Each dictionary must have the keys:
        'quantities': string or tuple of strings defining which quantities will
            be used in 'fun'. The full list of possible candidates: 'betax','betay',
            'alphax','alphay','etax','etay','etaxp','etayp','mux','muy','tunex',
            'tuney',
        'where': family name of the elements where to calculate
        'fun' : function which takes the quantities defined in 'quantities', in that
            order, and returns a float or numpy_ndarray of the same size as the
            indices in 'where'


        'type': type of the comparison to make


    """




res = _scyopt.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
            bounds=None, constraints=(), tol=None, callback=None, options=None)
