
import numpy as np
from pyaccel.optics.twiss import Twiss


def sigmax_error(bunch, local_twiss, eq_params, outprint=True):

    if isinstance(local_twiss, Twiss):
        beta1 = local_twiss.betax
        eta1 = local_twiss.etax
    else:
        beta1 = local_twiss.beta1
        eta1 = local_twiss.eta1

    emit1 = eq_params.emit1
    sigmae = eq_params.espread0

    sigmax_track = np.sqrt(np.nanvar(bunch[0, :]))
    sigmax_model = np.sqrt(emit1 * beta1 + (sigmae * eta1)**2)
    relative_error = abs(sigmax_model - sigmax_track)/sigmax_model

    if outprint:
        print('Sigma_x from tracking = ', sigmax_track)
        print("Sigma_x from model = ", sigmax_model)
        print("Relative sigmax error = ", 100*relative_error, " %")
    else:
        return relative_error, sigmax_track, sigmax_model


def calc_ellipse_equation(gamma, alpha, beta, emit, xbase, x_base):
    xmax, xmin = np.nanmax(xbase), np.nanmin(xbase)
    x_max, x_min = np.nanmax(x_base), np.nanmin(x_base)
    xrange = np.linspace(xmin, xmax, num=100)
    x_range = np.linspace(x_min, x_max, num=100)  # x_ = x'
    x, x_ = np.meshgrid(xrange, x_range)
    equation = gamma * x**2 + 2*alpha*x*x_ + beta*x_**2 - emit
    # The ellipse lies in equation = 0
    return x, x_, equation
