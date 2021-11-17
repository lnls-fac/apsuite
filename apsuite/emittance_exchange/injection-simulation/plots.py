import matplotlib.pyplot as plt
import pyaccel as pa
import numpy as np
from pyaccel.optics.twiss import Twiss
from diagnostics import calc_ellipse_equation, sigmax_error


def plot_phase_diagram(bunch, closed_orbit=None, title=""):
    """bunch: bunch array of type (positions, particles)"""
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    ax[0].scatter(bunch[0, :]*1e3, bunch[1, :]*1e3, marker='.', s=2)
    ax[0].set_xlabel('x [mm]')
    ax[0].set_ylabel("x' [mrad]")
    ax[0].grid()
    # ax[0].set_title("Phase Diagram for x")
    ax[1].scatter(bunch[2, :]*1e3, bunch[3, :]*1e3, marker='.', color='r', s=2)
    ax[1].set_xlabel('y [mm]')
    ax[1].set_ylabel("y' [mrad]")
    ax[1].grid()
    # ax[1].set_title("Phase Diagram for y")
    if closed_orbit is not None:
        ax[0].scatter(closed_orbit[0], closed_orbit[1], color='k',
                      marker='x', label='closed orbit')
        ax[1].scatter(closed_orbit[2], closed_orbit[3], color='k',
                      marker='x', label='closed orbit')
    fig.suptitle(title)
    if closed_orbit is not None:
        plt.legend()

    return fig, ax


def plot_phase_ellipse(bunch, local_twiss, eqparams, fig=None, ax=None):
    """."""
    if isinstance(local_twiss, Twiss):
        alpha1, alpha2 = local_twiss.alphax, local_twiss.alphay
        beta1, beta2 = local_twiss.betax, local_twiss.betay
    else:
        alpha1, alpha2 = local_twiss.alpha1, local_twiss.alpha2
        beta1, beta2 = local_twiss.beta1, local_twiss.beta2

    emit1, emit2 = eqparams.emit1, eqparams.emit2
    gamma1, gamma2 = (1 + alpha1**2)/beta1, (1 + alpha2**2)/beta2

    xb, x_b = bunch[0, :], bunch[1, :]  # The bunchs entries onlys serves to
    yb, y_b = bunch[2, :], bunch[3, :]  # give the x or y limits in the graph

    x, x_, eqx = calc_ellipse_equation(gamma1, alpha1, beta1, emit1, xb, x_b)
    y, y_, eqy = calc_ellipse_equation(gamma2, alpha2, beta2, emit2, yb, y_b)

    if (fig is not None) & (ax is not None):
        ax[0].contour(x*1e3, x_*1e3, eqx, [0], linewidths=2)
        ax[1].contour(y*1e3, y_*1e3, eqy, [0], linewidths=2)
    # Without fig and ax arguments, plots only x, x' ellipse
    else:
        plt.contour(x*1e3, x_*1e3, eqx, [0], linewidths=2)


def plot_phase_diagram2(bunch, local_twiss, eqparams, title="",
                        closed_orbit=None, emmit_error=False):
    """Plots phase diagram and the invariant ellipse based in emmitance and
    twiss parameters"""

    fig, ax = plot_phase_diagram(bunch, closed_orbit, title=title)
    plot_phase_ellipse(bunch, local_twiss, eqparams, fig, ax)

    if emmit_error:
        u_sigma, _, _ = sigmax_error(bunch, local_twiss, eqparams,
                                     outprint=False)
        ax[0].legend(['$ \sigma_x$ error = {:.2f} %'.format(u_sigma*100)],
                     loc='center left', bbox_to_anchor=(2.06, 0.5))

    return fig, ax


def plot_xtrajectories(model, part_out, init_idx, final_idx, title=""):
    """."""
    if type(init_idx) is not int:
        init_idx = init_idx[0]
    if type(final_idx) is not int:
        final_idx = final_idx[0]

    spos = pa.lattice.find_spos(model)
    xparticles = part_out[0, :, :]  # select rx for all particles and elements
    plt.figure()
    plt.title(title)
    for i in range(np.shape(part_out)[2]):
        plt.plot(spos[init_idx:final_idx+1], xparticles[i, :])
    plt.xlim(0, spos[final_idx])
    pa.graphics.draw_lattice(model, offset=-0.0011, height=0.001, gca=True)
    plt.show()
