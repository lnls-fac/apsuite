import numpy as np
from mathphys.beam_optics import beam_rigidity
import matplotlib.pyplot as plt


def si_nlk_kick(strength=None, fit_monomials=None, plot_flag=False, r0=0.0):

    if fit_monomials is None:
        fit_monomials = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                 dtype=int)
    if strength is None:
        strength = 0.565976805957669

    # NLK model without iron
    maxfield = np.array(
        [[-12.0, -2.4041E-02],
            [-11.5,	-2.9740E-02],
            [-11.0,	-3.5327E-02],
            [-10.5,	-4.0350E-02],
            [-10.0,	-4.4346E-02],
            [-9.5,	-4.6916E-02],
            [-9.0,	-4.7787E-02],
            [-8.5,	-4.6854E-02],
            [-8.0,	-4.4202E-02],
            [-7.5,	-4.0101E-02],
            [-7.0,	-3.4968E-02],
            [-6.5,	-2.9300E-02],
            [-6.0,	-2.3589E-02],
            [-5.5,	-1.8246E-02],
            [-5.0,	-1.3549E-02],
            [-4.5,	-9.6407E-03],
            [-4.0,	-6.5433E-03],
            [-3.5,	-4.2006E-03],
            [-3.0,	-2.5120E-03],
            [-2.5,	-1.3606E-03],
            [-2.0,	-6.3108E-04],
            [-1.5,	-2.1777E-04],
            [-1.0,	-2.7747E-05],
            [-0.5,	2.0482E-05],
            [0.0,	0.0000E+00],
            [0.5,	-2.0482E-05],
            [1.0,	2.7747E-05],
            [1.5,	2.1777E-04],
            [2.0,	6.3108E-04],
            [2.5,	1.3606E-03],
            [3.0,	2.5120E-03],
            [3.5,	4.2006E-03],
            [4.0,	6.5433E-03],
            [4.5,	9.6407E-03],
            [5.0,	1.3549E-02],
            [5.5,	1.8246E-02],
            [6.0,	2.3589E-02],
            [6.5,	2.9300E-02],
            [7.0,	3.4968E-02],
            [7.5,	4.0101E-02],
            [8.0,	4.4202E-02],
            [8.5,	4.6854E-02],
            [9.0,	4.7787E-02],
            [9.5,	4.6916E-02],
            [10.0,	4.4346E-02],
            [10.5,	4.0350E-02],
            [11.0,	3.5327E-02],
            [11.5,	2.9740E-02],
            [12.0,	2.4041E-02]]
    )

    x = maxfield[:, 0]*1e-3
    brho, *_ = beam_rigidity(energy=3e9)
    integ_field = strength * maxfield[:, 1]
    kickx = integ_field / brho

    coeffs = np.polyfit(x=x-r0, y=kickx, deg=np.max(fit_monomials))
    fit_kickx = np.polyval(coeffs, x)

    if plot_flag:
        plt.figure()
        plt.scatter(1e3*(x-r0), 1e3*kickx, label="data points")
        plt.plot(1e3*(x-r0), 1e3*fit_kickx, c=[0, 0.6, 0],
                 label="fitted curve")
        plt.xlabel('X [mm]')
        plt.ylabel("Kick @ 3 GeV [mrad]")
        plt.title("NLK Profile")
        plt.legend()

    LPolyB = np.zeros([1, 1 + np.max(fit_monomials)])
    LPolyB[0, fit_monomials] = - np.flip(coeffs)

    return x, integ_field, kickx, LPolyB
