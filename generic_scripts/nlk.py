import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.constants import vacuum_permeability
from mathphys.beam_optics import beam_rigidity


class Wire:
    """."""
    def __init__(self, position, current):
        self.position = position  # Column vector
        self.current = current

    def calc_magnetic_field(self, r):
        if r.shape[0] != 2:
            raise ValueError(
                "The input position vector r must have shape of type (2, N)")
        r = r.reshape(2, -1)
        mu0 = vacuum_permeability
        r_w = self.position[:, None]  # Wire positions
        rc = r - r_w  # Cursive r
        theta = _np.arctan2(rc[1, :], rc[0, :])
        theta_vec = _np.array([-_np.sin(theta), _np.cos(theta)])
        rc_norm = _np.linalg.norm(rc, axis=0)[None, :]
        mag_field = \
            mu0 * self.current/(2*_np.pi*rc_norm)*theta_vec + r_w
        return mag_field

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, position):
        position = self._check_r_size(position)
        self._pos = position

    @property
    def current(self):
        return self._curr

    @current.setter
    def current(self, current):
        self._curr = current

    @staticmethod
    def _check_r_size(r):
        if r.size == 2:
            return r.ravel()
        else:
            raise ValueError('r must be a numpy array of size 2.')


class NLK:
    """."""
    def __init__(self, positions=None, curr=1850):

        if positions is not None:
            wire_positions = positions
        else:
            wire_positions = _np.zeros([8, 2])
            s1 = _np.array([1, -1, 1, -1])
            s2 = _np.array([1, 1, -1, -1])
            for i in range(0, wire_positions.shape[0], 2):
                j = int(i/2)
                wire_positions[i] = \
                    _np.array([s1[j]*7.0,  s2[j]*5.21])[None, :]*1e-3
                wire_positions[i+1] = \
                    _np.array([s1[j]*10., s2[j]*5.85])[None, :]*1e-3

        # Setting wire currents
        currents = _np.zeros(8)
        for i in range(currents.size):
            if i % 2:
                currents[i] = -curr
            else:
                currents[i] = curr

        # Creating wires
        self._wires = []
        for i in range(8):
            pos = wire_positions[i]
            curr = currents[i]
            wire = Wire(pos, curr)
            self._wires.append(wire)

    @property
    def wires(self):
        return self._wires

    @wires.setter
    def wires(self, wires_objects):
        self._wires = wires_objects

    @property
    def positions(self):
        "Return positions of all wires"
        wires_positions = _np.zeros([2, 8])
        for i in range(wires_positions.shape[1]):
            wires_positions[:, [i]] = self._wires[i].position[:, None]
        return wires_positions

    @positions.setter  # Thinking in a good setter for this...
    def positions(self, wires_positions):
        for i in range(wires_positions.shape[1]):
            self._wires[i].position = wires_positions[:, i]

    @property
    def currents(self):
        "Return the currents of all wires"
        wires_currents = _np.zeros(len(self._wires))
        for i in range(len(self._wires)):
            wires_currents[i] = self._wires[i].current
        return wires_currents

    @currents.setter
    def currents(self, wires_currents):
        for i, current in enumerate(wires_currents):
            self._wires[i].current = current

    def calc_magnetic_field(self, r):
        if r.shape[0] != 2:
            raise ValueError(
                "The input position vector r must have shape of type (2, N)")
        mag_field = _np.zeros(r.shape)
        for wire in self.wires:
            mag_field += wire.calc_magnetic_field(r)
        return mag_field

    def get_magnetic_field_on_axis(self):
        "NLK vertical field at y=0 for x ∈ [-12, 12] mm."
        x_space = _np.linspace(-12, 12)[None, :]*1e-3
        y_space = _np.zeros(x_space.shape)
        r = _np.concatenate([x_space, y_space], axis=0)
        fieldy = self.calc_magnetic_field(r)[1, :]
        return x_space[0], fieldy


def polyfit(x, y, n):
    x = x.reshape([-1, 1])  # transforms into a column vector
    y = y.reshape([-1, 1])
    n = _np.array(n).reshape([1, -1])  # transforms into a row vector

    xn = x**n
    b = _np.dot(xn.T, y)
    X = _np.dot(xn.T, xn)
    coeffs = _np.linalg.solve(X, b)
    y_fit = _np.dot(xn, coeffs)

    return coeffs, y_fit


def si_nlk_kick(
        strength=None, fit_monomials=None, plot_flag=False,
        r0=0.0, errors=False, sigma_errors=1e-5):
    """Generates the nlk integrated polynom_b and its horizontal
    kick."""

    if fit_monomials is None:
        fit_monomials = _np.arange(10, dtype=int)
    if strength is None:
        strength = 0.27358145

    nlk = NLK()
    x, mag_field = nlk.get_magnetic_field_on_axis()

    brho, *_ = beam_rigidity(energy=3)  # Energy in GeV
    integ_field = strength * mag_field
    kickx = integ_field / brho

    coeffs, fit_kickx = polyfit(x=x-r0, y=kickx, n=fit_monomials)

    if plot_flag:
        _plt.figure()
        _plt.scatter(1e3*(x-r0), 1e3*kickx, label="data points")
        _plt.plot(
            1e3*(x-r0), 1e3*fit_kickx, c=[0, 0.6, 0], label="fitted curve")
        _plt.xlabel('X [mm]')
        _plt.ylabel("Kick @ 3 GeV [mrad]")
        _plt.title("NLK Profile")
        _plt.legend()

    LPolyB = _np.zeros([1, 1 + _np.max(fit_monomials)])
    LPolyB[0][fit_monomials] = -coeffs[:, 0]
    return x, integ_field, kickx, LPolyB
