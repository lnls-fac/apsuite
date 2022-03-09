import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.constants import vacuum_permeability
from mathphys.beam_optics import beam_rigidity


class Wire:
    """."""
    def __init__(self, position, current):
        self.pos = position  # Column vector
        self.curr = current

    def Bfield(self, x, y):
        mu0 = vacuum_permeability
        r_w = self.pos  # Wire positions
        r = _np.array([x, y])[:, None]  # Target position
        rc = r - r_w  # Cursive r
        theta = _np.arctan2(rc[1], rc[0])[0]
        theta_vec = _np.array([-_np.sin(theta), _np.cos(theta)])[:, None]
        B = mu0 * self.curr/(2*_np.pi*_np.linalg.norm(rc))*theta_vec + r_w
        return B

    def include_pos_error(self, sigma):
        self.error = _np.random.normal(loc=0, scale=sigma)
        self.pos = self.pos + self.error

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, position):
        self._pos = position

    @property
    def curr(self):
        return self._curr

    @curr.setter
    def curr(self, current):
        self._curr = current


class NLK:
    """."""
    def __init__(self, curr=1850, errors=False, sigma_errors=1e-5):
        # Setting wire positions
        wire_positions = []
        wire_positions.append(_np.array([7.0,  5.21])[:, None]*1e-3)
        wire_positions.append(_np.array([10.,  5.85])[:, None]*1e-3)
        wire_positions.append(_np.array([-7.,  5.21])[:, None]*1e-3)
        wire_positions.append(_np.array([-10,  5.85])[:, None]*1e-3)
        wire_positions.append(_np.array([7.0, -5.21])[:, None]*1e-3)
        wire_positions.append(_np.array([10., -5.85])[:, None]*1e-3)
        wire_positions.append(_np.array([-7., -5.21])[:, None]*1e-3)
        wire_positions.append(_np.array([-10, -5.85])[:, None]*1e-3)
        self._positions = wire_positions

        # Setting wire currents
        currents = []
        for i in range(8):
            if i % 2:
                currents.append(-curr)
            else:
                currents.append(curr)
        self._currents = currents

        # Creating wires
        self._wires = []
        for i in range(8):
            pos = wire_positions[i]
            curr = currents[i]
            wire = Wire(pos, curr)
            if errors:
                wire.include_pos_error(sigma=sigma_errors)
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
        return self._positions

    @positions.setter
    def positions(self, wires_positions):
        self._positions = wires_positions

    @property
    def currents(self):
        "Return currents of all wires"
        return self._currents

    @currents.setter
    def currents(self, wires_currents):
        self._currents = wires_currents

    def Bfield(self, x, y):
        field = _np.zeros([2, 1])
        for wire in self.wires:
            field += wire.Bfield(x, y)
        return field

    def include_pos_error(self, sigma_errors):
        for wire in self.wires:
            wire.include_pos_error(sigma_errors)

    def nlk_profile(self):
        "NLK vertical field at y=0 for x ∈ [-12, 12] mm."
        x_space = _np.linspace(-12, 12)*1e-3
        fieldy = _np.array([self.Bfield(x, 0)[1] for x in x_space])
        return x_space, fieldy


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

    nlk = NLK(errors=errors, sigma_errors=sigma_errors)
    x, B = nlk.nlk_profile()

    brho, *_ = beam_rigidity(energy=3)  # Energy in GeV
    integ_field = strength * B
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
