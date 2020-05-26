"""Calculate dynamic aperture."""

import numpy as _np

from ..commissioning_scripts import BaseClass as _BaseClass


class BaseClass(_BaseClass):
    """."""

    COLORS = ('k', 'b', 'r', 'g', 'm', 'c')

    def __str__(self):
        """."""
        return str(self.params)

    # class methods
    @classmethod
    def calc_resonances_for_bounds(cls, bounds, orders=3, symmetry=1):
        """."""
        orders = _np.asarray(orders)
        if not orders.shape:
            orders = _np.arange(1, orders+1)

        axis = _np.asarray(bounds).reshape(2, -1)
        points = _np.zeros((2, 4))
        points[:2, :2] = axis
        points[0, 2:] = _np.flip(axis[0])
        points[1, 2:] = axis[1]

        resons = []
        for order in orders:
            resons.extend(
                cls._calc_resonances_fixed_order(points, order, symmetry))

        # Unique resonances:
        ress = set()
        for reson in resons:
            gcd = _np.gcd.reduce(reson)
            if gcd > 1:
                reson = (reson[0]//gcd, reson[1]//gcd, reson[2]//gcd)
            ress.add(reson)
        resons = list(ress)
        return resons

    @staticmethod
    def _calc_resonances_fixed_order(points, order=3, symmetry=1):
        """."""
        points = _np.asarray(points)
        if points.shape[0] != 2:
            if points.shape[1] == 2:
                points = points.T
            else:
                raise TypeError('wrong number of dimensions for points.')

        ang_coeffs = _np.zeros((2*order + 1, 2), dtype=int)
        ang_coeffs[:, 0] = _np.arange(-order, order + 1)
        ang_coeffs[:, 1] = order - _np.abs(ang_coeffs[:, 0])

        consts = _np.dot(ang_coeffs, points)
        consts_min = _np.array(_np.ceil(consts.min(axis=1)), dtype=int)
        consts_max = _np.array(_np.floor(consts.max(axis=1)), dtype=int)
        resons = []
        for ang_i, c_min, c_max in zip(ang_coeffs, consts_min, consts_max):
            cons = _np.arange(c_min, c_max+1)
            for c_i in cons:
                if not c_i % symmetry:
                    resons.append((ang_i[0], ang_i[1], c_i))
        return resons

    @classmethod
    def add_resonances_to_axis(cls, axes, resons=None, orders=3, symmetry=1):
        """."""
        if resons is None:
            bounds = axes.axis()
            resons = cls.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)

        for coeffx, coeffy, coeffc in resons:
            order = int(_np.abs(coeffx) + _np.abs(coeffy))
            idx = order - 1
            cor = cls.COLORS[idx % len(cls.COLORS)]
            lwid = max(3-idx, 1)
            if coeffy:
                cls.add_reson_line(
                    axes, const=coeffc/coeffy, slope=-coeffx/coeffy,
                    color=cor, linewidth=lwid)
            else:
                cls.add_reson_line(
                    axes, const=coeffc/coeffx, slope=None,
                    color=cor, linewidth=lwid)

    @staticmethod
    def add_reson_line(axes, const=0, slope=None, **kwargs):
        """."""
        axis = axes.axis()
        if slope is not None:
            x11, x22 = axis[:2]
            y11, y22 = slope*x11 + const, slope*x22 + const
        else:
            x11, x22 = const, const
            y11, y22 = axis[2:]
        line = axes.plot([x11, x22], [y11, y22], **kwargs)
        axes.set_xlim(axis[:2])
        axes.set_ylim(axis[2:])
        return line
