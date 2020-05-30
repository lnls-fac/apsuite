"""Calculate dynamic aperture."""

import numpy as _np

import pyaccel.naff as _pynaff

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
                line = cls.add_reson_line(
                    axes, const=coeffc/coeffy, slope=-coeffx/coeffy,
                    color=cor, linewidth=lwid)
            else:
                line = cls.add_reson_line(
                    axes, const=coeffc/coeffx, slope=None,
                    color=cor, linewidth=lwid)
            if not coeffx:
                line.name = f'reson: {coeffy}y = {coeffc}'
            elif not coeffy:
                line.name = f'reson: {coeffx}x = {coeffc}'
            else:
                sig = '+' if coeffy > 0 else '-'
                line.name = \
                    f'reson: {coeffx}x {sig} {abs(coeffy)}y = {coeffc}'

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
        line = axes.plot([x11, x22], [y11, y22], **kwargs)[0]
        axes.set_xlim(axis[:2])
        axes.set_ylim(axis[2:])
        return line

    @staticmethod
    def _calc_dynap(x_in, y_in, lost_plane):
        """."""
        shape = x_in.shape
        nlost = _np.array([l is None for l in lost_plane], dtype=bool)
        nlost = nlost.reshape(shape)
        r_sqr = x_in*x_in + y_in*y_in
        idx = _np.unravel_index(_np.argmin(r_sqr), r_sqr.shape)
        tolook = set()
        tolook.add(idx)
        inner = set(idx)
        border = set()
        looked = set()
        neigbs = [
            (-1, 0), (0, -1), (0, 1), (1, 0),
            # uncomment these lines to include diagonal neighboors:
            # (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        while tolook:
            idx = tolook.pop()
            isborder = False
            for nei in neigbs:
                idxn = idx[0] + nei[0], idx[1] + nei[1]
                if 0 <= idxn[0] < shape[0] and 0 <= idxn[1] < shape[1]:
                    if nlost[idxn]:
                        if idxn not in looked:
                            tolook.add(idxn)
                            inner.add(idxn)
                    else:
                        isborder = True
            if isborder:
                border.add(idx)
            looked.add(idx)

        # tuple(zip(*border)) transforms:
        #     ((x1, y1), ..., (xn, yn)) --> ((x1, ..., xn), (y1, ..., yn))
        border = tuple(zip(*border))
        x_dyn = x_in[border]
        y_dyn = y_in[border]

        r_sqr = x_dyn*x_dyn + y_dyn*y_dyn
        theta = _np.arctan2(y_dyn, x_dyn)

        tup = _np.argsort(zip(theta, r_sqr))
        return x_dyn[tup], y_dyn[tup]

    @staticmethod
    def _calc_frequencies(rout, lost_plane):
        """."""
        if not rout.size or len(rout.shape) < 3:
            return None, None

        nmult = rout.shape[2] // 6
        left = rout.shape[2] % 6
        if nmult < 5:
            return None, None
        if left < 1:
            nmult -= 1

        nlost = _np.array([l is None for l in lost_plane], dtype=bool)

        nt_ini = nmult * 6 + 1
        x_ini = rout[0, :, :nt_ini]
        y_ini = rout[2, :, :nt_ini]

        x_freq = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        y_freq = _np.full(x_ini.shape[0], _np.nan, dtype=float)

        x_ini = x_ini[nlost, :]
        y_ini = y_ini[nlost, :]

        x_ini -= x_ini.mean(axis=1)[:, None]
        y_ini -= y_ini.mean(axis=1)[:, None]

        fx1, _ = _pynaff.naff_general(x_ini, nr_ff=1)
        fy1, _ = _pynaff.naff_general(y_ini, nr_ff=1)
        fx1 = _np.abs(fx1)
        fy1 = _np.abs(fy1)

        x_freq[nlost] = fx1
        y_freq[nlost] = fy1
        return x_freq, y_freq

    @staticmethod
    def _calc_fmap(rout, lost_plane):
        """."""
        if not rout.size or len(rout.shape) < 3:
            return 7*[None, ]

        nmult = rout.shape[2] // 12
        left = rout.shape[2] % 12
        if nmult < 5:
            return 7*[None, ]
        if left < 2:
            nmult -= 1

        nlost = _np.array([l is None for l in lost_plane], dtype=bool)

        nt_ini = nmult * 6 + 1
        nt_fin = nmult * 12 + 2
        x_ini = rout[0, :, :nt_ini]
        x_fin = rout[0, :, nt_ini:nt_fin]
        y_ini = rout[2, :, :nt_ini]
        y_fin = rout[2, :, nt_ini:nt_fin]

        x_freq_ini = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        x_freq_fin = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        y_freq_ini = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        y_freq_fin = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        x_diffusion = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        y_diffusion = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        diffusion = _np.full(x_ini.shape[0], _np.nan, dtype=float)

        x_ini = x_ini[nlost, :]
        x_fin = x_fin[nlost, :]
        y_ini = y_ini[nlost, :]
        y_fin = y_fin[nlost, :]

        x_ini -= x_ini.mean(axis=1)[:, None]
        x_fin -= x_fin.mean(axis=1)[:, None]
        y_ini -= y_ini.mean(axis=1)[:, None]
        y_fin -= y_fin.mean(axis=1)[:, None]

        fx1, _ = _pynaff.naff_general(x_ini, nr_ff=1)
        fx2, _ = _pynaff.naff_general(x_fin, nr_ff=1)
        fy1, _ = _pynaff.naff_general(y_ini, nr_ff=1)
        fy2, _ = _pynaff.naff_general(y_fin, nr_ff=1)
        fx1 = _np.abs(fx1)
        fx2 = _np.abs(fx2)
        fy1 = _np.abs(fy1)
        fy2 = _np.abs(fy2)

        diffx = _np.abs(fx1 - fx2)
        diffy = _np.abs(fy1 - fy2)
        diff = _np.sqrt(diffx*diffx + diffy*diffy)

        x_freq_ini[nlost] = fx1
        x_freq_fin[nlost] = fx2
        y_freq_ini[nlost] = fy1
        y_freq_fin[nlost] = fy2
        x_diffusion[nlost] = diffx
        y_diffusion[nlost] = diffy
        diffusion[nlost] = diff

        return x_freq_ini, x_freq_fin, y_freq_ini, y_freq_fin,\
            x_diffusion, y_diffusion, diffusion

    @staticmethod
    def _map_resons2real_plane(
            freqx, freqy, diff, resons, maxdist=1e-5, mindiff=1e-3):
        """."""
        indcs = []
        if maxdist is None or mindiff is None:
            return indcs

        ind = ~_np.isnan(freqx)

        ind1 = diff[ind] > mindiff
        freqx = freqx[ind]
        freqy = freqy[ind]
        idcs = ind.nonzero()[0]
        for coefx, coefy, coefc in resons:
            if coefy == 0:
                dist_to_reson = _np.abs(coefc/coefx - freqx)
            else:
                tan_theta = -coefx/coefy
                reson_y0 = coefc/coefy
                parallel_y0 = freqy - tan_theta*freqx
                delta_y0 = reson_y0 - parallel_y0
                dist_to_reson = _np.abs(delta_y0) / _np.sqrt(1 + tan_theta**2)
            ind2 = _np.logical_and(ind1, dist_to_reson < maxdist).nonzero()
            indcs.append(idcs[ind2])
        return indcs
