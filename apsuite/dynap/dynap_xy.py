"""Calculate dynamic aperture."""

import numpy as _np

import matplotlib.pyplot as _mpyplot
import matplotlib.gridspec as _mgridspec
import matplotlib.colors as _mcolors

import pyaccel.tracking as _pytrack
import pyaccel.naff as _pynaff

from ..commissioning_scripts import BaseClass as _BaseClass


class DynapXYParams():
    """."""

    def __init__(self):
        """."""
        self.nrturns = 512
        self.turn_by_turn = True
        self.x_min = -12.0e-3
        self.x_max = 0.0
        self.y_min = 0.0
        self.y_max = 4.0e-3
        self.x_nrpts = 70
        self.y_nrpts = 30
        self.de_offset = 0.0
        self.intnux = 49.0
        self.intnuy = 14.0

    def __str__(self):
        """."""
        strng = ''
        strng += 'nrturns      : {:d}\n'.format(self.nrturns)
        strng += 'turn_by_turn : {:s}\n'.format(str(self.turn_by_turn))
        strng += 'x_nrpts      : {:d}\n'.format(self.x_nrpts)
        strng += 'y_nrpts      : {:d}\n'.format(self.y_nrpts)
        strng += 'x_min [m]    : {:.2g}\n'.format(self.x_min)
        strng += 'x_max [m]    : {:.2g}\n'.format(self.x_max)
        strng += 'y_min [m]    : {:.2g}\n'.format(self.y_min)
        strng += 'y_max [m]    : {:.2g}\n'.format(self.y_max)
        strng += 'de_offset    : {:.3g}\n'.format(self.de_offset)
        strng += 'intnux       : {:.2f} (for graphs)\n'.format(self.intnux)
        strng += 'intnuy       : {:.2f} (for graphs)\n'.format(self.intnuy)
        return strng


class DynapXY(_BaseClass):
    """."""

    COLORS = ('k', 'b', 'r', 'g', 'm', 'c')

    def __init__(self, accelerator):
        """."""
        super().__init__()
        self._acc = accelerator
        self.params = DynapXYParams()
        self.data['x_in'] = _np.array([], dtype=float)
        self.data['y_in'] = _np.array([], dtype=float)
        self.data['rout'] = _np.array([], dtype=float)
        self.data['lost_turn'] = _np.array([], dtype=int)
        self.data['lost_element'] = _np.array([], dtype=int)
        self.data['lost_plane'] = _np.array([], dtype=int)
        self.x_dynap = _np.array([], dtype=float)
        self.y_dynap = _np.array([], dtype=float)
        self.x_freq_ini = _np.array([], dtype=float)
        self.x_freq_fin = _np.array([], dtype=float)
        self.y_freq_ini = _np.array([], dtype=float)
        self.y_freq_fin = _np.array([], dtype=float)
        self.x_diffusion = _np.array([], dtype=float)
        self.y_diffusion = _np.array([], dtype=float)
        self.diffusion = _np.array([], dtype=float)

    def __str__(self):
        """."""
        return str(self.params)

    def do_tracking(self):
        """."""
        x_in, y_in = _np.meshgrid(
            _np.linspace(
                self.params.x_min, self.params.x_max, self.params.x_nrpts),
            _np.linspace(
                self.params.y_min, self.params.y_max, self.params.y_nrpts))

        if self._acc.cavity_on:
            orb = _pytrack.find_orbit6(self._acc)
        else:
            orb = _np.zeros(6)
            orb[5] = self.params.de_offset
            orb[:4] = _pytrack.find_orbit4(
                self._acc, energy_offset=self.params.de_offset)
        rin = _np.tile(orb, (1, x_in.size))
        rin[0, :] += x_in.ravel()
        rin[2, :] += y_in.ravel()
        rin[1, :] += 1e-6
        rin[3, :] += 1e-6

        out = _pytrack.ring_pass(
            self._acc, rin, nr_turns=self.params.nrturns,
            turn_by_turn=self.params.turn_by_turn)

        self.data['x_in'] = x_in
        self.data['y_in'] = y_in
        self.data['rout'] = out[0]
        self.data['lost_turn'] = out[2]
        self.data['lost_element'] = out[3]
        self.data['lost_plane'] = out[4]

    def process_data(self):
        """."""
        self.calc_dynap()
        self.calc_fmap()

    def calc_dynap(self):
        """."""
        x_in = self.data['x_in']
        y_in = self.data['y_in']
        lost_plane = self.data['lost_plane']

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

        self.x_dynap = x_dyn[tup]
        self.y_dynap = y_dyn[tup]

    def calc_fmap(self):
        """."""
        rout = self.data['rout']
        lost_plane = self.data['lost_plane']

        if not rout.size or len(rout.shape) < 3:
            return

        nmult = rout.shape[2] // 12
        left = rout.shape[2] % 12
        if nmult < 5:
            return
        if left < 2:
            nmult -= 1

        nlost = _np.array([l is None for l in lost_plane], dtype=bool)

        nt_ini = nmult * 6 + 1
        nt_fin = nmult * 12 + 2
        x_ini = rout[0, :, :nt_ini]
        x_fin = rout[0, :, nt_ini:nt_fin]
        y_ini = rout[2, :, :nt_ini]
        y_fin = rout[2, :, nt_ini:nt_fin]

        self.x_freq_ini = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        self.x_freq_fin = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        self.y_freq_ini = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        self.y_freq_fin = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        self.x_diffusion = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        self.y_diffusion = _np.full(x_ini.shape[0], _np.nan, dtype=float)
        self.diffusion = _np.full(x_ini.shape[0], _np.nan, dtype=float)

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

        self.x_freq_ini[nlost] = fx1
        self.x_freq_fin[nlost] = fx2
        self.y_freq_ini[nlost] = fy1
        self.y_freq_fin[nlost] = fy2
        self.x_diffusion[nlost] = diffx
        self.y_diffusion[nlost] = diffy
        self.diffusion[nlost] = diff

    def map_resons_to_xyplane(self, resons, maxdist=1e-5):
        """."""
        xdata = self.data['x_in'].ravel()*1e3
        ydata = self.data['y_in'].ravel()*1e3
        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini

        ind = xdata != _np.nan
        xdata = xdata[ind]
        ydata = ydata[ind]
        freqx = freqx[ind]
        freqy = freqy[ind]

        data = []
        for coefx, coefy, coefc in resons:
            if coefy == 0:
                dist_to_reson = _np.abs(coefc/coefx - freqx)
            else:
                tan_theta = -coefx/coefy
                reson_y0 = coefc/coefy
                parallel_y0 = freqy - tan_theta*freqx
                delta_y0 = reson_y0 - parallel_y0
                dist_to_reson = _np.abs(delta_y0) / _np.sqrt(1 + tan_theta**2)

            ind = (dist_to_reson < maxdist).nonzero()
            data.append((xdata[ind], ydata[ind]))
        return data

    # Make figures
    def make_figure_diffusion(
            self, contour=True, resons=None, orders=3, symmetry=1,
            maxdist=1e-5):
        """."""
        fig = _mpyplot.figure(figsize=(7, 7))
        gs = _mgridspec.GridSpec(2, 20)
        gs.update(
            left=0.15, right=0.86, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.25)
        ax = _mpyplot.subplot(gs[0, :19])
        ay = _mpyplot.subplot(gs[1, :19])
        cbaxes = _mpyplot.subplot(gs[:, -1])

        diff = self.diffusion
        diff = _np.log10(diff)
        norm = _mcolors.Normalize(vmin=-10, vmax=-2)

        if contour:
            ax.contourf(
                self.data['x_in']*1e3,
                self.data['y_in']*1e3,
                diff.reshape(self.data['x_in'].shape),
                norm=norm, cmap='jet')
        else:
            ax.scatter(
                self.data['x_in'].ravel()*1e3,
                self.data['y_in'].ravel()*1e3,
                c=diff, norm=norm, cmap='jet')
            ax.grid(False)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')

        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini
        line = ay.scatter(
            freqx, freqy, c=diff, norm=norm, cmap='jet')
        ay.set_xlabel(r'$\nu_x$')
        ay.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = ay.axis()
            resons = self.calc_resonances_for_bounds(
                    bounds, orders=orders, symmetry=symmetry)

        map2xy = self.map_resons_to_xyplane(resons=resons, maxdist=maxdist)

        for (coefx, coefy, coefc), (xdata, ydata) in zip(resons, map2xy):
            order = int(_np.abs(coefx) + _np.abs(coefy))
            idx = order - 1
            ax.scatter(
                xdata, ydata,
                c=self.COLORS[idx % len(self.COLORS)])

        self.add_resonances_to_axis(ay, resons=resons)

        cbar = fig.colorbar(line, cax=cbaxes)
        cbar.set_label('Diffusion')

        return fig, ax, ay

    def make_figure_xandy_map_in_tune_plot(
            self, resons=None, orders=3, symmetry=1):
        """."""
        fig = _mpyplot.figure(figsize=(7, 7))
        gs = _mgridspec.GridSpec(2, 20)
        gs.update(
            left=0.15, right=0.86, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.25)
        ax = _mpyplot.subplot(gs[0, :19])
        ay = _mpyplot.subplot(gs[1, :19])
        cbx = _mpyplot.subplot(gs[0, -1])
        cby = _mpyplot.subplot(gs[1, -1])

        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini

        # X
        norm = _mcolors.Normalize(
            vmin=self.params.x_min*1e3,
            vmax=self.params.x_max*1e3)
        line = ax.scatter(
            freqx, freqy, c=self.data['x_in'].ravel()*1e3,
            norm=norm, cmap='jet')
        ax.set_xlabel(r'$\nu_x$')
        ax.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = ax.axis()
            resons = self.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)

        self.add_resonances_to_axis(ax, resons=resons)

        cbar = fig.colorbar(line, cax=cbx)
        cbar.set_label('X [mm]')

        # Y
        norm = _mcolors.Normalize(
            vmin=self.params.y_min*1e3,
            vmax=self.params.y_max*1e3)
        line = ay.scatter(
            freqx, freqy, c=self.data['y_in'].ravel()*1e3,
            norm=norm, cmap='jet')
        ay.set_xlabel(r'$\nu_x$')
        ay.set_ylabel(r'$\nu_y$')
        self.add_resonances_to_axis(ay, resons=resons)

        cbar = fig.colorbar(line, cax=cby)
        cbar.set_label('Y [mm]')

        return fig, ax, ay

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
