"""Calculate dynamic aperture."""

import numpy as _np

import matplotlib.pyplot as _mpyplot
import matplotlib.gridspec as _mgridspec
import matplotlib.colors as _mcolors
import matplotlib.text as _mtext

import pyaccel.tracking as _pytrack

from .base import BaseClass as _BaseClass


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
        self.xl_off = 1e-5
        self.yl_off = 1e-5
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
        strng += 'xl_off [rad] : {:.2g}\n'.format(self.xl_off)
        strng += 'yl_off [rad] : {:.2g}\n'.format(self.yl_off)
        strng += 'intnux       : {:.2f} (for graphs)\n'.format(self.intnux)
        strng += 'intnuy       : {:.2f} (for graphs)\n'.format(self.intnuy)
        return strng


class DynapXY(_BaseClass):
    """."""

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

    def do_tracking(self):
        """."""
        x_in, y_in = _np.meshgrid(
            _np.linspace(
                self.params.x_min, self.params.x_max, self.params.x_nrpts),
            _np.linspace(
                self.params.y_min, self.params.y_max, self.params.y_nrpts))

        if self._acc.cavity_on:
            orb = _np.squeeze(_pytrack.find_orbit6(self._acc))
        else:
            orb = _np.zeros(6)
            orb[5] = self.params.de_offset
            orb[:4] = _np.squeeze(_pytrack.find_orbit4(
                self._acc, energy_offset=self.params.de_offset))
        rin = _np.tile(orb, (x_in.size, 1)).T
        rin[0, :] += x_in.ravel()
        rin[1, :] += self.params.xl_off
        rin[2, :] += y_in.ravel()
        rin[3, :] += self.params.yl_off

        out = _pytrack.ring_pass(
            self._acc, rin, nr_turns=self.params.nrturns,
            turn_by_turn=self.params.turn_by_turn, parallel=True)

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

        self.x_dynap, self.y_dynap = self._calc_dynap(x_in, y_in, lost_plane)

    def calc_fmap(self):
        """."""
        rout = self.data['rout']
        lost_plane = self.data['lost_plane']

        fx1, fx2, fy1, fy2, diffx, diffy, diff = super()._calc_fmap(
            rout, lost_plane)

        self.x_freq_ini = fx1
        self.x_freq_fin = fx2
        self.y_freq_ini = fy1
        self.y_freq_fin = fy2
        self.x_diffusion = diffx
        self.y_diffusion = diffy
        self.diffusion = diff

    def map_resons2real_plane(self, resons, maxdist=1e-5, min_diffusion=1e-3):
        """."""
        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini
        diff = self.diffusion
        return super()._map_resons2real_plane(
            freqx, freqy, diff, resons, maxdist=maxdist, mindiff=min_diffusion)

    # Make figures
    def make_figure_diffusion(
            self, contour=True, resons=None, orders=3, symmetry=1,
            maxdist=1e-5, min_diffusion=1e-3):
        """."""
        fig = _mpyplot.figure(figsize=(7, 7))
        grid = _mgridspec.GridSpec(2, 20)
        grid.update(
            left=0.15, right=0.86, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.25)
        axx = _mpyplot.subplot(grid[0, :19])
        ayy = _mpyplot.subplot(grid[1, :19])
        cbaxes = _mpyplot.subplot(grid[:, -1])
        axx.name = 'XY'
        ayy.name = 'Tune'

        diff = self.diffusion
        diff = _np.log10(diff)
        norm = _mcolors.Normalize(vmin=-10, vmax=-2)

        if contour:
            axx.contourf(
                self.data['x_in']*1e3,
                self.data['y_in']*1e3,
                diff.reshape(self.data['x_in'].shape),
                norm=norm, cmap='jet')
        else:
            axx.scatter(
                self.data['x_in'].ravel()*1e3,
                self.data['y_in'].ravel()*1e3,
                c=diff, norm=norm, cmap='jet')
            axx.grid(False)
        axx.set_xlabel('X [mm]')
        axx.set_ylabel('Y [mm]')

        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini
        line = ayy.scatter(
            freqx, freqy, c=diff, norm=norm, cmap='jet')
        ayy.set_xlabel(r'$\nu_x$')
        ayy.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = ayy.axis()
            resons = self.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)

        map2xy = self.map_resons2real_plane(
            resons=resons, maxdist=maxdist, min_diffusion=min_diffusion)
        xdata = self.data['x_in'].ravel()*1e3
        ydata = self.data['y_in'].ravel()*1e3
        for (coefx, coefy, _), ind in zip(resons, map2xy):
            order = int(_np.abs(coefx) + _np.abs(coefy))
            idx = order - 1
            axx.scatter(
                xdata[ind], ydata[ind],
                c=self.COLORS[idx % len(self.COLORS)])

        self.add_resonances_to_axis(ayy, resons=resons)

        cbar = fig.colorbar(line, cax=cbaxes)
        cbar.set_label('Diffusion')
        fig.canvas.mpl_connect('button_press_event', self._onclick)

        ann = ayy.annotate(
            '', xy=(0, 0), xycoords='data',
            xytext=(20, 20), textcoords='offset points',
            arrowprops={'arrowstyle': '->'},
            bbox={'boxstyle': 'round', 'fc': 'w'})
        ann.set_visible(False)
        fig.canvas.mpl_connect('motion_notify_event', self._onhover)

        return fig, axx, ayy

    def _onclick(self, event):
        if not event.dblclick or event.inaxes is None:
            return
        fig = event.inaxes.figure
        ax_nu = [ax for ax in fig.axes if ax.name == 'Tune'][0]
        ax_xy = [ax for ax in fig.axes if ax.name == 'XY'][0]

        xdata = self.data['x_in'].ravel()*1e3
        ydata = self.data['y_in'].ravel()*1e3
        xfreq = self.params.intnux + self.x_freq_ini
        yfreq = self.params.intnuy + self.y_freq_ini

        if event.inaxes == ax_nu:
            xdiff = xfreq - event.xdata
            ydiff = yfreq - event.ydata
        if event.inaxes == ax_xy:
            xdiff = xdata - event.xdata
            ydiff = ydata - event.ydata

        ind = _np.nanargmin(_np.sqrt(xdiff*xdiff + ydiff*ydiff))
        ax_xy.scatter(xdata[ind], ydata[ind], c='k')
        ax_nu.scatter(xfreq[ind], yfreq[ind], c='k')
        fig.canvas.draw()

    def _onhover(self, event):
        if event.inaxes is None:
            return
        fig = event.inaxes.figure
        ax_nu = [ax for ax in fig.axes if ax.name == 'Tune'][0]
        chil = ax_nu.get_children()
        ann = [c for c in chil if isinstance(c, _mtext.Annotation)][0]
        if event.inaxes != ax_nu:
            ann.set_visible(False)
            fig.canvas.draw()
            return

        for line in ax_nu.lines:
            if line.contains(event)[0] and line.name.startswith('reson'):
                ann.set_text(line.name.split(':')[1])
                ann.xy = (event.xdata, event.ydata)
                ann.set_visible(True)
                break
        else:
            ann.set_visible(False)
        fig.canvas.draw()

    def make_figure_map_real2tune_planes(
            self, resons=None, orders=3, symmetry=1):
        """."""
        fig = _mpyplot.figure(figsize=(7, 7))
        grid = _mgridspec.GridSpec(2, 20)
        grid.update(
            left=0.15, right=0.86, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.25)
        axx = _mpyplot.subplot(grid[0, :19])
        ayy = _mpyplot.subplot(grid[1, :19])
        cbx = _mpyplot.subplot(grid[0, -1])
        cby = _mpyplot.subplot(grid[1, -1])

        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini

        # X
        norm = _mcolors.Normalize(
            vmin=self.params.x_min*1e3,
            vmax=self.params.x_max*1e3)
        line = axx.scatter(
            freqx, freqy, c=self.data['x_in'].ravel()*1e3,
            norm=norm, cmap='jet')
        axx.set_xlabel(r'$\nu_x$')
        axx.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = axx.axis()
            resons = self.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)

        self.add_resonances_to_axis(axx, resons=resons)

        cbar = fig.colorbar(line, cax=cbx)
        cbar.set_label('X [mm]')

        # Y
        norm = _mcolors.Normalize(
            vmin=self.params.y_min*1e3,
            vmax=self.params.y_max*1e3)
        line = ayy.scatter(
            freqx, freqy, c=self.data['y_in'].ravel()*1e3,
            norm=norm, cmap='jet')
        ayy.set_xlabel(r'$\nu_x$')
        ayy.set_ylabel(r'$\nu_y$')
        self.add_resonances_to_axis(ayy, resons=resons)

        cbar = fig.colorbar(line, cax=cby)
        cbar.set_label('Y [mm]')

        return fig, axx, ayy
