"""Calculate dynamic aperture."""

import numpy as _np

import matplotlib.pyplot as _mpyplot
import matplotlib.gridspec as _mgridspec

import pyaccel.tracking as _pytrack
import pyaccel.lattice as _pylatt

from .base import BaseClass as _BaseClass


class PhaseSpaceParams():
    """."""

    def __init__(self):
        """."""
        self.nrturns = 512
        self.x_min = -12.0e-3
        self.x_max = 0.0
        self.y_min = 0.0
        self.y_max = 4.0e-3
        self.de_min = -0.05
        self.de_max = 0.05
        self.x_nrpts = 20
        self.y_nrpts = 10
        self.de_nrpts = 20
        self.xl_off = 1e-5
        self.yl_off = 1e-5
        self.mom_compact = 1.68e-4
        self.intnux = 49.0
        self.intnuy = 14.0

    def __str__(self):
        """."""
        strng = ''
        strng += 'nrturns      : {:d}\n'.format(self.nrturns)
        strng += 'x_nrpts      : {:d}\n'.format(self.x_nrpts)
        strng += 'y_nrpts      : {:d}\n'.format(self.y_nrpts)
        strng += 'de_nrpts     : {:d}\n'.format(self.de_nrpts)
        strng += 'x_min [m]    : {:.2g}\n'.format(self.x_min)
        strng += 'x_max [m]    : {:.2g}\n'.format(self.x_max)
        strng += 'y_min [m]    : {:.2g}\n'.format(self.y_min)
        strng += 'y_max [m]    : {:.2g}\n'.format(self.y_max)
        strng += 'de_min       : {:.2g}\n'.format(self.de_min)
        strng += 'de_max       : {:.2g}\n'.format(self.de_max)
        strng += 'xl_off [rad] : {:.2g}\n'.format(self.xl_off)
        strng += 'yl_off [rad] : {:.2g}\n'.format(self.yl_off)
        strng += 'mom_compact  : {:.2g}\n'.format(self.mom_compact)
        strng += 'intnux       : {:.2f} (for graphs)\n'.format(self.intnux)
        strng += 'intnuy       : {:.2f} (for graphs)\n'.format(self.intnuy)
        return strng


class PhaseSpace(_BaseClass):
    """."""

    def __init__(self, accelerator):
        """."""
        super().__init__()
        self._acc = accelerator
        self.params = PhaseSpaceParams()
        self.data['x_in'] = _np.array([], dtype=float)
        self.data['y_in'] = _np.array([], dtype=float)
        self.data['de_in'] = _np.array([], dtype=float)
        self.data['x_rout'] = _np.array([], dtype=float)
        self.data['x_lost_turn'] = _np.array([], dtype=int)
        self.data['x_lost_element'] = _np.array([], dtype=int)
        self.data['x_lost_plane'] = _np.array([], dtype=int)
        self.data['de_rout'] = _np.array([], dtype=float)
        self.data['de_lost_turn'] = _np.array([], dtype=int)
        self.data['de_lost_element'] = _np.array([], dtype=int)
        self.data['de_lost_plane'] = _np.array([], dtype=int)
        self.data['de_rout'] = _np.array([], dtype=float)
        self.data['de_lost_turn'] = _np.array([], dtype=int)
        self.data['de_lost_element'] = _np.array([], dtype=int)
        self.data['de_lost_plane'] = _np.array([], dtype=int)
        self.x_freq = dict(
            x=_np.array([], dtype=float),
            y=_np.array([], dtype=float),
            de=_np.array([], dtype=float))
        self.y_freq = dict(
            x=_np.array([], dtype=float),
            y=_np.array([], dtype=float),
            de=_np.array([], dtype=float))

    def do_tracking(self):
        """."""
        x_in = _np.linspace(
            self.params.x_min, self.params.x_max, self.params.x_nrpts)
        y_in = _np.linspace(
            self.params.y_min, self.params.y_max, self.params.y_nrpts)
        de_in = _np.linspace(
            self.params.de_min, self.params.de_max, self.params.de_nrpts)

        if self._acc.cavity_on:
            rfidx = _pylatt.find_indices(
                self._acc, 'pass_method', 'cavity_pass')[0]
            freq0 = self._acc[rfidx].frequency
            orb = _np.squeeze(_pytrack.find_orbit6(self._acc))
        else:
            orb = _np.zeros(6)
            orb[:4] = _np.squeeze(_pytrack.find_orbit4(
                self._acc, energy_offset=0.0))

        rin = _np.tile(orb, (x_in.size, 1)).T
        rin[0, :] += x_in
        rin[1, :] += self.params.xl_off
        rin[3, :] += self.params.yl_off

        out = _pytrack.ring_pass(
            self._acc, rin, nr_turns=self.params.nrturns,
            turn_by_turn=True, parallel=True)

        self.data['x_in'] = x_in
        self.data['x_rout'] = out[0]
        self.data['x_lost_turn'] = out[2]
        self.data['x_lost_element'] = out[3]
        self.data['x_lost_plane'] = out[4]

        rin = _np.tile(orb, (y_in.size, 1)).T
        rin[2, :] += y_in
        rin[1, :] += self.params.xl_off
        rin[3, :] += self.params.yl_off

        out = _pytrack.ring_pass(
            self._acc, rin, nr_turns=self.params.nrturns,
            turn_by_turn=True)

        self.data['y_in'] = y_in
        self.data['y_rout'] = out[0]
        self.data['y_lost_turn'] = out[2]
        self.data['y_lost_element'] = out[3]
        self.data['y_lost_plane'] = out[4]

        rout = _np.zeros((6, de_in.size, self.params.nrturns+1))
        lost_turn = []
        lost_element = []
        lost_plane = []
        for i, den in enumerate(de_in):
            try:
                if self._acc.cavity_on:
                    rdeltaf = self.params.mom_compact * den
                    self._acc[rfidx].frequency = freq0*(1 - rdeltaf)
                    orb = _np.squeeze(_pytrack.find_orbit6(self._acc))
                else:
                    orb = _np.zeros(6)
                    orb[4] = den
                    orb[:4] = _np.squeeze(_pytrack.find_orbit4(
                        self._acc, energy_offset=den))
                orb[1] += self.params.xl_off
                orb[3] += self.params.yl_off
                de_in[i] = orb[4]

                out = _pytrack.ring_pass(
                    self._acc, orb, nr_turns=self.params.nrturns,
                    turn_by_turn=True)
            except _pytrack.TrackingException:
                out = [_np.nan, 0, 0, 0, 'x']

            rout[:, i, :] = out[0]
            lost_turn.append(out[2])
            lost_element.append(out[3])
            lost_plane.append(out[4])

        self.data['de_in'] = de_in
        self.data['de_rout'] = rout
        self.data['de_lost_turn'] = lost_turn
        self.data['de_lost_element'] = lost_element
        self.data['de_lost_plane'] = lost_plane

        if self._acc.cavity_on:
            self._acc[rfidx].frequency = freq0

    def process_data(self):
        """."""
        self.calc_frequencies(plane='x')
        self.calc_frequencies(plane='y')
        self.calc_frequencies(plane='de')

    def calc_frequencies(self, plane='x'):
        """."""
        rout = self.data[plane + '_rout']
        lost_plane = self.data[plane + '_lost_plane']

        fxx, fyy = super()._calc_frequencies(rout, lost_plane)
        if fxx is None:
            return
        self.x_freq[plane] = fxx
        self.y_freq[plane] = fyy

    # Make figures
    def make_figure(
            self, resons=None, orders=3, symmetry=1):
        """."""
        fig = _mpyplot.figure(figsize=(12, 7))
        grid = _mgridspec.GridSpec(2, 3)
        grid.update(
            left=0.1, right=0.98, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.35)
        phx = _mpyplot.subplot(grid[0, 0])
        phy = _mpyplot.subplot(grid[0, 1])
        tune = _mpyplot.subplot(grid[0, 2])
        axx = _mpyplot.subplot(grid[1, 0])
        ayy = _mpyplot.subplot(grid[1, 1])
        ade = _mpyplot.subplot(grid[1, 2])

        routx = self.data['x_rout']
        routy = self.data['y_rout']

        phx.scatter(
            routx[0].ravel()*1e3,
            routx[1].ravel()*1e3,
            c='b', s=2)
        phx.scatter(
            routy[0].ravel()*1e3,
            routy[1].ravel()*1e3,
            c='r', s=2)
        phx.set_xlabel('X [mm]')
        phx.set_ylabel("X' [mrad]")

        phy.scatter(
            routy[2].ravel()*1e3,
            routy[3].ravel()*1e3,
            c='r', s=2)
        phy.scatter(
            routx[2].ravel()*1e3,
            routx[3].ravel()*1e3,
            c='b', s=2)
        phy.set_xlabel('Y [mm]')
        phy.set_ylabel("Y' [mrad]")

        freqxx = self.params.intnux + self.x_freq['x']
        freqyx = self.params.intnuy + self.y_freq['x']
        freqxy = self.params.intnux + self.x_freq['y']
        freqyy = self.params.intnuy + self.y_freq['y']
        freqxe = self.params.intnux + self.x_freq['de']
        freqye = self.params.intnuy + self.y_freq['de']
        tune.plot(freqxx, freqyx, '-ob')
        tune.plot(freqxy, freqyy, '-or')
        tune.plot(freqxe, freqye, '-om')
        tune.set_xlabel(r'$\nu_x$')
        tune.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = tune.axis()
            resons = self.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)
        self.add_resonances_to_axis(tune, resons=resons)

        inx = self.data['x_in']
        iny = self.data['y_in']
        ine = self.data['de_in']
        minx = _np.argmin(_np.abs(inx))
        miny = _np.argmin(_np.abs(iny))
        mine = _np.argmin(_np.abs(ine))
        freqxx -= freqxx[minx]
        freqyx -= freqyx[minx]
        freqxy -= freqxy[miny]
        freqyy -= freqyy[miny]
        freqxe -= freqxe[mine]
        freqye -= freqye[mine]

        axx.plot(inx*1e3, freqxx, 'b-o', label=r'$\nu_x$')
        axx.plot(inx*1e3, freqyx, 'r-o', label=r'$\nu_y$')
        ayy.plot(iny*1e3, freqxy, 'b-o', label=r'$\nu_x$')
        ayy.plot(iny*1e3, freqyy, 'r-o', label=r'$\nu_y$')
        ade.plot(ine*1e2, freqxe, 'b-o', label=r'$\nu_x$')
        ade.plot(ine*1e2, freqye, 'r-o', label=r'$\nu_y$')

        axx.set_xlabel('X [mm]')
        ayy.set_xlabel('Y [mm]')
        ade.set_xlabel(r'$\delta$ [%]')
        axx.set_ylabel(r'$\Delta\nu$')

        axx.legend(loc='best')

        return fig, phx, phy, tune
