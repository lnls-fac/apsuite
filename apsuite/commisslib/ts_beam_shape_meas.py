"""."""

import time as _time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
from matplotlib.patches import Ellipse

import pyaccel
from siriuspy.devices import Screen
from apsuite.utils import MeasBaseClass
from pymodels import ts as pyts, si as pysi


35146337

class Params:

    def __init__(self):
        self.optics_mode = 'M1'
        self.coupling = 0.1  # fractional number
        self.emit0 = 3.5e-9  # m.rad
        self.energyspread = 8.7e-4  # relative value
        self.energy = 3e9  # eV
        self.plot_sigma_lim = 1  # mm
        self.sig_gauss_screen = 0.03  # mm


class BeamShapeTS(MeasBaseClass):

    def __init__(self):
        """."""
        super().__init__(params=Params())
        for i in range(1, 7):
            self.devices[i] = Screen(getattr(Screen.DEVICES, f'TS_{i:d}'))

        # create model
        si_mod = pysi.create_accelerator()
        idx = pyaccel.lattice.find_indices(si_mod, 'fam_name', 'InjSeptF')[0]
        si_mod = pyaccel.lattice.shift(si_mod, idx)
        self._si_mod = si_mod

    def get_data(self):
        """."""
        sigx, sigy, angl = [], [], []
        for i in range(1, 7):
            sigx.append(self.devices[i].sigmax)
            sigy.append(self.devices[i].sigmay)
            angl.append(self.devices[i].angle)
        self.data['sigmax'] = np.array(sigx)
        self.data['sigmay'] = np.array(sigy)
        self.data['angle'] = np.array(angl)
        self.data['timestamp'] = _time.time()

    def plot_data(self):
        optics_mode = self.params.optics_mode

        ts_mod, ts_twi0 = pyts.create_accelerator(optics_mode=optics_mode)
        ts_mod = ts_mod + self._si_mod
        ts_mod.energy = self.params.energy
        tstwi, *_ = pyaccel.optics.calc_twiss(ts_mod, ts_twi0, indices='open')

        scrn_idx = pyaccel.lattice.find_indices(ts_mod, 'fam_name', 'Scrn')
        idx = pyaccel.lattice.find_indices(ts_mod, 'fam_name', 'InjNLKckr')
        idcs = np.r_[scrn_idx + idx]

        fig  = plt.figure(figsize=(18, 7))
        gs = mpl_gs.GridSpec(2, len(idcs), height_ratios=[2, 1])
        gs.update(left=0.07, right=0.98, top=0.92, bottom=0.1, hspace=0.25, wspace=0.05)
        ax1 = fig.add_subplot(gs[0, :])

        axs = [fig.add_subplot(gs[1, i], aspect='equal') for i in range(len(idcs))]

        ax1.plot(tstwi.spos[idcs], tstwi.betax[idcs], 'bo', linewidth=3)
        ax1.plot(tstwi.spos[idcs], tstwi.betay[idcs], 'ro', linewidth=3)
        ax1.plot(tstwi.spos[idcs], 100*tstwi.etax[idcs], 'go', linewidth=3)
        ax1.plot(tstwi.spos, tstwi.betax, 'b-', linewidth=3, label='Beta X')
        ax1.plot(tstwi.spos, tstwi.betay, 'r-', linewidth=3, label='Beta Y')
        ax1.plot(tstwi.spos, 100*tstwi.etax, 'g-', linewidth=3, label='Disp X')
        ax1.set_xlim([0, 35])
        ax1.legend()
        ax1.set_xlabel('spos [m]')
        ax1.set_ylabel('twiss')

        ax1.set_title(pyts.lattice_version + ' ' + optics_mode)

        emit0 = self.params.emit0
        coup = self.params.coupling
        sigmae = self.params.energyspread

        emity = emit0*coup/(1+coup)
        emitx = emit0/(1+coup)
        etax =  tstwi.etax[idcs]
        betax = tstwi.betax[idcs]
        betay = tstwi.betay[idcs]
        sx = 1e3*np.sqrt(betax * emitx + (etax * sigmae)**2)
        sy = 1e3*np.sqrt(betay * emity)

        sc_siz = self.params.plot_sigma_lim
        for i, idx in enumerate(idcs):
            ell = Ellipse(xy=(0.0, 0.0), width=2*sx[i], height=2*sy[i])
            axs[i].annotate(
                f'Model=({sx[i]:.2f}, {sy[i]:.2f})',
                xy=(-sc_siz*0.8, sc_siz*0.7), fontsize=12, color='C0')
            axs[i].add_patch(ell)
            axs[i].set_xlim([-sc_siz, sc_siz])
            axs[i].set_ylim([-sc_siz, sc_siz])
            axs[i].set_xlabel('X [mm]')
            if i:
                axs[i].set_yticklabels([])

        if not self.data:
            return fig
        sigx = self.data['sigmax']
        sigy = self.data['sigmay']
        angl = self.data['angle']
        sigg = self.params.sig_gauss_screen
        for i, idx in enumerate(scrn_idx):
            ell = Ellipse(
                xy=(0.0, 0.0), width=2*(sigx[i]-sigg), height=2*(sigy[i]-sigg),
                angle=angl[i] * 180/np.pi,
                facecolor='none', edgecolor='C1', linewidth=2)
            axs[i].annotate(
                f'meas=({sigx[i]:.2f}, {sigy[i]:.2f}) \n'
                f' angle={angl[i]*180/np.pi:.2f}º',
                xy=(-sc_siz*0.8, -sc_siz*0.8), fontsize=12, color='C1')
            axs[i].add_patch(ell)
            axs[i].set_xlim([-sc_siz, sc_siz])
            axs[i].set_ylim([-sc_siz, sc_siz])
        axs[0].set_ylabel('Y [mm]')
        return fig
