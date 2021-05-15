#!/usr/bin/env python-sirius

import time

import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs
import matplotlib.cm as mcmap
from matplotlib import rcParams

import pyaccel
from pymodels import si
from apsuite.utils import DataBaseClass, ParamsBaseClass
from apsuite.orbcorr import OrbitCorr
from siriuspy.sofb.utils import si_calculate_bump


rcParams.update({
    'lines.linewidth': 2, 'font.size': 14, 'axes.grid': True,
    'grid.alpha': 0.5, 'grid.linestyle': '--'})


class Params(ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nparticles = 1000
        self.nturns = 40
        self.measurement_type = 'dynap'
        self.obstruction_section = '13M1'
        self.obstruction_size = 3e-3
        self.obstruction_side = 'negative'
        self.obstruction_index_offset = -1  # offset in relation to the BPM
        self.injection_position = 7.5e-3
        self.dynap_kick = 0.5e-3
        self.delta_energy = -0.0
        self.watch_point = 'obstruction'

    def suggest_name(self):
        """."""
        size = self.obstruction_size * 1e3
        kick = self.dynap_kick * 1e6
        pos = self.injection_position * 1e3
        strg = ''
        strg += f'type_{self.measurement_type:s}-'
        if self.measurement_type == 'dynap':
            strg1 = f'kick_{kick:.0f}urad'
        else:
            strg1 = f'injpos_{pos:.2f}mm'
        strg1 = strg1.replace('-', 'm').replace('.', 'p')
        strg += f'{strg1:s}-'
        strg += f'section_{self.obstruction_section:s}-'
        strg += f'offset_{self.obstruction_index_offset:d}'.replace('-', 'm')
        strg += f'-side_{self.obstruction_side:s}-'
        strg += f'size_{size:.2f}mm'.replace('-', 'm').replace('.', 'p')
        strg += f'-watch_{self.watch_point:s}'
        return strg


class ObstructionStudy(DataBaseClass):
    """."""

    def __init__(self, model):
        """."""
        super().__init__(params=Params())
        self.model = model
        self.model.cavity_on = True
        self.model.radiation_on = True
        self.model.vchamber_on = True
        self.famdata = si.get_family_data(model)
        self.section_mapping = si.families.get_section_name_mapping(model)
        self.twiss, _ = pyaccel.optics.calc_twiss(model)

    def do_study(self):
        """."""
        mod = self.model
        npart = self.params.nparticles
        nturns = self.params.nturns
        typem = self.params.measurement_type
        kick = self.params.dynap_kick
        pos = self.params.injection_position
        dener = self.params.delta_energy
        phasespace = self.params.watch_point
        section = self.params.obstruction_section

        famdata = self.famdata
        twiss = self.twiss

        if typem == 'dynap':
            init_idx = pyaccel.lattice.find_indices(
                mod, 'fam_name', 'InjDpKckr')[0]
            emit0 = 0.25e-9
            coup = 0.03/100
            emitx = emit0/(1 + coup)
            emity = emitx * coup
            sigmae = 0.85e-3
            sigmas = 3.5e-3
        else:
            init_idx = pyaccel.lattice.find_indices(
                mod, 'fam_name', 'InjNLKckr')[0]
            emit0 = 7.5e-9
            coup = 1
            emitx = emit0/(1 + coup)
            emity = emitx * coup
            sigmae = 0.87e-3
            sigmas = 11e-3

        init_idx += 1
        twi = twiss[init_idx]
        bpm_idx = famdata['BPM']['devnames'].index(f'SI-{section:s}:DI-BPM')
        bpm_idcs = np.array(famdata['BPM']['index']).ravel()
        obs_idx = bpm_idcs[bpm_idx] - self.params.obstruction_index_offset
        self.data['obstruction_index'] = obs_idx

        hmin0, hmax0, hmin, hmax = self._get_chamber(obs_idx)
        mod[obs_idx].hmin = hmin
        mod[obs_idx].hmax = hmax

        bun = pyaccel.tracking.generate_bunch(
            emitx, emity, sigmae, sigmas, twi, npart)

        init = pyaccel.tracking.find_orbit6(mod, )
        if typem == 'dynap':
            kickerlen = 0.5
            pos = kick * kickerlen**2
            init += np.array([pos, kick, 0, 0, dener, 0])[:, None]
        else:
            init += np.array([pos, 0, 0, 0, dener, 0])[:, None]
        bun += init
        self.data['rin'] = bun

        tini = time.time()
        if phasespace == 'obstruction':
            bun, *_ = pyaccel.tracking.line_pass(
                mod[init_idx:obs_idx+1], bun, )
            init_idx = obs_idx + 1

        part_out, _, lost_turn, lost_element, _ = pyaccel.tracking.ring_pass(
            mod, bun, nr_turns=nturns, turn_by_turn=True,
            element_offset=init_idx, parallel=True)

        print(f'Tracking took = {time.time()-tini:.2f} seconds')
        mod[obs_idx].hmin = hmin0
        mod[obs_idx].hmax = hmax0

        lost_element = np.array(lost_element)
        lost_turn = np.array(lost_turn)

        self.data['rout'] = part_out
        self.data['lost_turn'] = lost_turn
        self.data['lost_element'] = lost_element

        idcs = lost_turn < nturns
        lost_element = lost_element[idcs]
        lost_turn = lost_turn[idcs]
        idcs = np.argsort(lost_turn)
        lost_turn = lost_turn[idcs]
        lost_element = lost_element[idcs]

        possum = npart*bpm_idcs.size*np.ones(nturns)
        for i, ele in zip(lost_turn, lost_element):
            if i >= nturns:
                continue
            possum[i] -= np.sum(bpm_idcs <= ele)
            possum[i+1:] -= bpm_idcs.size

        possum *= 100 / (npart * bpm_idcs.size)
        self.data['bpms_sum'] = possum

    def plot_results(self):
        """."""
        phasespace = self.params.watch_point
        possum = self.data['bpms_sum']
        rout = self.data['rout']
        posx = calc_inverse_poly(rout[0])
        posx = np.nanmean(posx, axis=0)
        posx = 1e3*calc_poly(posx)

        obidx = self.data['obstruction_index']

        fig = mplt.figure(figsize=(10, 10))
        gs_ = mgs.GridSpec(
            3, 2,
            left=0.12, right=0.98, top=0.9, bottom=0.08,
            hspace=0.4, wspace=0.1, height_ratios=[1, 1, 2])
        axis_lturn = fig.add_subplot(gs_[0, :])
        axis_posx = fig.add_subplot(gs_[1, :])
        axis_phspc = fig.add_subplot(gs_[2, 0])
        axis_mean = fig.add_subplot(
            gs_[2, 1], sharex=axis_phspc, sharey=axis_phspc)

        axis_lturn.plot(possum, 'ok')
        axis_posx.plot(posx, 'o')

        nturns = possum.size
        cm_ = mcmap.jet(np.linspace(0, 1, nturns + 1))
        minx = 0
        for turn in range(nturns+1):
            trn = nturns - turn
            if not turn or possum[trn] < 0.001:
                continue
            x = 1e3*rout[0, :, trn]
            xl = 1e3*rout[1, :, trn]
            axis_phspc.plot(x, xl, '.', color=cm_[trn])
            x = np.nanmean(x)
            xl = np.nanmean(xl)
            minx = min(minx, x)
            axis_mean.plot(x, xl, 'o', color=cm_[trn])

        if phasespace == 'obstruction':
            _, _, hmin, hmax = self._get_chamber(obidx)
            axis_mean.axvline(hmin*1e3, linestyle='--', color='k')
            axis_phspc.axvline(hmin*1e3, linestyle='--', color='k')
            axis_mean.axvline(hmax*1e3, linestyle='--', color='k')
            axis_phspc.axvline(hmax*1e3, linestyle='--', color='k')

        axis_lturn.set_title(self._get_title())
        axis_lturn.text(
            0.95, 0.95, f'Loss = {100-possum[-1]:.1f}%',
            horizontalalignment='right', verticalalignment='top',
            transform=axis_lturn.transAxes)

        if phasespace != 'obstruction':
            axis_mean.text(
                0.02, 0.98, r'$x_{\mathrm{min}}$ = '+f'{minx:.2f}mm',
                horizontalalignment='left', verticalalignment='top',
                transform=axis_mean.transAxes)
        axis_phspc.set_title('Phase Space @ ' + phasespace.title())
        axis_mean.set_title('Beam Centroid @ ' + phasespace.title())
        axis_lturn.set_xlabel('Number of Turns')
        axis_lturn.set_ylabel('BPMs Sum Signal [%]')
        axis_posx.set_ylabel('Pos X [mm]')
        axis_posx.set_xlabel('Number of Turns')
        axis_mean.set_xlabel('x [mm]')
        axis_phspc.set_xlabel('x [mm]')
        axis_phspc.set_ylabel('xl [mrad]')
        mplt.setp(axis_mean.get_yticklabels(), visible=False)
        return fig

    def plot_lost_element_histogram(self):
        """."""
        lelem = self.data['lost_element']
        lturn = self.data['lost_turn']
        obidx = self.data['obstruction_index']
        lelem = np.array(lelem)
        lturn = np.array(lturn)
        idcs = (lturn < self.params.nturns)
        lelem = lelem[idcs]
        lpos = self.twiss.spos[lelem]
        obpos = self.twiss.spos[obidx]

        fig = mplt.figure(figsize=(10, 4))
        gs_ = mgs.GridSpec(
            1, 1,
            left=0.12, right=0.98, top=0.85, bottom=0.14,
            hspace=0.4, wspace=0.1)
        axis_lelem = fig.add_subplot(gs_[0, 0])

        axis_lelem.hist(lpos, bins=int(self.model.length / 3))
        axis_lelem.axvline(obpos, linestyle='--', color='k')

        axis_lelem.set_title(self._get_title())
        axis_lelem.set_xlabel('Position [m]')
        axis_lelem.set_ylabel('Number of Losses')
        return fig

    def _get_chamber(self, obs_idx):
        obst = self.params.obstruction_size
        side = self.params.obstruction_side

        hmin0 = self.model[obs_idx].hmin
        hmax0 = self.model[obs_idx].hmax
        if side == 'negative':
            hmin = self.model[obs_idx].hmin + obst
            hmax = hmax0
        elif side == 'positive':
            hmin = hmin0
            hmax = self.model[obs_idx].hmax - obst
        elif side in {'both', 'symmetric'}:
            hmin = self.model[obs_idx].hmin + obst
            hmax = self.model[obs_idx].hmax - obst
        return hmin0, hmax0, hmin, hmax

    def _get_title(self):
        secmap = self.section_mapping
        typem = self.params.measurement_type
        obst = self.params.obstruction_size
        kick = self.params.dynap_kick
        pos = self.params.injection_position
        obidx = self.data['obstruction_index']

        tit = f'injpos = {pos*1e3:.3f}mm'
        if typem == 'dynap':
            tit = f'kick={kick*1e3:.3f}mrad'
        title = ''
        title += f'{typem.title():s} Study with Obstruction at '
        title += f'{secmap[obidx]:s}-{self.model[obidx].fam_name:s}\n'
        title += f'Obst. Size = {obst*1e3:.1f} mm,   {tit:s}'
        return title


def _calc_poly(p):
    p2 = p*p
    p3 = p2*p
    p5 = p3*p2
    p7 = p5*p2
    p9 = p7*p2

    return (
        p*8.57433e+06 + p3*4.01544e+06 + p5*3.94658e+06 + p7*-1.1398e+06 +
        p9*2.43619e+07)


def calc_poly(x):
    return _calc_poly(x/8.57433e-3) * 1e-9


_x = np.linspace(-5e-3, 5e-3, 1000)
invpol = np.polynomial.polynomial.polyfit(
    calc_poly(_x), _x, deg=[1, 3, 5, 7, 9])
print(invpol)


def calc_inverse_poly(x):
    return np.polynomial.polynomial.polyval(x, invpol)


def main():
    """."""
    model = si.create_accelerator()

    # # increase vacuum chamber size
    # for i in range(len(model)):
    #     model[i].hmin *= 2
    #     model[i].hmax *= 2

    # # make bumps
    # subsec = '13SA'
    # nrbpm_ignore = 3
    # subidx = int(subsec[:2]) - 1
    # bpm1 = 8*subidx - 1
    # bpm2 = 8*subidx

    # orbcorr = OrbitCorr(model, 'SI')
    # orbcorr.params.enbllistbpm[bpm1-nrbpm_ignore:bpm1] = False
    # orbcorr.params.enbllistbpm[bpm2+1:bpm2+1+nrbpm_ignore] = False
    # orbcorr.params.tolerance = 1e-5

    # orb = orbcorr.get_orbit()
    # orbx, orby = np.split(orb, 2)
    # orbx, orby = si_calculate_bump(orbx, orby, subsec, psx=-1e-3)
    # gorb = np.r_[orbx, orby]
    # orbcorr.correct_orbit(goal_orbit=gorb)

    # start obstruction study
    obs = ObstructionStudy(model)

    obs.model.radiation_on = True
    obs.params.nparticles = 200
    obs.params.nturns = 2000
    obs.params.measurement_type = 'dynap'
    obs.params.obstruction_size = 0e-3
    obs.params.obstruction_side = 'negative'
    obs.params.dynap_kick = -0.050e-3/85*50
    obs.params.injection_position = -8e-3
    obs.params.delta_energy = -0.0/100
    obs.params.watch_point = 'Kicker'
    obs.params.obstruction_section = '13M1'
    obs.params.obstruction_index_offset = -1

    # obs.do_study()
    # obs.save_data(
    #     obs.params.suggest_name(), overwrite=False)

    # name = (
    #     'type_dynap-kick_m250urad-section_13M1-offset_m1-' +
    #     'side_negative-size_0p00mm-watch_Kicker')
    # obs.load_and_apply(name)
    obs.load_and_apply(obs.params.suggest_name())
    # fig = obs.plot_lost_element_histogram()
    fig = obs.plot_results()
    mplt.show()


if __name__ == '__main__':
    main()
