"""."""

import time as _time
import numpy as _np
from siriuspy.devices import SOFB, CurrInfoSI
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
from apsuite.orbcorr.si_bumps import SiCalcBumps

from ..utils import (
    ThreadedMeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)


class BumpParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
        self.posx_min = -100  # [um]
        self.posx_max = 100  # [um]
        self.nr_steps_x = 11
        self.posy_min = -100  # [um]
        self.posy_max = 100  # [um]
        self.nr_steps_y = 7

        self.angx_min = -100  # [um]
        self.angx_max = 100  # [um]
        self.nr_steps_angx = 11
        self.angy_min = -100  # [um]
        self.angy_max = 100  # [um]
        self.nr_steps_angy = 7

        self.n_bpms_out = 3
        self.minsingval = 0.2
        self.buffer_sloworb = 20
        self.bump_residue = 5
        self.bump_max_residue = 10

        self.wait_meas = 2  # [s]
        self.orbcorr_nr_iters = 5
        self.orbcorr_residue = 5  # [um]
        enbl_bump = _np.ones(160, dtype=int)
        enbl_bump[1:7] = 0
        enbl_bump[-7:-1] = 0
        self.orbcorr_bpm_enbl = enbl_bump

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('posx_min', self.posx_min, '[um]')
        stg += ftmp('posx_max', self.posx_max, '[um]')
        stg += dtmp('nr_stepsx', self.nr_steps_x)
        stg += ftmp('posy_min', self.posy_min, '[um]')
        stg += ftmp('posy_max', self.posy_max, '[um]')
        stg += dtmp('nr_stepsy', self.nr_steps_y)
        stg += dtmp('buffer_sloworb', self.buffer_sloworb)

        stg += ftmp('wait_meas', self.wait_meas, '[s]')
        stg += dtmp('orbcorr_nr_iters', self.orbcorr_nr_iters)
        stg += ftmp('orbcorr_residue', self.orbcorr_residue, '[um]')
        stg += stmp('orbcorr_bpm_enbl', '\n' + str(self.orbcorr_bpm_enbl), '')
        return stg


class Bump(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, params=BumpParams(), target=self._do_scan, isonline=isonline
        )
        self.data = dict()
        self.reforbx = None
        self.reforby = None
        self.bumptools = SiCalcBumps()
        if self.isonline:
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['currinfo'] = CurrInfoSI()
            self._configdb = _ConfigDBClient(config_type='si_orbit')
            self.update_reforb()
            self.configure_sofb()

    def config_sofb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.nr_points = self.params.buffer_sloworb
        sofb.cmd_change_opmode_to_sloworb()
        sofb.cmd_reset()
        # NOTE: the factor of 8 is because the current version of SOFB is too
        # slow to update multi-turn orbits.
        sofb.wait_buffer(timeout=sofb.nr_points * 0.5 * 8)
        sofb.cmd_turn_off_autocorr()
        self._bpmxenbl = self.devices['sofb'].bpmxenbl
        self._bpmyenbl = self.devices['sofb'].bpmyenbl

    def get_measurement_data(self):
        """."""
        bbbh, bbbv = self.devices['bbbh'], self.devices['bbbv']
        trajx, trajy = self.get_trajectory_sofb()
        data = {
            'timestamp': _time.time(),
            'stored_current': bbbh.dcct.current,
            'rf_frequency': bbbh.rfcav.dev_rfgen.frequency,
            'spech_mag': bbbh.sram.spec_mag,
            'spech_freq': bbbh.sram.spec_freq,
            'bbbh_data_mean': bbbh.sram.data_mean,
            'spech_mk1_mag': bbbh.sram.spec_marker1_mag,
            'spech_mk1_tune': bbbh.sram.spec_marker1_tune,
            'trajx': trajx,
            'specv_mag': bbbv.sram.spec_mag,
            'specv_freq': bbbv.sram.spec_freq,
            'bbbv_data_mean': bbbv.sram.data_mean,
            'specv_mk1_mag': bbbv.sram.spec_marker1_mag,
            'specv_mk1_tune': bbbv.sram.spec_marker1_tune,
            'trajy': trajy,
        }
        return data

    def restore_sofb_reforb(self):
        sofb = self.devices['sofb']
        clt = self._configdb
        ref_orb = clt.get_config_value('ref_orb')
        refx = _np.array(ref_orb['x'])
        refy = _np.array(ref_orb['y'])
        sofb.refx = refx
        sofb.refy = refy
        sofb.bpmxenbl = _np.ones(refx.size, dtype=bool)
        sofb.bpmyenbl = _np.ones(refx.size, dtype=bool)

    @staticmethod
    def subsec_2_sectype_nr(subsec):
        section_nr = int(subsec[:2])
        if not 1 <= section_nr <= 20:
            raise ValueError('Section must be between 01..20.')
        section_type = subsec[2:]
        return section_type, section_nr

    def remove_sofb_bpms(self, section_type, section_nr, n_bpms_out):
        sofb = self.devices['sofb']
        idcs_out = self.bumptools.get_closest_bpms_indices(
            section_type=section_type,
            sidx=section_nr - 1,
            n_bpms_out=n_bpms_out,
        )
        sofb.bpmxenbl[idcs_out[: n_bpms_out * 2]] = False
        sofb.bpmyenbl[idcs_out[n_bpms_out * 2 :]] = False
        _time.sleep(0.5)  # NOTE: For some reason We have to wait here.

    def get_orbrms(self, refx, refy, idx):
        dorbx = self.devices['sofb'].orbx - refx
        dorby = self.devices['sofb'].orby - refy
        dorbx = dorbx[idx]
        dorby = dorby[idx]
        return _np.hstack([dorbx, dorby]).std()

    def implement_bump(
        self, refx=None, refy=None, agx=0, agy=0, psx=0, psy=0, subsec=None
    ):
        """."""
        sofb = self.devices['sofb']
        refx = refx or self.reforbx
        refy = refy or self.reforby
        n_bpms_out = self.params.n_bpms_out
        minsingval = self.params.minsingval
        nr_iters = self.params.orbcorr_nr_iters
        residue = self.params.orbcorr_residue
        bump_residue = self.params.bump_residue
        bump_max_residue = self.bump_max_residue

        orbx, orby = sofb.si_calculate_bumps(
            refx,
            refy,
            subsec,
            agx=agx,
            agy=agy,
            psx=psx,
            psy=psy,
            n_bpms_out=n_bpms_out,
            minsingval=minsingval,
        )
        section_type, section_nr = self.subsec_2_sectype_nr(subsec)

        self.remove_sofb_bpms(section_type, section_nr)

        idcs_bpm = self.bumptools.get_bpm_indices(
            section_type=section_type, sidx=section_nr - 1
        )

        # Set orbit
        sofb.refx = orbx
        sofb.refy = orby

        # Verify orbit correction
        rms_residue = bump_residue + 1
        print('Waiting orbit...')
        while rms_residue > bump_residue:
            _ = sofb.correct_orbit_manually(nr_iters=nr_iters, residue=residue)
            rms_residue = self.get_orbrms(orbx, orby, idcs_bpm)
            print(f'    rms_residue = {rms_residue:.3f} um')
            bump_residue *= 1.2
            if bump_residue > bump_max_residue:
                raise ValueError('Could not correct orbit.')

        print('Done!')

    def update_reforb(self):
        """."""
        self.reforb = self._configdb.get_config_value('ref_orb')
        self.reforbx = _np.array(self.reforb['x'])
        self.reforby = _np.array(self.reforb['y'])

    def calc_bump_span(self):
        """."""
        prms = self.params
        posx_span = _np.linspace(prms.posx_min, prms.posx_max, prms.nr_stepsx)
        posy_span = _np.linspace(prms.posy_min, prms.posy_max, prms.nr_stepsy)
        return posx_span, posy_span

    def _do_scan(self):
        print(
            'NOTE:\n'
            + 'Remember to turn off the septa and their orbit feedforward.\n'
        )
        prms = self.params
        posx_span, posy_span = self.calc_bump_span()

        # zig-zag type of scan in the y plane
        idy, idx = _np.meshgrid(range(prms.nr_stepsy), range(prms.nr_stepsx))
        idy[1::2] = _np.flip(idy[1::2])
        idx, idy = idx.ravel(), idy.ravel()

        data = list()
        for iter in range(idx.size):
            posx = posx_span[idx[iter]]
            posy = posy_span[idy[iter]]

            self.config_sofb_sloworb()
            self.implement_bump(psx=posx, psy=posy)
            self.config_sofb_multiturn()
            _time.sleep(prms.wait_meas)
            data.append(self.get_measurement_data())
            ampx = data[-1]['spech_mk1_mag']
            ampy = data[-1]['specv_mk1_mag']
            fstr = f'(x, y) = ({posx:6.1f}, {posy:6.1f}) um --> '
            fstr += f'(ampx, ampy) = ({ampx:.1f}, {ampy:.1f}) dB'
            print(fstr)
            data[-1]['bump'] = (posx, posy)
            self.data = data
            if self._stopevt.is_set():
                print('Stopping...')
                break

        # return to initial reference orbit
        self.config_sofb_sloworb()
        print('Returning to ref_orb...')
        self.devices['sofb'].refx = self.reforbx
        self.devices['sofb'].refy = self.reforby
        self.devices['sofb'].correct_orbit_manually(
            nr_iters=self.params.orbcorr_nr_iters,
            residue=self.params.orbcorr_residue,
        )
        print('Finished!')

    def process_data(self):
        """."""
        posx, posy, magx, magy = [], [], [], []
        trajx, trajy = [], []
        data_index = []
        for idx, datai in enumerate(self.data):
            trjx = datai['trajx'].copy()
            trjx -= trjx.mean(axis=0)[None, :]
            trjy = datai['trajy'].copy()
            trjy -= trjy.mean(axis=0)[None, :]
            trajx.append(trjx.std())
            trajy.append(trjy.std())
            posx.append(datai['bump'][0])
            posy.append(datai['bump'][1])
            magx.append(datai['spech_mk1_mag'])
            magy.append(datai['specv_mk1_mag'])
            data_index.append(idx)

        nsteps = self.params.nr_stepsx
        anly = dict()
        anly['posx_bump'] = self._reshape_vec(posx, axis0_size=nsteps)
        anly['posy_bump'] = self._reshape_vec(posy, axis0_size=nsteps)
        anly['bbbh_mag'] = self._reshape_vec(magx, axis0_size=nsteps)
        anly['bbbv_mag'] = self._reshape_vec(magy, axis0_size=nsteps)
        anly['tbt_trajx_std'] = self._reshape_vec(trajx, axis0_size=nsteps)
        anly['tbt_trajy_std'] = self._reshape_vec(trajy, axis0_size=nsteps)
        anly['data_index'] = self._reshape_vec(data_index, axis0_size=nsteps)
        self.analysis = anly

    def plot_bbb_mag(self, plane='HV', shading='auto', cmap='jet'):
        """."""
        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        anly = self.analysis
        posx, posy = anly['posx_bump'], anly['posy_bump']

        if plane.upper() == 'HV':
            _mx, _my = anly['bbbh_mag'], anly['bbbv_mag']
            _mx, _my = 10 ** (_mx / 20), 10 ** (_my / 20)
            mag = _np.log(_mx**2 + _my**2)
        elif plane.upper() == 'H':
            mag = 10 ** (anly['bbbh_mag'] / 20)
        elif plane.upper() == 'V':
            mag = 10 ** (anly['bbbv_mag'] / 20)
        else:
            raise Exception('plane argument must be HV, H or V.')

        pcm = ax1.pcolormesh(posx, posy, mag, shading=shading, cmap=cmap)
        ax1.set_xlabel(r'Bump x [$\mu$m]')
        ax1.set_ylabel(r'Bump y [$\mu$m]')
        cbar = fig.colorbar(pcm, ax=ax1)
        cbar.set_label(plane + ' BbB Mag')
        ax1.set_title('')
        fig.tight_layout()
        return fig, ax1

    def plot_tbt_traj(self, plane='HV', shading='auto', cmap='jet'):
        """."""
        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        anly = self.analysis
        posx, posy = anly['posx_bump'], anly['posy_bump']

        if plane.upper() == 'HV':
            _tx, _ty = anly['tbt_trajx_std'], anly['tbt_trajy_std']
            traj = _np.sqrt(_tx**2 + _ty**2)
        elif plane.upper() == 'H':
            traj = anly['tbt_trajx_std']
        elif plane.upper() == 'V':
            traj = anly['tbt_trajy_std']
        else:
            raise Exception('plane argument must be HV, H or V.')

        pcm = ax1.pcolormesh(posx, posy, traj, shading=shading, cmap=cmap)
        ax1.set_xlabel(r'Bump x [$\mu$m]')
        ax1.set_ylabel(r'Bump y [$\mu$m]')
        cbar = fig.colorbar(pcm, ax=ax1)
        cbar.set_label(plane + r'. TbT std distortion [$\mu$m]')
        ax1.set_title('')
        fig.tight_layout()
        return fig, ax1

    @staticmethod
    def _reshape_vec(vec, axis0_size):
        new_vec = _np.array(vec).reshape((axis0_size, -1))
        new_vec[1::2, :] = new_vec[1::2, ::-1]
        return new_vec
