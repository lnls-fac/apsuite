"""."""
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, BunchbyBunch
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class BumpNLKParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
        self.posx_min = -700  # [um]
        self.posx_max = 300  # [um]
        self.nr_stepsx = 11
        self.posy_min = -400  # [um]
        self.posy_max = 200  # [um]
        self.nr_stepsy = 7
        self.acq_nrsamples_pre = 50
        self.acq_nrsamples_post = 450
        self.buffer_sloworb = 20
        self.buffer_multiturn = 20
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
        stg += dtmp('nr_stepsx', self.nr_stepsx)
        stg += ftmp('posy_min', self.posy_min, '[um]')
        stg += ftmp('posy_max', self.posy_max, '[um]')
        stg += dtmp('nr_stepsy', self.nr_stepsy)
        stg += dtmp('acq_nrsamples_pre', self.acq_nrsamples_pre)
        stg += dtmp('acq_nrsamples_post', self.acq_nrsamples_post)
        stg += dtmp('buffer_sloworb', self.buffer_sloworb)
        stg += dtmp('buffer_multiturn', self.buffer_multiturn)
        stg += ftmp('wait_meas', self.wait_meas, '[s]')
        stg += dtmp('orbcorr_nr_iters', self.orbcorr_nr_iters)
        stg += ftmp('orbcorr_residue', self.orbcorr_residue, '[um]')
        stg += stmp('orbcorr_bpm_enbl', '\n'+str(self.orbcorr_bpm_enbl), '')
        return stg


class BumpNLK(_BaseClass):
    """."""

    DEFAULT_SS = '01SA'

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, params=BumpNLKParams(),
            target=self._do_measure, isonline=isonline)
        self.data = dict()
        self.reforbx = None
        self.reforby = None
        if self.isonline:
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
            self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
            self._configdb = _ConfigDBClient(config_type='si_orbit')
            self.update_reforb()

    def get_trajectory_sofb(self):
        """."""
        sofb = self.devices['sofb']
        trajx = sofb.mt_trajx.reshape((-1, sofb.data.nr_bpms))
        trajy = sofb.mt_trajy.reshape((-1, sofb.data.nr_bpms))
        return trajx, trajy

    def config_sofb_multiturn(self):
        """."""
        sofb = self.devices['sofb']
        sofb.nr_points = self.params.buffer_multiturn
        sofb.trigsamplepre = self.params.acq_nrsamples_pre
        sofb.trigsamplepost = self.params.acq_nrsamples_post
        sofb.cmd_change_opmode_to_multiturn()
        sofb.cmd_reset()
        # NOTE: the factor of 8 is because the current version of SOFB is too
        # slow to update multi-turn orbits.
        sofb.wait_buffer(timeout=sofb.nr_points*0.5*8)

    def config_sofb_sloworb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.nr_points = self.params.buffer_sloworb
        sofb.cmd_change_opmode_to_sloworb()
        sofb.cmd_reset()
        # NOTE: the factor of 8 is because the current version of SOFB is too
        # slow to update multi-turn orbits.
        sofb.wait_buffer(timeout=sofb.nr_points*0.5*8)

    def get_measurement_data(self):
        """."""
        bbbh, bbbv = self.devices['bbbh'], self.devices['bbbv']
        trajx, trajy = self.get_trajectory_sofb()
        data = {
            'timestamp': _time.time(),
            'stored_current': bbbh.dcct.current,
            'spech_mag': bbbh.sram.spec_mag,
            'spech_freq': bbbh.sram.spec_freq,
            'bbbh_data_mean': bbbh.sram.data_mean,
            'spech_mk1_mag': bbbh.sram.spec_marker1_mag,
            'spech_mk1_tune': bbbh.sram.spec_marker1_tune,
            'trajx': trajx,

            'specv_magv': bbbv.sram.spec_mag,
            'specv_freqv': bbbv.sram.spec_freq,
            'bbbv_data_mean': bbbv.sram.data_mean,
            'specv_mk1_mag': bbbv.sram.spec_marker1_mag,
            'specv_mk1_tune': bbbv.sram.spec_marker1_tune,
            'trajy': trajy,
            }
        return data

    def implement_bump(
            self, refx0=None, refy0=None, agx=0, agy=0, psx=0, psy=0,
            sec=None):
        """."""
        sofb = self.devices['sofb']
        refx0 = refx0 or self.reforbx
        refy0 = refy0 or self.reforby
        nr_iters = self.params.orbcorr_nr_iters
        residue = self.params.orbcorr_residue
        sec = sec if sec is None else BumpNLK.DEFAULT_SS

        nrefx, nrefy = sofb.si_calculate_bumps(
            refx0, refy0, sec, agx=agx, agy=agy, psx=psx, psy=psy)

        sofb.refx = nrefx
        sofb.refy = nrefy
        enblx0 = sofb.bpmxenbl
        enbly0 = sofb.bpmyenbl
        sofb.bpmxenbl = self.params.orbcorr_bpm_enbl
        sofb.bpmyenbl = self.params.orbcorr_bpm_enbl
        _time.sleep(0.5)  # NOTE: For some reason We have to wait here.
        sofb.correct_orbit_manually(nr_iters=nr_iters, residue=residue)
        sofb.bpmxenbl = enblx0
        sofb.bpmyenbl = enbly0

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

    def _do_measure(self):
        print(
            'NOTE:\n' +
            'Remember to turn off the septa and their orbit feedforward.\n')
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
            residue=self.params.orbcorr_residue)
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
            _mx, _my = 10**(_mx/20), 10**(_my/20)
            mag = _np.log(_mx**2 + _my**2)
        elif plane.upper() == 'H':
            mag = 10**(anly['bbbh_mag']/20)
        elif plane.upper() == 'V':
            mag = 10**(anly['bbbv_mag']/20)
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
