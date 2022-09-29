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
        self.posx_min = 0  # [um]
        self.posx_max = 0  # [um]
        self.nr_stepsx = 1
        self.posy_min = 0  # [um]
        self.posy_max = 0  # [um]
        self.nr_stepsy = 1
        self.nlk_kick = 0  # [urad]
        self.acq_nrsamples_pre = 0
        self.acq_nrsamples_post = 100
        self.buffer_sloworb = 50
        self.buffer_multiturn = 1
        self.wait_meas = 7  # [s]
        self.filename = ''
        self.save_data = True

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:}\n'.format
        stg = ftmp('posx_min', self.posx_min, '[um]')
        stg += ftmp('posx_max', self.posx_max, '[um]')
        stg += dtmp('nr_stepsx', self.nr_stepsx)
        stg += ftmp('posy_min', self.posy_min, '[um]')
        stg += ftmp('posy_max', self.posy_max, '[um]')
        stg += dtmp('nr_stepsy', self.nr_stepsy)
        stg += ftmp('nlk_kick', self.nlk_kick, '[urad]')
        stg += dtmp('acq_nrsamples_pre', self.acq_nrsamples_pre)
        stg += dtmp('acq_nrsamples_post', self.acq_nrsamples_post)
        stg += dtmp('buffer_sloworb', self.buffer_sloworb)
        stg += dtmp('buffer_multiturn', self.buffer_multiturn)
        stg += ftmp('wait_meas', self.wait_meas, '[s]')
        stg += stmp('filename', self.filename)
        stg += dtmp('save_data', self.save_data)
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
        if self.isonline:
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
            self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
            self._configdb = _ConfigDBClient(config_type='si_orbit')
            self.reforb = self._configdb.get_config_value('ref_orb')
            self.reforbx = _np.array(self.reforb['x'])
            self.reforby = _np.array(self.reforb['y'])

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
        sofb.wait_buffer()

    def config_sofb_sloworb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.nr_points = self.params.buffer_sloworb
        sofb.cmd_change_opmode_to_sloworb()
        sofb.cmd_reset()
        sofb.wait_buffer()

    def get_measurement_data(self):
        """."""
        bbbh, bbbv = self.devices['bbbh'], self.devices['bbbv']
        trajx, trajy = self.get_trajectory_sofb()
        data = {
            'timestamp': _time.time(),
            'stored_current': bbbh.dcct.current,
            'spech_mag': bbbh.sram.spec_mag,
            'spech_freq': bbbh.sram.spec_freq,
            'spech_mk1_mag': bbbh.sram.spec_marker1_mag,
            'spech_mk1_tune': bbbh.sram.spec_marker1_tune,
            'trajx': trajx,

            'specv_magv': bbbv.sram.spec_mag,
            'specv_freqv': bbbv.sram.spec_freq,
            'specv_mk1_mag': bbbv.sram.spec_marker1_mag,
            'specv_mk1_tune': bbbv.sram.spec_marker1_tune,
            'trajy': trajy,
            }
        return data

    def implement_bump(
            self, refx0=None, refy0=None,
            agx=0, agy=0, psx=0, psy=0, nr_iters=5, residue=1):
        """."""
        sofb = self.devices['sofb']
        refx0 = refx0 or self.reforbx
        refy0 = refy0 or self.reforby
        nrefx, nrefy = sofb.si_calculate_bumps(
                refx0, refy0, BumpNLK.DEFAULT_SS,
                agx=agx, agy=agy, psx=psx, psy=psy)
        sofb.refx, sofb.refy = nrefx, nrefy
        sofb.correct_orbit_manually(nr_iters=nr_iters, residue=residue)

    def _do_measure(self):
        prms = self.params
        if prms.save_data:
            self.save_data(fname=prms.filename, overwrite=False)
        posx_span = _np.linspace(prms.posx_min, prms.posx_max, prms.nr_stepsx)
        posy_span = _np.linspace(prms.posy_min, prms.posy_max, prms.nr_stepsy)

        # zig-zag type of scan in the y plane
        idy, idx = _np.meshgrid(range(prms.nr_stepsy), range(prms.nr_stepsx))
        idy[1::2] = _np.flip(idy[1::2])
        idx, idy = idx.ravel(), idy.ravel()

        self.config_sofb_sloworb()
        print('Setting initial bump...')
        self.implement_bump(psx=posx_span[idx[0]], psy=posy_span[idy[0]])

        data = list()
        for iter in range(idx.size):
            posx = posx_span[idx[iter]]
            posy = posy_span[idy[iter]]

            self.config_sofb_sloworb()
            self.implement_bump(psx=posx, psy=posy, nr_iters=3)
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
            if prms.save_data:
                self.save_data(fname=prms.filename, overwrite=True)
            if self._stopevt.is_set():
                print('Stopping...')
                break

        # return to initial reference orbit
        self.config_sofb_sloworb()
        print('Returning to ref_orb...')
        self.implement_bump(psx=0, psy=0, nr_iters=10)
        print('Finished!')

    def process_data(self):
        """."""
        posx, posy, magx, magy = [], [], [], []
        trajx, trajy = [], []
        for datai in self.data:
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

        nsteps = self.params.nr_stepsx
        anly = dict()
        anly['posx_bump'] = self._reshape_vec(posx, dim=nsteps)
        anly['posy_bump'] = self._reshape_vec(posy, dim=nsteps)
        anly['bbbh_mag'] = self._reshape_vec(magx, dim=nsteps)
        anly['bbbv_mag'] = self._reshape_vec(magy, dim=nsteps)
        anly['tbt_trajx_std'] = self._reshape_vec(trajx, dim=nsteps)
        anly['tbt_trajy_std'] = self._reshape_vec(trajy, dim=nsteps)
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
        _mplt.tight_layout(True)
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
        _mplt.tight_layout(True)
        return fig, ax1

    @staticmethod
    def _reshape_vec(vec, dim):
        new_vec = _np.array(vec).reshape((dim, -1))
        new_vec[1::2, :] = new_vec[1::2, ::-1]
        return new_vec
