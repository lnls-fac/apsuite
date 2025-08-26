"""."""

import time as _time
import numpy as _np
from siriuspy.devices import SOFB, HLFOFB, CurrInfoSI
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
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
        self.pts_psx = 0  # 1d numpy array [um]
        self.pts_psy = 0  # 1d numpy array [um]
        self.pts_agx = 0  # 1d numpy array [urad]
        self.pts_agy = 0  # 1d numpy array [urad]
        self.subsec = '01C1'
        self.do_angular_bumps = True

        self.n_bpms_out = 3
        self.minsingval = 0.2
        self.bump_residue = 5  # [um]
        self.bump_max_residue = 10  # [um]
        self.fofb_max_kick = 4  # [urad]

        self.buffer_sloworb = 20
        self.wait_meas = 2  # [s]
        self.orbcorr_nr_iters = 5
        self.orbcorr_residue = 5  # [um]

        self.use_fofb = False
        self.timeout_fofb_ramp = 15

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        # stmp = '{0:20s} = {1:9.4f}  {1:9.4f} {2:s}\n'.format
        stg = 'subsec = {} \n'.format(self.subsec)
        stg += dtmp('do_angular_bumps', self.do_angular_bumps)

        stg += dtmp('n_bpms_out', self.n_bpms_out, '')
        stg += ftmp('minsingval', self.minsingval, '')
        stg += ftmp('bump_residue', self.bump_residue, '[um]')
        stg += ftmp('bump_max_residue', self.bump_max_residue, '[um]')
        stg += ftmp('fofb_max_kick', self.fofb_max_kick, '[urad]')

        stg += dtmp('buffer_sloworb', self.buffer_sloworb, '')
        stg += ftmp('wait_meas', self.wait_meas, '[s]')
        stg += dtmp('orbcorr_nr_iters', self.orbcorr_nr_iters)
        stg += ftmp('orbcorr_residue', self.orbcorr_residue, '[um]')

        # stg += stmp('pts_psx', self.pts_psx[0], self.pts_psx[-1], '[um]')
        # stg += stmp('pts_psy', self.pts_psy[0], self.pts_psy[-1], '[um]')
        # stg += stmp(
        #     'pts_agx', self.pts_agx[0], self.pts_agx[-1], '[um] or [urad]'
        # )
        # stg += stmp(
        #     'pts_agy', self.pts_agy[0], self.pts_agy[-1], '[um] or [urad]'
        # )
        return stg


class Bump(_BaseClass):
    """."""

    def __init__(self, params, isonline=True):
        """."""
        _BaseClass.__init__(
            self,
            params=params if params is not None else BumpParams(),
            target=self._do_scan,
            isonline=isonline,
        )
        self.data = dict()
        self.reforbx = None
        self.reforby = None
        self.bumptools = SiCalcBumps()
        if self.isonline:
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['fofb'] = HLFOFB(HLFOFB.DEVICES.SI)
            self.devices['currinfo'] = CurrInfoSI()
            self._configdb = _ConfigDBClient(config_type='si_orbit')
            fofb = self.devices['fofb']
            self._orig_fofb_bpmxenbl = fofb.bpmxenbl
            self._orig_fofb_bpmyenbl = fofb.bpmyenbl

    @staticmethod
    def do_measurement():
        print('Not a measurement!')

    def _is_beam_alive(self):
        if self.devices['currinfo'].storedbeam:
            return True
        print('Beam is dead!')
        self.restore_sofb_reforb()

    @staticmethod
    def subsec_2_sectype_nr(subsec):
        """Convert subsec string to subsec type and number.

        Args:
            subsec (str): Subsection name. example "02C1"

        Raises:
            ValueError: Must be a valid section

        Returns:
            str, int: section type and section number
        """
        section_nr = int(subsec[:2])
        if not 1 <= section_nr <= 20:
            raise ValueError('Section must be between 01..20.')
        section_type = subsec[2:]
        return section_type, section_nr

    def update_reforb(self, refx, refy):
        """Update reforb.

        Args:
            refx (1d numpy array): Horizontal orbit
            refy (1d numpy array): Vertical orbit
        """
        self.reforbx = refx
        self.reforby = refy

    def get_measurement_data(self):
        """."""
        currinfo = self.devices['currinfo']
        data = {'timestamp': _time.time(), 'current': currinfo.current}
        data['measurements:'] = self.do_measurement()
        return data

    def config_sofb(self):
        """."""
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        if sofb.autocorrsts or fofb.loop_state:
            fofb.cmd_turn_off_loop_state(timeout=self.params.timeout_fofb_ramp)
            sofb.cmd_turn_off_autocorr()
        sofb.nr_points = self.params.buffer_sloworb
        sofb.cmd_change_opmode_to_sloworb()
        sofb.cmd_reset()
        # NOTE: the factor of 8 is because the current version of SOFB is too
        # slow to update multi-turn orbits.
        sofb.wait_buffer(timeout=sofb.nr_points * 0.5 * 8)
        if self.params.use_fofb:
            sofb.cmd_turn_on_autocorr()
            fofb.cmd_turn_on_loop_state()
        else:
            sofb.cmd_turn_off_autocorr()
            fofb.cmd_turn_off_loop_state()
        self._bpmxenbl = self.devices['sofb'].bpmxenbl
        self._bpmyenbl = self.devices['sofb'].bpmyenbl

    def restore_sofb_reforb(self):
        """."""
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        clt = self._configdb
        ref_orb = clt.get_config_value('ref_orb')
        refx = _np.array(ref_orb['x'])
        refy = _np.array(ref_orb['y'])
        sofb.refx = refx
        sofb.refy = refy
        sofb.bpmxenbl = _np.ones(refx.size, dtype=bool)
        sofb.bpmyenbl = _np.ones(refx.size, dtype=bool)
        if self.params.use_fofb:
            fofb.bpmxenbl = self._orig_fofb_bpmxenbl
            fofb.bpmyenbl = self._orig_fofb_bpmyenbl

    def remove_bpms(self, section_type, section_nr, n_bpms_out):
        """Remove BPMs from correction system.

        Args:
            section_type (str): Section type. Ex: "C1", "C2"
            section_nr (int): Number of section
            n_bpms_out (int): Number of BPMs to remove from each
             side of the bump.
        """
        sofb = self.devices['sofb']
        idcs_out = self.bumptools.get_closest_bpms_indices(
            section_type=section_type,
            sidx=section_nr - 1,
            n_bpms_out=n_bpms_out,
        )
        enblx = _np.ones(self.reforbx.size, dtype=bool)
        enbly = _np.ones(self.reforby.size, dtype=bool)
        enblx[idcs_out[: n_bpms_out * 2]] = False
        enbly[idcs_out[n_bpms_out * 2:] - 160] = False

        sofb.bpmxenbl = enblx
        sofb.bpmyenbl = enbly
        sofb.bpmxenbl[idcs_out[: n_bpms_out * 2]] = False
        sofb.bpmyenbl[idcs_out[n_bpms_out * 2:] - 160] = False
        if self.params.use_fofb:
            fofb = self.devices['fofb']
            enblx = _np.copy(self._orig_fofb_bpmxenbl)
            enbly = _np.copy(self._orig_fofb_bpmyenbl)
            enblx[idcs_out[: n_bpms_out * 2]] = False
            enbly[idcs_out[n_bpms_out * 2:] - 160] = False
            fofb.bpmxenbl = enblx
            fofb.bpmyenbl = enbly
        _time.sleep(0.5)  # NOTE: For some reason We have to wait here.

    def get_orbrms(self, refx, refy, idx):
        """Calculate rms of orbit distortion.

        Args:
            refx (1d numpy array): Horizontal ref. orb
            refy (1d numpy array): Vertical ref. orb
            idx (1d numpy array): Indices of BPMs used in bump.

        Returns:
            float: rms of orbit distortion
        """
        idcx = idx[:2]
        idcy = idx[2:] - 160
        dorbx = self.devices['sofb'].orbx - refx
        dorby = self.devices['sofb'].orby - refy
        dorbx = dorbx[idcx]
        dorby = dorby[idcy]
        return _np.sqrt(_np.sum(_np.hstack([dorbx, dorby])))

    def set_orb(self, orbx, orby):
        """Update orbit of corr. sytems.

        Args:
            orbx (1d numpy array): Horizontal orbit
            orby (1d numpy array): Vertical orbit
        """
        if self.params.use_fofb:
            fofb = self.devices['fofb']
            fofb.refx = orbx
            fofb.refy = orby
        sofb = self.devices['sofb']
        sofb.refx = orbx
        sofb.refy = orby

    def implement_bump(
        self, refx=None, refy=None, agx=0, agy=0, psx=0, psy=0, subsec=None
    ):
        """."""
        sofb = self.devices['sofb']
        refx0 = refx or self.reforbx
        refy0 = refy or self.reforby
        subsec = subsec or self.params.subsec
        n_bpms_out = self.params.n_bpms_out
        minsingval = self.params.minsingval
        nr_iters = self.params.orbcorr_nr_iters
        residue = self.params.orbcorr_residue
        bump_residue = self.params.bump_residue
        bump_max_residue = self.params.bump_max_residue
        fofb_max_kick = self.params.fofb_max_kick

        refx, refy = sofb.si_calculate_bumps(
            refx0,
            refy0,
            subsec,
            agx=agx,
            agy=agy,
            psx=psx,
            psy=psy,
            n_bpms_out=n_bpms_out,
            minsingval=minsingval,
        )
        section_type, section_nr = self.subsec_2_sectype_nr(subsec)


        idcs_bpm = self.bumptools.get_bpm_indices(
            section_type=section_type, sidx=section_nr - 1
        )

        # Set orbit
        self.set_orb(refx, refy)

        # Verify orbit correction
        rms_residue = bump_residue + 1
        kick = fofb_max_kick - 1
        if self.params.use_fofb:
            fofb = self.devices['fofb']
        print('Waiting orbit...')
        while rms_residue > bump_residue or kick > fofb_max_kick:
            rms_residue = self.get_orbrms(refx, refy, idcs_bpm)
            if self.params.use_fofb:
                kick = _np.max((
                    _np.abs(fofb.kickch_acc),
                    _np.abs(fofb.kickcv_acc),
                ))
            else:
                _ = sofb.correct_orbit_manually(
                    nr_iters=nr_iters, residue=residue
                )
            print(
                f'    orb_rms = {rms_residue:.3f} um, '
                f'maxkick = {kick:.3f} urad'
            )
            bump_residue *= 1.2
            if bump_residue > bump_max_residue:
                raise ValueError('Could not correct orbit.')

        print('Done!')

    def _do_scan(self):
        subsec = self.params.subsec
        n_bpms_out = self.params.n_bpms_out
        section_type, section_nr = self.subsec_2_sectype_nr(subsec)
        self.remove_bpms(section_type, section_nr, n_bpms_out)
        prms = self.params
        if prms.do_angular_bumps:
            x_span, y_span = prms.pts_agx, prms.pts_agy
        else:
            x_span, y_span = prms.pts_psx, prms.pts_psy

        # zig-zag type of scan in the y plane
        idy, idx = _np.meshgrid(range(len(x_span)), range(len(y_span)))
        idy[1::2] = _np.flip(idy[1::2])
        idx, idy = idx.ravel(), idy.ravel()

        data = list()
        for i in range(idx.size):
            if not self._is_beam_alive():
                break
            x = x_span[idx[i]]
            y = y_span[idy[i]]

            self.config_sofb()
            if prms.do_angular_bumps:
                self.implement_bump(agx=x, agy=y)
                unit = 'urad'
            else:
                self.implement_bump(psx=x, psy=y)
                unit = 'um'
            _time.sleep(prms.wait_meas)
            data.append(self.get_measurement_data())
            fstr = f'(x, y) = ({x:6.1f}, {y:6.1f}) ' + unit
            print(fstr)
            data[-1]['bump'] = (x, y)
            self.data = data
            if self._stopevt.is_set():
                print('Stopping...')
                break

        # return to initial reference orbit
        print('Returning to ref_orb...')
        self.restore_sofb_reforb()
        print('Finished!')
