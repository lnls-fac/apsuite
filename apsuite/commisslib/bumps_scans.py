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
        self.pts_psx = _np.array([0])  # 1d numpy array [um]
        self.pts_psy = _np.array([0])  # 1d numpy array [um]
        self.pts_agx = _np.array([0])  # 1d numpy array [urad]
        self.pts_agy = _np.array([0])  # 1d numpy array [urad]
        self.subsec = '01C1'
        self.do_angular_bumps = True

        self.n_bpms_out = 3
        self.minsingval = 0.2
        self.bump_residue = 3  # [um]
        self.bump_max_residue = 10  # [um]
        self.fofb_max_kick = 4  # [urad]

        self.buffer_sloworb = 20
        self.wait_meas = 2  # [s]
        self.orbcorr_nr_iters = 5
        self.orbcorr_residue = 5  # [um]

        self.closed_loops = False
        self.timeout_fofb_ramp = 15

        self.sleep_time = 0.5  # [s]

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:9.2f} - {2:9.2f} {3:s} with {4:1.0f} pts\n'.format
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

        stg += dtmp('closed_loops', self.closed_loops)
        stg += dtmp('timeout_fofb_ramp', self.timeout_fofb_ramp, '')
        stg += dtmp('sleep_time', self.sleep_time, '')

        stg += stmp(
            'pts_psx',
            self.pts_psx[0],
            self.pts_psx[-1],
            '[um]',
            len(self.pts_psx),
        )
        stg += stmp(
            'pts_psy',
            self.pts_psy[0],
            self.pts_psy[-1],
            '[um]',
            len(self.pts_psy),
        )
        stg += stmp(
            'pts_agx',
            self.pts_agx[0],
            self.pts_agx[-1],
            '[urad]',
            len(self.pts_agx),
        )
        stg += stmp(
            'pts_agy',
            self.pts_agy[0],
            self.pts_agy[-1],
            '[urad]',
            len(self.pts_agy),
        )
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
        self.bumptools = SiCalcBumps()
        if self.isonline:
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['fofb'] = HLFOFB(HLFOFB.DEVICES.SI)
            self.devices['currinfo'] = CurrInfoSI()

    @staticmethod
    def do_measurement():
        """Measurement function.

        Change this function by an external
        function that does the measurement
        you want to perform at each bump point.
        """
        print('Not a measurement!')
        return {}

    def get_initial_state(self):
        """Get initial state of the SOFB and FOFB."""
        clt = _ConfigDBClient(config_type='si_orbit')
        ref_orb = clt.get_config_value('ref_orb')
        refx = _np.array(ref_orb['x'])
        refy = _np.array(ref_orb['y'])
        self.reforbx = refx
        self.reforby = refy
        self.get_sofb_bpm_enbl()
        self.get_fofb_bpm_enbl()
        self._initial_state = {}
        self._initial_state['refx'] = refx
        self._initial_state['refy'] = refy
        self._initial_state['bpmxenbl'] = self._bpmxenbl
        self._initial_state['bpmyenbl'] = self._bpmyenbl
        self._initial_state['fofb_bpmxenbl'] = self._fofb_bpmxenbl
        self._initial_state['fofb_bpmyenbl'] = self._fofb_bpmyenbl
        self.data['initial_state'] = self._initial_state

    def _is_beam_alive(self):
        if self.devices['currinfo'].storedbeam:
            return True
        print('Beam is dead!')
        self.restore_initial_state()

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
        print('Configuring sofb...')
        if sofb.autocorrsts or fofb.loop_state:
            fofb.cmd_turn_off_loop_state(timeout=self.params.timeout_fofb_ramp)
            sofb.cmd_turn_off_autocorr()
        sofb.nr_points = self.params.buffer_sloworb
        sofb.cmd_change_opmode_to_sloworb()
        sofb.cmd_reset()
        # NOTE: the factor of 8 is because the current version of SOFB is too
        # slow to update multi-turn orbits.
        sofb.wait_buffer(timeout=sofb.nr_points * 0.5 * 8)
        print('Done!')
        if self.params.closed_loops:
            sofb.cmd_turn_on_autocorr()
            fofb.cmd_turn_on_loop_state()
        else:
            sofb.cmd_turn_off_autocorr()
            fofb.cmd_turn_off_loop_state()

    def get_sofb_bpm_enbl(self):
        """."""
        self._bpmxenbl = _np.copy(self.devices['sofb'].bpmxenbl)
        self._bpmyenbl = _np.copy(self.devices['sofb'].bpmyenbl)

    def get_fofb_bpm_enbl(self):
        """."""
        self._fofb_bpmxenbl = _np.copy(self.devices['fofb'].bpmxenbl)
        self._fofb_bpmyenbl = _np.copy(self.devices['fofb'].bpmyenbl)

    def _generate_bpm_enbl(self, n_bpms_out, enblx, enbly, idcs_out):
        if n_bpms_out != 0:
            enblx[idcs_out[: n_bpms_out * 2]] = False
            enbly[idcs_out[n_bpms_out * 2 :] - 160] = False
        return enblx, enbly

    def restore_initial_state(self):
        """."""
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        sofb.refx = self.reforbx
        sofb.refy = self.reforby
        sofb.bpmxenbl = self._bpmxenbl
        sofb.bpmyenbl = self._bpmyenbl
        if self.params.closed_loops:
            fofb.bpmxenbl = self._fofb_bpmxenbl
            fofb.bpmyenbl = self._fofb_bpmyenbl

    def remove_bpms(self):
        """Remove BPMs from correction system."""
        subsec = self.params.subsec
        n_bpms_out = self.params.n_bpms_out
        section_type, section_nr = self.subsec_2_sectype_nr(subsec)

        sofb = self.devices['sofb']
        idcs_out = self.bumptools.get_closest_bpms_indices(
            section_type=section_type,
            sidx=section_nr - 1,
            n_bpms_out=n_bpms_out,
        )
        enblx = self._bpmxenbl
        enbly = self._bpmyenbl
        enblx, enbly = self._generate_bpm_enbl(
            n_bpms_out, enblx, enbly, idcs_out
        )
        sofb.bpmxenbl = enblx
        sofb.bpmyenbl = enbly

        if self.params.closed_loops:
            fofb = self.devices['fofb']
            enblx = self._fofb_bpmxenbl
            enbly = self._fofb_bpmyenbl
            enblx, enbly = self._generate_bpm_enbl(
                n_bpms_out, enblx, enbly, idcs_out
            )
            fofb.bpmxenbl = enblx
            fofb.bpmyenbl = enbly
        _time.sleep(
            self.params.sleep_time
        )  # NOTE: For some reason We have to wait here.

    def get_orbrms(self, refx, refy, idx):
        """Calculate rms of orbit distortion.

        Args:
            refx (1d numpy array): Horizontal ref. orb
            refy (1d numpy array): Vertical ref. orb
            idx (1d numpy array): Indices of BPMs used in bump.

        Returns:
            float: rms of orbit distortion
        """
        sofb = self.devices['sofb']
        ref = _np.r_[refx, refy]
        orb = _np.r_[sofb.orbx, sofb.orby]
        dorb = (orb - ref)[idx] ** 2
        return _np.sqrt(dorb.sum())

    def set_reforb(self, orbx, orby):
        """Update orbit of corr. sytems.

        Args:
            orbx (1d numpy array): Horizontal orbit
            orby (1d numpy array): Vertical orbit
        """
        if self.params.closed_loops:
            fofb = self.devices['fofb']
            fofb.refx = orbx
            fofb.refy = orby
        sofb = self.devices['sofb']
        sofb.refx = orbx
        sofb.refy = orby

    def _check_rms_conditions(self, rms_residue, bump_residue):
        sofb = self.devices['sofb']
        bump_max_residue = self.params.bump_max_residue
        sofb.cmd_reset()
        sofb.wait_buffer()
        print(
            f'    orb_rms = {rms_residue:.3f} um, '
            f'    bump_rms = {bump_residue:.3f} um, '
        )
        bump_residue *= 1.2
        if bump_residue > bump_max_residue:
            raise ValueError('Could not correct orbit.')

    def implement_bump(self, agx=0, agy=0, psx=0, psy=0, subsec=None):
        """."""
        sofb = self.devices['sofb']
        refx0 = self.reforbx
        refy0 = self.reforby
        subsec = subsec or self.params.subsec
        n_bpms_out = self.params.n_bpms_out
        minsingval = self.params.minsingval
        nr_iters = self.params.orbcorr_nr_iters
        residue = self.params.orbcorr_residue
        bump_residue = self.params.bump_residue
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
        self.set_reforb(refx, refy)

        # Verify orbit correction
        rms_residue = bump_residue + 1
        kick = fofb_max_kick - 1
        if self.params.closed_loops:
            fofb = self.devices['fofb']
        print('Waiting orbit...')
        if self.params.closed_loops:
            while rms_residue > bump_residue and kick > fofb_max_kick:
                kick = _np.max((
                    _np.abs(fofb.kickch_acc),
                    _np.abs(fofb.kickcv_acc),
                ))
                rms_residue = self.get_orbrms(refx, refy, idcs_bpm)
                self._check_rms_conditions(rms_residue, bump_residue)
                print(f'    kick fofb = {kick:.3f} urad, ')
        else:
            while rms_residue > bump_residue:
                _ = sofb.correct_orbit_manually(
                    nr_iters=nr_iters, residue=residue
                )
                rms_residue = self.get_orbrms(refx, refy, idcs_bpm)
                self._check_rms_conditions(rms_residue, bump_residue)
        print('Done!')

    def do_scan(self):
        """Start bumps scan."""
        prms = self.params
        if prms.do_angular_bumps:
            x_span, y_span = prms.pts_agx, prms.pts_agy
        else:
            x_span, y_span = prms.pts_psx, prms.pts_psy

        # zig-zag type of scan in the y plane
        idx, idy = _np.meshgrid(range(len(x_span)), range(len(y_span)))
        idx[1::2] = _np.flip(idx[1::2])
        idy, idx = idy.ravel(), idx.ravel()

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
            data[-1]['orbx'] = self.devices['sofb'].orbx
            data[-1]['orby'] = self.devices['sofb'].orby
            self.data['bumps_data'] = data
            if self._stopevt.is_set():
                print('Stopping...')
                break

        print('Finished!')
