"""Classes for FOFB system identification."""

import time as _time
import numpy as _np

from siriuspy.devices import CurrInfoSI, \
    Trigger, RFGen, FamFOFBSysId, BPM, HLFOFB, SOFB

from ..asparams import SI_NUM_BPMS
from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class FOFBSysIdAcqParams(_ParamsBaseClass):
    """FOFB system identification acquisition parameters."""

    def __init__(self):
        # timing
        self.trigger = 'SI-Fam:TI-FOFB'
        self.trigger_delay = 0.0
        self.trigger_nrpulses = 1
        self.event = 'FOFBS'
        self.event_delay = 0.0
        self.event_mode = 'External'
        # acquisition
        self.acq_timeout = 40
        self.acq_nrpoints_before = 0
        self.acq_nrpoints_after = 93000
        self.acq_channel = 3  # sysid_applied
        self.acq_repeat = False
        self.acq_external = True
        # prbs signal
        self.prbs_step_duration = 10
        self.prbs_lfsr_length = 5
        self.prbs_mov_avg_taps = 4
        # prbs SVD levels settings
        self.svd_levels_regularize_matrix = True
        self.svd_levels_reg_sinval_min = 0.01
        self.svd_levels_reg_tikhonov_const = 0
        self.svd_levels_bpmsx_enbllist = _np.ones(SI_NUM_BPMS, dtype=bool)
        self.svd_levels_bpmsy_enbllist = _np.ones(SI_NUM_BPMS, dtype=bool)
        self.svd_levels_ch_enbllist = _np.ones(80, dtype=bool)
        self.svd_levels_cv_enbllist = _np.ones(80, dtype=bool)
        self.svd_levels_rf_enbllist = _np.ones(1, dtype=bool)
        self.svd_levels_respmat = _np.zeros((320, 161))
        self.svd_levels_singmode_idx = 0
        self.svd_levels_ampmax = 5000
        # prbs fofbacc levels
        self.prbs_fofbacc_enbl = False
        self.prbs_fofbacc_lvl0 = _np.zeros(160)
        self.prbs_fofbacc_lvl1 = _np.zeros(160)
        # prbs bpmpos levels
        self.prbs_bpmpos_enbl = False
        self.prbs_bpmposx_lvl0 = _np.zeros(160)
        self.prbs_bpmposx_lvl1 = _np.zeros(160)
        self.prbs_bpmposy_lvl0 = _np.zeros(160)
        self.prbs_bpmposy_lvl1 = _np.zeros(160)
        self.prbs_bpms_to_get_data = _np.ones(160, dtype=bool)
        self.prbs_corrs_to_get_data = _np.ones(160, dtype=bool)
        # power supply current loop
        self.corr_currloop_kp = 5000000*_np.ones(160)
        self.corr_currloop_ki = 2000*_np.ones(160)

    def __str__(self):
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:26s} = {1:9}  {2:s}\n'.format
        stg = ''
        stg += stmp('trigger', self.trigger, '')
        stg += ftmp('trigger_delay', self.trigger_delay, '[us]')
        stg += dtmp('trigger_nrpulses', self.trigger_nrpulses, '')
        stg += stmp('event', self.event, '')
        stg += ftmp('event_delay', self.event_delay, '[us]')
        stg += stmp('event_mode', self.event_mode, '')
        stg += ftmp('acq_timeout', self.acq_timeout, '[s]')
        stg += dtmp('acq_nrpoints_before', self.acq_nrpoints_before, '')
        stg += dtmp('acq_nrpoints_after', self.acq_nrpoints_after, '')
        stg += dtmp('acq_channel', self.acq_channel, '')
        stg += dtmp('acq_repeat', self.acq_repeat, '')
        stg += dtmp('acq_external', self.acq_external, '')
        stg += dtmp('prbs_step_duration', self.prbs_step_duration, '')
        stg += dtmp('prbs_lfsr_length', self.prbs_lfsr_length, '')
        stg += dtmp('prbs_mov_avg_taps', self.prbs_mov_avg_taps, '')
        stg += dtmp(
            'svd_levels_regularize_matrix',
            self.svd_levels_regularize_matrix, '')
        stg += ftmp('svd_levels_reg_sinval_min',
                    self.svd_levels_reg_sinval_min, '')
        stg += ftmp('svd_levels_reg_tikhonov_const',
                    self.svd_levels_reg_tikhonov_const, '')
        stg += stmp('svd_levels_bpmsx_enbllist',
                    str(self.svd_levels_bpmsx_enbllist), '')
        stg += stmp('svd_levels_bpmsy_enbllist',
                    str(self.svd_levels_bpmsy_enbllist), '')
        stg += stmp('svd_levels_ch_enbllist',
                    str(self.svd_levels_ch_enbllist), '')
        stg += stmp('svd_levels_cv_enbllist',
                    str(self.svd_levels_cv_enbllist), '')
        stg += stmp('svd_levels_rf_enbllist',
                    str(self.svd_levels_rf_enbllist), '')
        stg += stmp('svd_levels_respmat', str(self.svd_levels_respmat), '')
        stg += dtmp('svd_levels_ampmax', self.svd_levels_ampmax, '')
        stg += dtmp(
            'svd_levels_singmode_idx', self.svd_levels_singmode_idx, '')
        stg += stmp('prbs_fofbacc_enbl', str(self.prbs_fofbacc_enbl), '')
        stg += stmp('prbs_fofbacc_lvl0', str(self.prbs_fofbacc_lvl0), '')
        stg += stmp('prbs_fofbacc_lvl1', str(self.prbs_fofbacc_lvl1), '')
        stg += stmp('prbs_bpmpos_enbl', str(self.prbs_bpmpos_enbl), '')
        stg += stmp('prbs_bpmposx_lvl0', str(self.prbs_bpmposx_lvl0), '')
        stg += stmp('prbs_bpmposx_lvl1', str(self.prbs_bpmposx_lvl1), '')
        stg += stmp('prbs_bpmposy_lvl0', str(self.prbs_bpmposy_lvl0), '')
        stg += stmp('prbs_bpmposy_lvl1', str(self.prbs_bpmposy_lvl1), '')
        stg += stmp('corr_currloop_kp', str(self.corr_currloop_kp), '')
        stg += stmp('corr_currloop_ki', str(self.corr_currloop_ki), '')
        return stg


class FOFBSysIdAcq(_BaseClass):
    """FOFB system identification acquisition."""

    def __init__(self, isonline=True):
        super().__init__(params=FOFBSysIdAcqParams(), isonline=isonline)
        self._fname = None
        if self.isonline:
            self.create_devices()

    def create_devices(self):
        """Connect to devices."""
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['famsysid'] = FamFOFBSysId()
        self.devices['trigger'] = Trigger(self.params.trigger)
        self.devices['event'] = self.devices['famsysid'].evtdev
        self.devices['rfgen'] = RFGen()
        self.devices['auxbpm'] = BPM('SI-01M2:DI-BPM')
        self.devices['fofb'] = HLFOFB()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)

    @property
    def fname(self):
        """Filename with acquisition data."""
        return self._fname

    @fname.setter
    def fname(self, val):
        self._fname = val

    # ---- calculate excitation arrays ----

    def _calc_svd(self, respm):
        """Calculate matrix SVD."""
        bpmxenbl = self.params.svd_levels_bpmsx_enbllist
        bpmyenbl = self.params.svd_levels_bpmsy_enbllist
        chenbl = self.params.svd_levels_ch_enbllist
        cvenbl = self.params.svd_levels_cv_enbllist
        rfenbl = self.params.svd_levels_rf_enbllist

        selbpm = _np.hstack([bpmxenbl, bpmyenbl])
        selcorr = _np.hstack([chenbl, cvenbl, rfenbl])
        selmat = selbpm[:, None] * selcorr[None, :]
        if selmat.size != respm.size:
            raise ValueError(
                f'Incompatiple selection ({selmat.size}) '
                f'and matrix size {respm.size}')
        mat = respm.copy()
        mat = mat[selmat]
        mat = _np.reshape(mat, [selbpm.sum(), selcorr.sum()])

        # calculate SVD for converted matrix
        _uc, _sc, _vc = _np.linalg.svd(mat, full_matrices=False)
        u11 = _np.zeros((respm.shape[0], _sc.size))
        v11 = _np.zeros((_sc.size, respm.shape[1]))
        u11[selbpm, :] = _uc
        v11[:, selcorr] = _vc

        return u11, _sc, v11

    def _calc_matrix_regularization(self, respm):
        """Calculate matriz regularization."""
        sinval_min = self.params.svd_levels_reg_sinval_min
        tikhonov_reg = self.params.svd_levels_reg_tikhonov_const

        _uo, _so, _vo = self._calc_svd(respm)

        # handle singular values
        # select singular values greater than minimum
        idcs = _so > sinval_min
        _sr = _so[idcs]
        nrs = _np.sum(idcs)
        if not nrs:
            raise ValueError('All Singular Values below minimum.')
        # apply Tikhonov regularization
        regc = tikhonov_reg
        regc *= regc
        inv_s = _np.zeros(_so.size, dtype=float)
        inv_s[idcs] = _sr/(_sr*_sr + regc)
        # calculate processed singular values
        _sp = _np.zeros(_so.size, dtype=float)
        _sp[idcs] = 1/inv_s[idcs]

        # reconstruct filtered and regularized matrix in physical units
        respmat_proc = _np.dot(_uo*_sp, _vo)
        return respmat_proc

    def _convert_respmat_from_phys2hard_units(self, respm):
        # convert matrix to hardware units
        famsysid = self.devices['famsysid']
        str2curr = _np.r_[famsysid.strength_2_current_factor, 1.0]
        # unit convertion: um/urad (1)-> nm/urad (2)-> nm/A
        matc = respm * 1e3
        matc = matc / str2curr
        return respm

    def _convert_respmat_from_hard2phys_units(self, respm):
        # convert matrix to physics units
        famsysid = self.devices['famsysid']
        str2curr = _np.r_[famsysid.strength_2_current_factor, 1.0]
        # unit convertion: nm/A -> nm/urad (1) -> um/urad (2)
        matc = respm * str2curr
        matc /= 1e3
        return respm

    def get_levels_corrs_from_svd(self, lvl0, lvl1=None):
        """Get levels from SVD for corrector devices.

        Args:
            lvl0 (int): maximum level for PRBS level 0
            lvl1 (optional, int): maximum level for PRBS level 1
                If None, we consider level1 = -level0. Defaults to None.

        Returns:
            lvls0 (numpy.ndarray, 160):
                array with FOFBAcc level for PRBS level 0
            lvls1 (numpy.ndarray, 160):
                array with FOFBAcc level for PRBS level 1

        """
        respm = self.params.svd_levels_respmat
        singval = self.params.svd_levels_singmode_idx

        if self.params.svd_levels_regularize_matrix:
            respm = self._calc_matrix_regularization(respm)
        respm = self._convert_respmat_from_phys2hard_units(respm)

        *_, v = self._calc_svd(respm)
        vs = v[singval]
        vs /= _np.abs(vs).max()
        if lvl1 is None:
            lvl1 = - lvl0
        amp = (lvl1-lvl0)/2
        off = (lvl1+lvl0)/2
        lvls0 = off - amp * vs
        lvls1 = off + amp * vs
        return lvls0[:-1], lvls1[:-1]

    def get_levels_bpms_from_svd(self, ampmax, lvl0, lvl1=None):
        """Get levels from SVD for BPMs devices.

        Args:
            lvl0 (int): minimum level for PRBS level 0
            lvl1 (optional, int): minimum level for PRBS level 1
                If None, we consider level1 = -level0. Defaults to None.
            ampmax (int): maximum level after SV scaling

        Returns:
            lvls0x (numpy.ndarray, SI_NUM_BPMS):
                array with BPM Pos X level for PRBS level 0
            lvls0y (numpy.ndarray, SI_NUM_BPMS):
                array with BPM Pos Y level for PRBS level 0
            lvls1x (numpy.ndarray, SI_NUM_BPMS):
                array with BPM Pos X level for PRBS level 1
            lvls1y (numpy.ndarray, SI_NUM_BPMS):
                array with BPM Pos Y level for PRBS level 1

        """
        respm = self.params.svd_levels_respmat
        # Zeroing RF column
        respm[:, -1].fill(0)
        singval = self.params.svd_levels_singmode_idx

        if self.params.svd_levels_regularize_matrix:
            respm = self._calc_matrix_regularization(respm)
        respm = self._convert_respmat_from_phys2hard_units(respm)

        u, s, _ = self._calc_svd(respm)
        us = u[:, singval]
        ss = s[singval]
        ss /= _np.abs(s).min()

        if lvl1 is None:
            amp = _np.abs(lvl0)
            # Scales the amplitude with its corresponding singular value
            # (normalized to the lesser one).
            # The amplitudes are saturated to 'ampmax'.
            amp = min(amp * ss, ampmax)
            lvls0 = - amp * us
            lvls0x, lvls0y = lvls0[:SI_NUM_BPMS], lvls0[SI_NUM_BPMS:],
            return lvls0x, lvls0y

        else:
            amp = (lvl1-lvl0)/2
            off = (lvl1+lvl0)/2
            # Scales the amplitude with its corresponding singular value
            # (normalized to the lesser one).
            # The amplitudes are saturated to 'ampmax'.
            amp = min(amp * ss, ampmax)
            lvls0 = off - amp * us
            lvls1 = off + amp * us
            lvls0x, lvls1x = lvls0[:SI_NUM_BPMS], lvls1[:SI_NUM_BPMS]
            lvls0y, lvls1y = lvls0[SI_NUM_BPMS:], lvls1[SI_NUM_BPMS:]
            return lvls0x, lvls0y, lvls1x, lvls1y

    def get_levels_corrs_indiv_exc(self, corrname, lvl0, lvl1=None):
        """Get levels for excitation with only one corrector."""
        famsysid = self.devices['famsysid']
        corrindex = famsysid.psnames.index(corrname)
        lvls0 = _np.zeros(len(famsysid.psnames))
        lvls0[corrindex] = lvl0
        if lvl1 is None:
            lvl1 = -lvl0
        lvls1 = _np.zeros(len(famsysid.psnames))
        lvls1[corrindex] = lvl1
        return lvls0, lvls1

    # ---- interact with devices ----

    def prepare_timing(self):
        """Prepare timing for acquisitions."""
        self.devices['trigger'].delay = self.params.trigger_delay
        self.devices['trigger'].nr_pulses = self.params.trigger_nrpulses
        self.devices['trigger'].source = self.params.event
        self.devices['event'].delay = self.params.event_delay
        self.devices['event'].mode = self.params.event_mode

        # Update event configurations in EVG
        self.devices['evg'].cmd_update_events()

    def trigger_timing_signal(self):
        """Trigger timing."""
        self.devices['event'].cmd_external_trigger()

    def sync_prbs(self):
        """Prepare PRBS signal for all FOFB controllers."""
        return self.devices['famsysid'].cmd_sync_prbs()

    def prepare_prbs(self):
        """Prepare PRBS signal for all FOFB controllers."""
        famsysid = self.devices['famsysid']

        ret = famsysid.config_prbs(
            step_duration=self.params.prbs_step_duration,
            lfsr_len=self.params.prbs_lfsr_length)
        if not ret:
            print('Could not configurate PRBS.')
            return False

        ret = famsysid.set_prbs_mov_avg_taps(self.params.prbs_mov_avg_taps)
        if not ret:
            print('Could not set number of taps of moving average filter.')
            return False

        return True

    def prepare_fofbacc_prbs(self):
        """Prepare FOFBAcc PRBS levels."""
        famsysid = self.devices['famsysid']

        lvl0 = self.params.prbs_fofbacc_lvl0
        lvl1 = self.params.prbs_fofbacc_lvl1
        famsysid.set_prbs_fofbacc_levels(lvl0, lvl1)
        ret = famsysid.wait_prbs_fofbacc_levels(lvl0, lvl1)
        if not ret:
            print('FOFBAcc PRBS levels not applied')
            return False

        if self.params.prbs_fofbacc_enbl:
            ret = famsysid.cmd_prbs_fofbacc_enable()
        else:
            ret = famsysid.cmd_prbs_fofbacc_disable()
        if not ret:
            print('FOFBAcc PRBS enable state not applied')
            return False

        return True

    def prepare_bpms_prbs(self):
        """Prepare BPM Pos PRBS levels."""
        famsysid = self.devices['famsysid']
        lvl0x = self.params.prbs_bpmposx_lvl0
        lvl1x = self.params.prbs_bpmposx_lvl1

        famsysid.set_prbs_bpmposx_levels(lvl0x, lvl1x)
        ret = famsysid.wait_prbs_bpmposx_levels(lvl0x, lvl1x)
        if not ret:
            print('BPM PosX PRBS levels not applied')
            return False

        lvl0y = self.params.prbs_bpmposy_lvl0
        lvl1y = self.params.prbs_bpmposy_lvl1
        famsysid.set_prbs_bpmposy_levels(lvl0y, lvl1y)
        ret = famsysid.wait_prbs_bpmposy_levels(lvl0y, lvl1y)
        if not ret:
            print('BPM PosY PRBS levels not applied')
            return False

        if self.params.prbs_bpmpos_enbl:
            ret = famsysid.cmd_prbs_bpms_enable()
        else:
            ret = famsysid.cmd_prbs_bpms_disable()
        if not ret:
            print('BPM Pos PRBS enable state not applied')
            return False

        return True

    def prepare_acquisition(self):
        """Prepare acquisition."""
        return self.devices['famsysid'].config_acquisition(
            nr_points_after=self.params.acq_nrpoints_after,
            nr_points_before=self.params.acq_nrpoints_before,
            channel=self.params.acq_channel,
            repeat=self.params.acq_repeat,
            external=self.params.acq_external)

    def check_data_valid(self):
        """Check whether data is valid."""
        return self.devices['famsysid'].check_data_valid()

    def acquire_data(self, wait_time=None):
        """Acquire data."""
        ret = self.prepare_acquisition()
        if ret < 0:
            print(f'FOFB controller {-ret} did not finish last acquisition.')
            return False
        elif ret > 0:
            print(f'FOFB controller {ret} is not ready for acquisition.')
            return False

        self.devices['famsysid'].update_initial_timestamps(
            bpmenbl=self.params.prbs_bpms_to_get_data,
            correnbl=self.params.prbs_corrs_to_get_data)

        if wait_time is not None:
            _time.sleep(wait_time)

        self.trigger_timing_signal()

        time0 = _time.time()
        ret = self.devices['famsysid'].wait_update_data(
            timeout=self.params.acq_timeout,
            bpmenbl=self.params.prbs_bpms_to_get_data,
            correnbl=self.params.prbs_corrs_to_get_data)
        print(f'It took {_time.time()-time0:02f}s to update bpms')
        if ret != 0:
            print(f'There was a problem with acquisition. Error code {ret:d}')
            return False

        ret = self.check_data_valid()
        if ret < 0:
            print('FOFB controller 1 TimeFrameData is not monotonic.')
            return False
        if ret > 0:
            print(
                f'FOFB controller {ret} has data different from controller 1.')
            return False

        self.data = self.get_data()

        return True

    def get_data(self):
        """Get data."""
        famsysid = self.devices['famsysid']

        data = dict()
        data['timestamp'] = _time.time()
        data['psnames'] = famsysid.psnames
        data['bpmnames'] = famsysid.bpmnames

        # timeframe data
        data['timeframe_data'] = famsysid.timeframe_data

        # prbs data
        data['prbs_data'] = famsysid.prbs_data
        data['prbs_step_duration'] = famsysid.prbs_step_duration
        data['prbs_lfsr_len'] = famsysid.prbs_lfsr_len

        # prbs excitation config
        data['prbs_fofbacc_enbl'] = famsysid.prbs_fofbacc_enbl
        data['prbs_fofbacc_lvl0'] = famsysid.prbs_fofbacc_lvl0
        data['prbs_fofbacc_lvl1'] = famsysid.prbs_fofbacc_lvl1
        data['prbs_bpmpos_enbl'] = famsysid.prbs_bpmpos_enbl
        data['prbs_bpmposx_lvl0_beam_order'] = _np.roll(
            famsysid.prbs_bpmposx_lvl0, -1
            )
        data['prbs_bpmposx_lvl1_beam_order'] = _np.roll(
            famsysid.prbs_bpmposx_lvl1, -1
            )
        data['prbs_bpmposy_lvl0_beam_order'] = _np.roll(
            famsysid.prbs_bpmposy_lvl0, -1
            )
        data['prbs_bpmposy_lvl1_beam_order'] = _np.roll(
            famsysid.prbs_bpmposy_lvl1, -1
            )
        data['corr_currloop_kp'] = famsysid.currloop_kp
        data['corr_currloop_ki'] = famsysid.currloop_ki

        # acquisition
        orbx, orby, currdata, kickdata = famsysid.get_data(
            bpmenbl=self.params.prbs_bpms_to_get_data,
            correnbl=self.params.prbs_corrs_to_get_data)
        data['orbx'], data['orby'] = orbx, orby
        data['currdata'], data['kickdata'] = currdata, kickdata

        # fofb
        fofb = self.devices['fofb']
        data['fofb_loop_state'] = fofb.loop_state
        data['fofb_loop_gain_h_mon'] = fofb.loop_gain_h_mon
        data['fofb_loop_gain_v_mon'] = fofb.loop_gain_v_mon
        data['fofb_ch_accsatmax'] = fofb.ch_accsatmax
        data['fofb_cv_accsatmax'] = fofb.cv_accsatmax
        data['fofb_refx'] = fofb.refx
        data['fofb_refy'] = fofb.refy
        data['fofb_bpmxenbl'] = fofb.bpmxenbl
        data['fofb_bpmyenbl'] = fofb.bpmyenbl
        data['fofb_chenbl'] = fofb.chenbl
        data['fofb_cvenbl'] = fofb.cvenbl
        data['fofb_rfenbl'] = fofb.rfenbl
        data['fofb_singval_min'] = fofb.singval_min
        data['fofb_tikhonov_reg_const'] = fofb.tikhonov_reg_const
        data['fofb_singvalsraw_mon'] = fofb.singvalsraw_mon
        data['fofb_singvals_mon'] = fofb.singvals_mon
        data['fofb_respmat'] = fofb.respmat
        data['fofb_respmat_mon'] = fofb.respmat_mon
        data['fofb_invrespmat_mon'] = fofb.invrespmat_mon
        data['psconfig_mat'] = fofb.psconfigmat

        # sofb
        sofb = self.devices['sofb']
        data['sofb_loop_state'] = sofb.autocorrsts
        data['sofb_loop_pid_ch_kp'] = sofb.loop_pid_ch_kp
        data['sofb_loop_pid_ch_ki'] = sofb.loop_pid_ch_ki
        data['sofb_loop_pid_ch_kd'] = sofb.loop_pid_ch_kd
        data['sofb_loop_pid_cv_kp'] = sofb.loop_pid_cv_kp
        data['sofb_loop_pid_cv_ki'] = sofb.loop_pid_cv_ki
        data['sofb_loop_pid_cv_kd'] = sofb.loop_pid_cv_kd
        data['sofb_loop_pid_rf_kp'] = sofb.loop_pid_rf_kp
        data['sofb_loop_pid_rf_ki'] = sofb.loop_pid_rf_ki
        data['sofb_loop_pid_rf_kd'] = sofb.loop_pid_rf_kd
        data['sofb_refx'] = sofb.refx
        data['sofb_refy'] = sofb.refy
        data['sofb_bpmxenbl'] = sofb.bpmxenbl
        data['sofb_bpmyenbl'] = sofb.bpmyenbl
        data['sofb_chenbl'] = sofb.chenbl
        data['sofb_cvenbl'] = sofb.cvenbl
        data['sofb_rfenbl'] = sofb.rfenbl
        data['sofb_singval_min'] = sofb.singval_min
        data['sofb_respmat'] = sofb.respmat
        data['sofb_respmat_mon'] = sofb.respmat_mon
        data['sofb_invrespmat_mon'] = sofb.invrespmat

        # auxiliary data
        data['stored_current'] = self.devices['currinfo'].current
        rf_freq = self.devices['rfgen'].frequency
        data['rf_frequency'] = rf_freq
        bpmaux = self.devices['auxbpm']
        data['sampling_frequency'] = bpmaux.get_sampling_frequency(
            rf_freq, acq_rate='FOFB')

        return data
