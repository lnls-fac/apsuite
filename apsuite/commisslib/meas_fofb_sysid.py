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
        # power supply current loop
        self.corr_currloop_kp = 5000000*_np.ones(160)
        self.corr_currloop_ti = 2000*_np.ones(160)

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
        stg += dtmp('acq_data_type', self.acq_data_type, '')
        stg += ftmp('acq_timeout', self.acq_timeout, '[s]')
        stg += dtmp('acq_nrpoints_before', self.acq_nrpoints_before, '')
        stg += dtmp('acq_nrpoints_after', self.acq_nrpoints_after, '')
        stg += dtmp('acq_channel', self.acq_channel, '')
        stg += dtmp('acq_repeat', self.acq_repeat, '')
        stg += dtmp('acq_external', self.acq_external, '')
        stg += dtmp('prbs_step_duration', self.prbs_step_duration, '')
        stg += dtmp('prbs_lfsr_length', self.prbs_lfsr_length, '')
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
        stg += stmp('corr_currloop_ti', str(self.corr_currloop_ti), '')
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

    def _calc_matrix_regularization(self, respm):
        """Calculate matriz regularization."""
        bpmxenbl = self.params.svd_levels_bpmsx_enbllist
        bpmyenbl = self.params.svd_levels_bpmsy_enbllist
        chenbl = self.params.svd_levels_ch_enbllist
        cvenbl = self.params.svd_levels_cv_enbllist
        rfenbl = self.params.svd_levels_rf_enbllist
        sinval_min = self.params.svd_levels_reg_sinval_min
        tikhonov_reg = self.params.svd_levels_reg_tikhonov_const

        selbpm = _np.hstack([bpmxenbl, bpmyenbl])
        selcorr = _np.hstack([chenbl, cvenbl, rfenbl])
        selmat = selbpm[:, None] * selcorr[None, :]
        if selmat.size != respm.size:
            raise ValueError(
                f'Incompatiple selection ({selmat.size}) '
                f'and matrix size {respm.size}')
        mat = respm.copy()
        mat = mat[selmat]
        mat = _np.reshape(mat, [sum(selbpm), sum(selcorr)])

        _uo, _so, _vo = _np.linalg.svd(mat, full_matrices=False)

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
        matr = _np.dot(_uo*_sp, _vo)

        respmat_proc = _np.zeros(respm.shape, dtype=float)
        respmat_proc[selmat] = matr.ravel()
        return respmat_proc

    def _calc_svd(self, respm):
        """Calculate matrix SVD."""
        if self.params.svd_levels_regularize_matrix:
            respm = self._calc_matrix_regularization(respm)

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
        mat = _np.reshape(mat, [sum(selbpm), sum(selcorr)])

        # convert matrix to hardware units
        famsysid = self.devices['famsysid']
        str2curr = _np.r_[famsysid.strength_2_current_factor, 1.0]
        # unit convertion: um/urad (1)-> nm/urad (2)-> nm/A
        matc = mat * 1e3
        matc = matc / str2curr[selcorr]

        # calculate SVD for converted matrix
        _uc, _sc, _vc = _np.linalg.svd(matc, full_matrices=False)
        return _uc, _sc, _vc

    def get_levels_corrs_from_svd(self, lvl0, lvl1):
        """Get levels from SVD for corrector devices.

        Args:
            lvl0 (int): maximum level for PRBS level 0
            lvl1 (int): maximum level for PRBS level 1

        Returns:
            lvls0 (numpy.ndarray, 160):
                array with FOFBAcc level for PRBS level 0
            lvls1 (numpy.ndarray, 160):
                array with FOFBAcc level for PRBS level 1

        """
        chenbl = self.params.svd_levels_ch_enbllist
        cvenbl = self.params.svd_levels_cv_enbllist
        rfenbl = self.params.svd_levels_rf_enbllist
        selcorr = _np.hstack([chenbl, cvenbl, rfenbl])
        respm = self.params.svd_levels_respmat
        singval = self.params.svd_levels_singmode_idx

        u, s, v = self._calc_svd(respm)
        vs = v[singval]
        vs /= _np.abs(vs).max()
        amp = (lvl1-lvl0)/2
        off = (lvl1+lvl0)/2
        lvls0 = _np.zeros(respm.shape[1])
        lvls1 = _np.zeros(respm.shape[1])
        lvls0[selcorr] = off - amp * vs
        lvls1[selcorr] = off + amp * vs
        return lvls0[:-1], lvls1[:-1]

    def get_levels_bpms_from_svd(self, lvl0, lvl1):
        """Get levels from SVD for BPMs devices.

        Args:
            lvl0 (int): maximum level for PRBS level 0
            lvl1 (int): maximum level for PRBS level 1

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
        bpmxenbl = self.params.svd_levels_bpmsx_enbllist
        bpmyenbl = self.params.svd_levels_bpmsy_enbllist
        selbpm = _np.hstack([bpmxenbl, bpmyenbl])
        respm = self.params.svd_levels_respmat
        singval = self.params.svd_levels_singmode_idx

        u, s, v = self._calc_svd(respm)
        us = u[:, singval]
        us /= _np.abs(us).max()
        amp = (lvl1-lvl0)/2
        off = (lvl1+lvl0)/2
        lvls0 = _np.zeros(respm.shape[0])
        lvls1 = _np.zeros(respm.shape[0])
        lvls0[selbpm] = off - amp * us
        lvls1[selbpm] = off + amp * us
        lvls0x, lvls1x = lvls0[:SI_NUM_BPMS], lvls1[:SI_NUM_BPMS]
        lvls0y, lvls1y = lvls0[SI_NUM_BPMS:], lvls1[SI_NUM_BPMS:]
        return lvls0x, lvls0y, lvls1x, lvls1y

    def get_levels_corrs_indiv_exc(self, corrname, lvl0, lvl1):
        """Get levels for excitation with only one corrector."""
        famsysid = self.devices['famsysid']
        corrindex = famsysid.psnames.index(corrname)
        lvls0 = _np.zeros(len(famsysid.psnames))
        lvls0[corrindex] = lvl0
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
        return self.devices['famsysid'].config_prbs(
            step_duration=self.params.prbs_step_duration,
            lfsr_len=self.params.prbs_lfsr_length)

    def prepare_fofbacc_prbs(self):
        """Prepare FOFBAcc PRBS levels."""
        famsysid = self.devices['famsysid']

        lvl0 = self.params.prbs_fofbacc_lvl0
        lvl1 = self.params.prbs_fofbacc_lvl1
        famsysid.set_prbs_fofbacc_levels(lvl0, lvl1)
        ret = famsysid.check_prbs_fofbacc_levels(lvl0, lvl1)
        if not ret:
            print('FOFBAcc PRBS levels not applied')

        if self.params.prbs_fofbacc_enbl:
            ret = famsysid.cmd_prbs_fofbacc_enable()
        else:
            ret = famsysid.cmd_prbs_fofbacc_disable()
        if not ret:
            print('FOFBAcc PRBS enable state not applied')

    def prepare_bpms_prbs(self, from_svd=False):
        """Prepare BPM Pos PRBS levels."""
        famsysid = self.devices['famsysid']
        lvl0x = self.params.prbs_bpmposx_lvl0
        lvl1x = self.params.prbs_bpmposx_lvl1

        famsysid.set_prbs_bpmposx_levels(lvl0x, lvl1x)
        ret = famsysid.check_prbs_bpmposx_levels(lvl0x, lvl1x)
        if not ret:
            print('BPM PosX PRBS levels not applied')

        lvl0y = self.params.prbs_bpmposy_lvl0
        lvl1y = self.params.prbs_bpmposy_lvl1
        famsysid.set_prbs_bpmposy_levels(lvl0y, lvl1y)
        ret = famsysid.check_prbs_bpmposy_levels(lvl0y, lvl1y)
        if not ret:
            print('BPM PosY PRBS levels not applied')

        if self.params.prbs_bpmpos_enbl:
            ret = famsysid.cmd_prbs_bpms_enable()
        else:
            ret = famsysid.cmd_prbs_bpms_disable()
        if not ret:
            print('BPM Pos PRBS enable state not applied')

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

    def acquire_data(self):
        """Acquire data."""
        ret = self.prepare_acquisition()
        if ret < 0:
            print(f'FOFB controller {-ret} did not finish last acquisition.')
        elif ret > 0:
            print(f'FOFB controller {ret} is not ready for acquisition.')

        self.devices['famsysid'].update_initial_timestamps()

        self.trigger_timing_signal()

        time0 = _time.time()
        ret = self.devices['famsysid'].wait_update_data(
            timeout=self.params.acq_timeout)
        print(f'It took {_time.time()-time0:02f}s to update bpms')
        if ret != 0:
            print(f'There was a problem with acquisition. Error code {ret:d}')
            return

        ret = self.check_data_valid()
        if ret < 0:
            print(f'FOFB controller 1 TimeFrameData is not monotonic.')
            return
        if ret > 0:
            print(
                f'FOFB controller {ret} has data different from controller 1.')
            return

        self.data = self.get_data()

    def get_data(self):
        """Get data."""
        famsysid = self.devices['famsysid']

        data = dict()
        data['timestamp'] = _time.time()
        data['psnames'] = famsysid.psnames
        data['bpmnames'] = famsysid.bpmnames

        # prbs data
        data['prbs_data'] = famsysid.prbs_data
        data['prbs_sync_enbl'] = famsysid.prbs_sync_enbl
        data['prbs_step_duration'] = famsysid.prbs_step_duration
        data['prbs_lfsr_len'] = famsysid.prbs_lfsr_len

        # prbs excitation config
        data['prbs_fofbacc_enbl'] = famsysid.prbs_fofbacc_enbl
        data['prbs_fofbacc_lvl0'] = famsysid.prbs_fofbacc_lvl0
        data['prbs_fofbacc_lvl1'] = famsysid.prbs_fofbacc_lvl1
        data['prbs_bpmpos_enbl'] = famsysid.prbs_bpmpos_enbl
        data['prbs_bpmposx_lvl0'] = famsysid.prbs_bpmposx_lvl0
        data['prbs_bpmposx_lvl1'] = famsysid.prbs_bpmposx_lvl1
        data['prbs_bpmposy_lvl0'] = famsysid.prbs_bpmposy_lvl0
        data['prbs_bpmposy_lvl1'] = famsysid.prbs_bpmposy_lvl1
        data['corr_currloop_kp'] = famsysid.currloop_kp
        data['corr_currloop_ti'] = famsysid.currloop_ti

        # acquisition
        orbx, orby, currdata, kickdata = famsysid.get_data()
        data['orbx'], data['orby'] = orbx, orby
        data['currdata'], data['kickdata'] = currdata, kickdata

        # fofb
        fofb = self.devices['fofb']
        data['fofb_loop_state'] = fofb.loop_state
        data['fofb_loop_gain_h_mon'] = fofb.loop_gain_h_mon
        data['fofb_loop_gain_v_mon'] = fofb.loop_gain_v_mon
        data['fofb_ch_accsatmax'] = fofb.ch_accsatmax
        data['fofb_cv_accsatmax'] = fofb.cv_accsatmax
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
