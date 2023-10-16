"""."""
import time as _time
import numpy as _np
from siriuspy.devices import RFGen, SOFB, PowerSupply, Tune
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from pymodels import si
import pyaccel
from apsuite.orbcorr.calc_orbcorr_mat import OrbRespmat
from mathphys.functions import get_namedtuple as _get_namedtuple


class CorrectVerticalDispersionParams(_ParamsBaseClass):
    """."""

    CORR_METHOD = _get_namedtuple('CorrMethod', ['QS', 'CV'])

    def __init__(self):
        """."""
        super().__init__()
        self.delta_rf_freq = 100  # [Hz]
        self.nr_points_sofb = 20  # [Hz]
        self.nr_iters_corr = 1
        self.nr_svals = 100  # all qs
        self.ks_amp_factor2apply = 1
        self.qs2use_idx = _np.arange(0, 100)
        self.cv2use_idx = _np.arange(0, 160)
        self._corr_method = self.CORR_METHOD.CV

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        # stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('delta_rf_freq', self.delta_rf_freq, '[Hz]')
        stg += dtmp('nr_points_sofb', self.nr_points_sofb)
        stg += dtmp('nr_iters_corr', self.nr_iters_corr)
        stg += dtmp('nr_svals', self.nr_svals)
        if self.corr_method == self.CORR_METHOD.QS:
            stg += ftmp('ks_amp_factor2apply', self.ks_amp_factor2apply, '')
            stg += '{0:25s} = ['.format('qs2use_idx')
            stg += ','.join(f'{idx:3d}' for idx in self.qs2use_idx) + ']'
        else:
            pass # mod print string
        return stg
    
    @property
    def corr_method(self):
        return self._corr_method

    @corr_method.setter
    def corr_method(self, method_str):
        if method_str == 'QS':
            self.params.corr_method =\
                  CorrectVerticalDispersionParams.CORR_METHOD.QS 
        elif method_str == 'CV':
            self.params.corr_method =\
                  CorrectVerticalDispersionParams.CORR_METHOD.CV
        else:
            raise ValueError('The method should be "QS" or "CV"')

class CorrectVerticalDispersion(_ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(target=self._meas_func, isonline=isonline)
        self.model = si.create_accelerator()
        self.fam_data = si.get_family_data(self.model)
        self.model_alpha = pyaccel.optics.get_mcf(self.model)
        self.model_ks_dispmat = None
        self.meas_ks_dispmat = None
        self.model_vk_dispmat = None # already calculated: "dispy_DRM.txt" (160 cv X 160 bpm)
        self.orm = OrbRespmat(model=self.model, acc='SI', dim='6d')
        self.qs_names = None # to avoid errors
        if self.isonline:
            self.devices['rfgen'] = RFGen(
                props2init=['GeneralFreq-SP', 'GeneralFreq-RB'])
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.create_skews() # check the corr_method first (inside func)

    def create_skews(self):
        """."""
        if self.params.corr_method != \
                CorrectVerticalDispersionParams.CORR_METHOD.QS:
            return
        qs_names = _np.array(self.fam_data['QS']['devnames'], dtype=object)
        self.qs_names = qs_names
        for name in self.qs_names:
            self.devices[name] = PowerSupply(name)

    def get_vertical_kicks(self): # return vertical kicks of sofb-CV
        sofb = self.devices['sofb']
        return _np.array(sofb.kickcv).ravel()

    def get_skew_strengths(self):
        """."""
        if self.params.corr_method != \
                CorrectVerticalDispersionParams.CORR_METHOD.QS:
            return None
        names = self.qs_names[self.params.qs2use_idx]
        stren = [self.devices[name].strength for name in names]
        return _np.array(stren)

    def apply_skew_strengths(self, strengths, factor=1):
        """."""
        if self.params.corr_method != \
                CorrectVerticalDispersionParams.CORR_METHOD.QS:
            return
        names = self.qs_names[self.params.qs2use_idx]
        for idx, name in enumerate(names):
            self.devices[name].strength = factor * strengths[idx]

    def apply_skew_delta_strengths(self, delta_strengths, factor=1):
        """."""
        if self.params.corr_method != \
                CorrectVerticalDispersionParams.CORR_METHOD.QS:
            return
        names = self.qs_names[self.params.qs2use_idx]
        for idx, name in enumerate(names):
            self.devices[name].strength += factor * delta_strengths[idx]

    def get_orb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer()
        return _np.r_[sofb.orbx, sofb.orby]

    def measure_dispersion(self):
        """."""
        rfgen, sofb = self.devices['rfgen'], self.devices['sofb']
        prms = self.params
        rf_freq0 = rfgen.frequency
        sofb.nr_points = prms.nr_points_sofb
        dfreq = prms.delta_rf_freq
        rfgen.set_frequency(rf_freq0 + dfreq/2)
        orbplus = self.get_orb()
        rfgen.set_frequency(rf_freq0 - dfreq/2)
        orbminus = self.get_orb()
        dorb_dfreq = (orbplus - orbminus)/dfreq
        disp = - self.model_alpha * rf_freq0 * dorb_dfreq * 1e-6
        rfgen.set_frequency(rf_freq0)
        return disp, rf_freq0

    def _meas_func(self):
        data = {}
        data['timestamp'] = _time.time()
        data['dispersion_history'] = []

        # CORR WITH QS
        data['dksl_history'] = []
        data['ksl_history'] = []
        data['initial_ksl'] = self.get_skew_strengths()

        # CORR WITH CV
        data['dvk_history'] = []
        data['vk_history'] = []
        data['initial_vk'] = self.get_vertical_kicks()

        data['rf_frequency'] = []
        data['tunex'] = []
        data['tuney'] = []

        # QS correction
        if self.params.corr_method == \
            CorrectVerticalDispersionParams.CORR_METHOD.QS:
            if self.model_ks_dispmat is None:
                all_qsidx_model = _np.array(self.fam_data['QS']['index']).ravel()
                subset_qsidx = all_qsidx_model[self.params.qs2use_idx]
                self.model_ks_dispmat = self.calc_ks_dispmat(
                    self.model, qsidx=subset_qsidx)[160:, :]
            idmat = self.invert_dispmat(self.model_ks_dispmat, self.params.nr_svals)
            for idx in range(self.params.nr_iters_corr):
                if self._stopevt.is_set():
                    print('Stop!')
                    break
                disp, rffreq = self.measure_dispersion()
                dispy = disp[160:]
                data['dispersion_history'].append(disp)
                data['rf_frequency'].append(rffreq)
                data['tunex'].append(self.devices['tune'].tunex)
                data['tuney'].append(self.devices['tune'].tuney)
                dstrens = self.calc_ks_correction(dispy, idmat)
                stg = f'Iter. {idx+1:02d}/{self.params.nr_iters_corr:02d} | '
                stg += f'dispy rms: {_np.std(dispy)*1e6:.2f} [um], '
                stg += f'dKsL rms: {_np.std(dstrens)*1e3:.4f} [1/km] \n'
                print(stg)
                print('-'*50)
                data['dksl_history'].append(dstrens)
                data['ksl_history'].append(self.get_skew_strengths())
                self.apply_skew_delta_strengths(
                    dstrens, factor=self.params.ks_amp_factor2apply)
                self.devices['sofb'].correct_orbit_manually(nr_iters=10, residue=5)

            dispf, rffreq = self.measure_dispersion()
            data['final_dispersion'] = dispf
            data['final_rf_frequency'] = rffreq
            data['final_ksl'] = self.get_skew_strengths()

            print('='*50)
            print('Correction result:')
            dispy0, dispyf = data['dispersion_history'][0][160:], dispf[160:]
            ksl0, kslf = data['initial_ksl'], data['final_ksl']
            stg = f'dispy rms {_np.std(dispy0)*1e6:.2f} [um] '
            stg += f'--> {_np.std(dispyf)*1e6:.2f} [um] \n'
            stg += f"KsL rms: {_np.std(ksl0)*1e3:.4f} [1/km] "
            stg += f'--> {_np.std(kslf)*1e3:.4f} [1/km] \n'
            print(stg)
            print('='*50)
            print('Finished!')
            self.data = data

        # CV correction
        elif self.params.corr_method == \
            CorrectVerticalDispersionParams.CORR_METHOD.CV: # in develop
                imat = self.invert_dispmat(self.model_vk_dispmat, self.params.nr_svals)
                for idx in range(self.params.nr_iters_corr):
                    if self._stopevt.is_set():
                        print('Stop!')
                        break
                    disp, rffreq = self.measure_dispersion()
                    dispy = disp[160:]
                    data['dispersion_history'].append(disp)
                    data['rf_frequency'].append(rffreq)
                    data['tunex'].append(self.devices['tune'].tunex)
                    data['tuney'].append(self.devices['tune'].tuney)
                    dvk = self.calc_cv_correction(dispy, imat)
                    stg = f'Iter. {idx+1:02d}/{self.params.nr_iters_corr:02d} | '
                    stg += f'dispy rms: {_np.std(dispy)*1e6:.2f} [um], '
                    stg += f'delta Vkicks rms: {_np.std(dvk)*1e3:.4f} [1/km] \n'
                    print(stg)
                    print('-'*50)
                    data['dvk_history'].append(dvk)
                    data['vk_history'].append(self.get_vertical_kicks())
                    
                    self.devices['sofb'].deltakickcv = dvk # sofb setter to CV delta vkicks

                    # i dont know the limits of operating the sofb with manual SP of delta kicks

                    ### to correct the horizontal orbit, 
                    ### must restrict the sofb manual correction to only CH:

                    #self.devices['sofb'].correct_orbit_manually(nr_iters=10, residue=5) # i didnt mod yet

                dispf, rffreqf = self.measure_dispersion()
                data['final_dispersion'] = dispf
                data['final_rf_frequency'] = rffreqf
                data['final_vk'] = self.get_vertical_kicks()

                print('='*50)
                print('Correction result:')
                dispy0, dispyf = data['dispersion_history'][0][160:], dispf[160:]
                vk0, vkf = data['initial_vk'], data['final_vk']
                stg = f'dispy rms {_np.std(dispy0)*1e6:.2f} [um] '
                stg += f'--> {_np.std(dispyf)*1e6:.2f} [um] \n'
                stg += f"KsL rms: {_np.std(vk0)*1e3:.4f} [1/km] "
                stg += f'--> {_np.std(vkf)*1e3:.4f} [1/km] \n' 
                print(stg)
                print('='*50)
                print('Finished!')
                self.data = data

    def calc_model_dispersion(self, model=None):
        """."""
        if model is None:
            model = self.model
        self.orm.model = model
        respmat = self.orm.get_respm()
        rf_freq = self.orm.model[self.orm.rf_idx[0]].frequency
        alpha = pyaccel.optics.get_mcf(model)
        return - alpha * rf_freq * respmat[:, -1]

    def calc_ks_dispmat(self, qsidx=None, dksl=1e-6):
        """."""
        print('--- calculating dispersion/KsL matrix')
        if qsidx is None:
            qsidx = _np.array(self.fam_data['QS']['index']).ravel()
        dispmat = []
        for idx, qs in enumerate(qsidx):
            print(f'{idx+1:02d}/{qsidx.size:02d} ')
            modt = self.model[:]
            modt[qs].KsL += dksl/2
            dispp = self.calc_model_dispersion(modt)
            modt[qs].KsL -= dksl
            dispn = self.calc_model_dispersion(modt)
            dispmat.append((dispp-dispn)/dksl)
            modt[qs].KsL += dksl
        dispmat = _np.array(dispmat).T
        self.model_ks_dispmat = dispmat
        return dispmat

    @staticmethod
    def invert_dispmat(dispmat, nr_svals=None):
        """."""
        umat, smat, vhmat = _np.linalg.svd(dispmat, full_matrices=False)
        ismat = 1/smat
        if nr_svals is not None:
            ismat[nr_svals:] = 0
        inv_dispmat = vhmat.T @ _np.diag(ismat) @ umat.T
        return inv_dispmat

    def calc_ks_correction(self, dispy, inv_dispmat=None, nr_svals=None):
        """."""
        if inv_dispmat is None:
            dispmat = self.calc_ks_dispmat(self.model)
            inv_dispmat = self.invert_dispmat(dispmat, nr_svals)
        dstren = - inv_dispmat @ dispy
        return dstren
    
    def calc_cv_correction(self, dispy, inv_dispmat=None, nr_svals=None):
        """."""
        if inv_dispmat is None:
            dispmat = self.model_vk_dispmat
            inv_dispmat = self.invert_dispmat(dispmat, nr_svals)
        dvk = (-1) * _np.dot(inv_dispmat, dispy)
        return dvk

    def measure_ks_dispmat(self, dksl=1e-4):
        """."""
        meas_ks_dispmat = []
        names = self.qs_names[self.params.qs2use_idx]
        print('Measuring respmat: dispersion/KsL skew')
        for name in names:
            print(f'{name:s}')
            qsmag = self.devices[name]
            stren0 = qsmag.strength
            qsmag.set_strength(stren0 + dksl/2)
            dispp, _ = self.measure_dispersion()
            qsmag.set_strength(stren0 - dksl/2)
            dispm, _ = self.measure_dispersion()
            meas_ks_dispmat.append((dispp-dispm)/dksl)
            qsmag.set_strength(stren0)
        meas_ks_dispmat = _np.array(meas_ks_dispmat).T
        print('Finished!')
        self.meas_ks_dispmat = meas_ks_dispmat
        return meas_ks_dispmat



# functions i used to obtain the DRM (with CV)

# def calc_vdisp(model, indices):
#     disp = calc_disp(model=model, indices=indices)
#     return disp[int(len(disp)/2):]

# def calc_hdisp(model, indices):
#     disp = calc_disp(model=model, indices=indices)
#     return disp[:int(len(disp)/2)]

# def calc_disp(model, indices):
#     orbp = pa.tracking.find_orbit4(model, indices=indices, energy_offset=+5e-6) 
#     orbn = pa.tracking.find_orbit4(model, indices=indices, energy_offset=-5e-6)
#     return np.hstack([(orbp[0,:] - orbn[0,:])/(2e-6), (orbp[2,:] - orbn[2,:])/(2e-6)])

# def getline(model, idx, delta, bpmidx):
#     deltak = delta/2 
#     v0 = model[idx].vkick_polynom
#     model[idx].vkick_polynom = v0 + (deltak)
#     dispp = calc_vdisp(model,'open')[bpmidx]
#     model[idx].vkick_polynom = v0 - (deltak)
#     dispn = calc_vdisp(model,'open')[bpmidx]
#     model[idx].vkick_polynom = v0
#     disp = (dispp - dispn)/(2*deltak)
#     return disp

# def calc_disp_respmat(model, famdata, delta):
#     cvidx = np.array(famdata['CV']['index']).ravel()
#     bpmidx = np.array(famdata['BPM']['index']).ravel()
#     lines=[]
#     # for i in chidx:
#     #     lines.append(getline(model, i, 'CH', delta))
#     for i in cvidx:
#         lines.append(getline(model, i, delta, bpmidx))
#     lines = np.array(lines).T
#     return lines