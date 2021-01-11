"""."""

import numpy as np

import pyaccel
from pymodels import si
from apsuite.optics_analysis.tune_correction import TuneCorr

class IDParams():
    """."""

    ID_MODELS_FOLDER = '/home/facs/repos/MatlabMiddleLayer/Release' + \
        '/lnls/fac_scripts/sirius/insertion_devices/id_modelling/'

    BEAMLINES_NAMES = [
        'CARNAUBA',
        'CATERETE',
        'EMA',
        'IPE',
        'MANACA']

    def __init__(self, id_name=None, phase=0):
        """."""
        self.id_name = id_name
        self.phase = phase
        self.kicktable_fname = None
        self.section_nr = None
        self.straight_label = None
        self.seg_nr = None
        if id_name == 'MANACA':
            self.params_manaca(phase)
        elif id_name == 'CARNAUBA':
            self.params_carnauba(phase)
        elif id_name == 'CATERETE':
            self.params_caterete(phase)
        elif id_name == 'EMA':
            self.params_ema(phase)
        elif id_name == 'IPE':
            self.params_ipe(phase)

    def __str__(self):
        """."""
        dtmp = '{0:20s}: {1:2d}  {2:s}\n'.format
        stmp = '{0:20s}: {1:s}  {2:s}\n'.format
        stg = stmp('ID Name ', self.id_name, '')
        stg += dtmp('Section ', self.section_nr, '')
        stg += stmp('Straight section ', self.straight_label, '')
        stg += dtmp('Number of segments ', self.seg_nr, '')
        stg += stmp('Kicktable filename ', self.kicktable_fname, '')
        return stg

    def params_carnauba(self, phase=0):
        """."""
        self.id_name = 'CARNAUBA'
        self.kicktable_fname = IDParams.ID_MODELS_FOLDER + \
            'APU22/APUKyma22mm_kickmap_shift_' + str(phase) + '.txt'
        self.section_nr = 6
        self.straight_label = 'B'
        self.seg_nr = 20

    def params_caterete(self, phase=0):
        """."""
        self.id_name = 'CATERETE'
        self.kicktable_fname = IDParams.ID_MODELS_FOLDER + \
            'APU22/APUKyma22mm_kickmap_shift_' + str(phase) + '.txt'
        self.section_nr = 7
        self.straight_label = 'P'
        self.seg_nr = 20

    def params_ema(self, phase=0):
        """."""
        self.id_name = 'EMA'
        self.kicktable_fname = IDParams.ID_MODELS_FOLDER + \
            'APU22/APUKyma22mm_kickmap_shift_' + str(phase) + '.txt'
        self.section_nr = 8
        self.straight_label = 'B'
        self.seg_nr = 20

    def params_manaca(self, phase=0):
        """."""
        self.id_name = 'MANACA'
        self.kicktable_fname = IDParams.ID_MODELS_FOLDER + \
            'APU22/APUKyma22mm_kickmap_shift_' + str(phase) + '.txt'
        self.section_nr = 9
        self.straight_label = 'A'
        self.seg_nr = 20

    def params_ipe(self, phase=0):
        """."""
        self.id_name = 'IPE'
        self.kicktable_fname = IDParams.ID_MODELS_FOLDER + \
            'APU58/APUKyma58mm_kickmap_shift_' + str(phase) + '.txt'
        self.section_nr = 11
        self.straight_label = 'P'
        self.seg_nr = 20


class ID():
    """."""

    DEFAULT_KNOBS = [
        'QFA', 'QDA',
        'QFB', 'QDB1', 'QDB2',
        'QFP', 'QDP1', 'QDP2']
    DEFAULT_DELTAKL = 1e-6

    def __init__(self, id_data):
        """."""
        self.id_data = id_data
        # self.bare_model = si.lattice.create_lattice()
        self.bare_model = si.create_accelerator()
        self.fam_data = si.get_family_data(self.bare_model)
        if self.id_data.straight_label == 'A':
            self.id_data.straight_center = 'mia'
        elif self.id_data.straight_label == 'B':
            self.id_data.straight_center = 'mib'
        elif self.id_data.straight_label == 'P':
            self.id_data.straight_center = 'mip'
        else:
            raise ValueError('Non-existent straight section label')

    def insert_ids(self):
        """."""
        if 'APU22' in self.id_data.kicktable_fname:
            idfam = self.fam_data['APU22']
        elif 'APU58' in self.id_data.kicktable_fname:
            idfam = self.fam_data['APU58']

        for sub in idfam['subsection']:
            if str(self.id_data.section_nr) in sub:
                idx = idfam['subsection'].index(sub)
                break

        lat_idx = idfam['index'][idx]
        id_half = self.bare_model[lat_idx[0]]

        id_kickmap = pyaccel.elements.kickmap(
            fam_name=self.id_data.id_name,
            kicktable_fname=self.id_data.kicktable_fname,
            nr_steps=40)

        kickmap_half_len = id_kickmap.length/2
        rescale = id_half.length/kickmap_half_len

        id_half_kickmap = pyaccel.elements.kickmap(
            fam_name=self.id_data.id_name,
            kicktable_fname=self.id_data.kicktable_fname,
            nr_steps=40,
            rescale_length=0.5*rescale,
            rescale_kicks=0.5*rescale)

        mod_id = self.bare_model[:]
        mod_id[lat_idx[0]] = id_half_kickmap
        mod_id[lat_idx[1]] = id_half_kickmap
        return mod_id

    def fix_tunes(self, model):
        """."""
        tunecorr = TuneCorr(
            model, 'SI', method='Proportional', grouping='TwoKnobs')
        tunes0 = tunecorr.get_tunes(self.bare_model)
        print('    tunes init  : ', tunecorr.get_tunes(model))
        tunemat = tunecorr.calc_jacobian_matrix()
        tunecorr.correct_parameters(
            model=model,
            goal_parameters=tunes0,
            jacobian_matrix=tunemat)
        print('    tunes final : ', tunecorr.get_tunes(model))
        return model

    def symmetrize_straight_section(
            self, model, factor=1, max_niter=20, tol=1e-6):
        """."""
        mcidx = pyaccel.lattice.find_indices(model, 'fam_name', 'mc')
        model = pyaccel.lattice.shift(model, mcidx[-1])

        mcidx = pyaccel.lattice.find_indices(model, 'fam_name', 'mc')
        mcidx = np.unique([0] + mcidx + [len(model)])
        sec_nr = self.id_data.section_nr
        line_idx = np.arange(mcidx[sec_nr-1], mcidx[sec_nr]+1)
        sline = model[line_idx]
        knobs = ID.get_knob_idx(sline, ID.DEFAULT_KNOBS)

        data = dict()
        data['symmetry_point'] = pyaccel.lattice.find_indices(
            sline, 'fam_name', self.id_data.straight_center)[0]
        data['initial_point'] = line_idx[0]
        data['final_point'] = line_idx[-1]
        data['twiss0'], _ = pyaccel.optics.calc_twiss(model)

        res_vec = ID.local_residue(sline, data)
        res = ID.calc_rms(res_vec)
        jmat = ID.calc_jacobian_matrix(sline, knobs, data)
        ijmat = ID.calc_inverse_jacobian(jmat)
        dkl = np.zeros(knobs.size)

        nr_iters = 0
        print('initial residue {}:'.format(res))
        while res > tol and nr_iters < max_niter and factor > 1e-3:
            print('residue {}:'.format(res))
            dk_temp = -1 * ijmat @ res_vec
            dk_temp *= factor
            ID.set_dkl(sline, dk_temp, knobs)
            newres_vec = ID.local_residue(sline, data)
            newres = ID.calc_rms(newres_vec)
            if newres < res:
                res = newres
                res_vec = newres_vec
                jmat = ID.calc_jacobian_matrix(sline, knobs, data)
                ijmat = ID.calc_inverse_jacobian(jmat)
                dkl += dk_temp
                factor = np.min([1, 2*factor])
            else:
                ID.set_dkl(sline, -dk_temp, knobs)
                factor /= 2
            nr_iters += 1
        print('final residue {}:'.format(res))
        model[line_idx] = sline

        start = pyaccel.lattice.find_indices(model, 'fam_name', 'start')
        model = pyaccel.lattice.shift(model, start[-1])
        return model

    # static methods
    @staticmethod
    def get_knob_idx(sline, knobs_names):
        """."""
        knobs_idx = []
        for knb_nam in knobs_names:
            idx = np.array(
                pyaccel.lattice.find_indices(
                    sline, 'fam_name', knb_nam)).ravel()
            if idx.size:
                knobs_idx.append(idx)
        return np.array(knobs_idx).ravel()

    @staticmethod
    def set_dkl(model, dkl_value, kl_idx):
        """."""
        for kli_num, kli in enumerate(kl_idx):
            model[kli].KL += dkl_value[kli_num]

    @staticmethod
    def calc_rms(residue):
        """."""
        return np.sqrt(np.mean(residue*residue))

    @staticmethod
    def local_residue(sline, data, include_tune=False):
        """."""
        symm_point = data['symmetry_point']
        ini_point = data['initial_point']
        final_point = data['final_point']
        twiss0 = data['twiss0']

        betax0 = twiss0.betax[ini_point]
        betay0 = twiss0.betay[ini_point]
        alphax0 = twiss0.alphax[ini_point]
        alphay0 = twiss0.alphay[ini_point]

        etax0 = twiss0.etax[ini_point]
        etaxl0 = twiss0.etapx[ini_point]

        mux0 = twiss0.mux[final_point] - twiss0.mux[ini_point]
        mux0 /= 2
        muy0 = twiss0.muy[final_point] - twiss0.muy[ini_point]
        muy0 /= 2

        scales = dict()
        scales['alpha'] = 1e-5
        scales['mu'] = 1e-3

        input_twiss = pyaccel.optics.Twiss.make_new(
            beta=[betax0, betay0],
            alpha=[alphax0, alphay0],
            etax=[etax0, etaxl0])

        twiss, *_ = pyaccel.optics.calc_twiss(
            sline, init_twiss=input_twiss, indices='open')

        alphax = twiss[symm_point].alphax
        alphay = twiss[symm_point].alphay
        dmux = twiss[symm_point].mux - mux0
        dmuy = twiss[symm_point].muy - muy0

        res = [alphax/scales['alpha'], alphay/scales['alpha']]
        if include_tune:
            res.append(dmux/scales['mu'])
            res.append(dmuy/scales['mu'])
        return np.array(res)

    @staticmethod
    def calc_jacobian_matrix(model, knobs, data):
        """."""
        res0 = ID.local_residue(model, data)
        mat = np.zeros((res0.size, knobs.size))
        for knb_num, knb in enumerate(knobs):
            kl_orig = model[knb].KL
            model[knb].KL = kl_orig + ID.DEFAULT_DELTAKL/2
            res_up = ID.local_residue(model, data)
            model[knb].KL = kl_orig - ID.DEFAULT_DELTAKL/2
            res_down = ID.local_residue(model, data)
            mat[:, knb_num] = (res_up - res_down)/ID.DEFAULT_DELTAKL
            model[knb].KL = kl_orig
        return mat

    @staticmethod
    def calc_inverse_jacobian(jmat):
        """."""
        umat, smat, vhmat = np.linalg.svd(jmat, full_matrices=False)
        ismat = 1/smat
        ismat[np.isnan(ismat)] = 0
        ismat[np.isinf(ismat)] = 0
        ismat = np.diag(ismat)
        ijmat = np.dot(np.dot(vhmat.T, ismat), umat.T)
        return ijmat


# def insert_ids(self):
    #     """."""
    #     mod = self.bare_model[:]
    #     mcidx = pyaccel.lattice.find_indices(mod, 'fam_name', 'mc')
    #     mod = pyaccel.lattice.shift(mod, mcidx[-1])

    #     ssidx = pyaccel.lattice.find_indices(mod, 'fam_name', 'id_enda')
    #     ssidx += pyaccel.lattice.find_indices(mod, 'fam_name', 'id_endb')
    #     ssidx += pyaccel.lattice.find_indices(mod, 'fam_name', 'id_endp')
    #     # del mod[ssidx]
    #     for ssi in sorted(ssidx, reverse=True):
    #         del mod[ssi]

    #     mcidx = pyaccel.lattice.find_indices(mod, 'fam_name', 'mc')
    #     section_nr = self.id_data.section_nr

    #     mc = np.unique([0] + mcidx + [len(mod)])
    #     elem = list(range(mc[section_nr-1], mc[section_nr]+1))
    #     center_idx = pyaccel.lattice.find_indices(
    #         mod[elem[0]:elem[-1]], 'fam_name', self.id_data.straight_center)
    #     center_idx = elem[center_idx[0]]
    #     mod = self.insert_kicktable(mod, center_idx)
    #     return mod

    # def insert_kicktable(self, model, center_idx):
    #     """."""
    #     id_kickmap = pyaccel.elements.kickmap(
    #         fam_name=self.id_data.id_name,
    #         kicktable_fname=self.id_data.kicktable_filename,
    #         nr_steps=40)

    #     print(id_kickmap.length)
    #     idx_dws = center_idx + 1
    #     while model[idx_dws].pass_method == 'drift_pass':
    #         idx_dws += 1
    #     idx_dws -= 1

    #     idx_ups = center_idx - 1
    #     while model[idx_ups].pass_method == 'drift_pass':
    #         idx_ups -= 1
    #     idx_ups += 1

    #     range_ups = np.arange(idx_ups, center_idx+1)
    #     range_dws = np.arange(center_idx, idx_dws+1)

    #     lens_ups = pyaccel.lattice.get_attribute(model, 'length', range_ups)
    #     lens_dws = pyaccel.lattice.get_attribute(model, 'length', range_dws)

    #     ups_drift = model[idx_ups]
    #     dws_drift = model[idx_dws]
    #     ups_drift.pass_method = 'drift_pass'
    #     dws_drift.pass_method = 'drift_pass'

    #     ups_drift.length = sum(lens_ups) - id_kickmap.length/2
    #     dws_drift.length = sum(lens_dws) - id_kickmap.length/2

    #     id_kickmap.length /= 2
    #     if ups_drift.length < 0 or dws_drift.length < 0:
    #         raise Exception(
    #             'there is no space to insert id within the defined location!')

    #     model_id = model[:idx_ups]
    #     model_id.append(ups_drift)
    #     model_id.append(id_kickmap)
    #     model_id += model[center_idx]
    #     model_id.append(id_kickmap)
    #     model_id.append(dws_drift)
    #     model_id += model[idx_dws+1:]
    #     idx = pyaccel.lattice.find_indices(model_id, 'fam_name', 'start')
    #     model_id = pyaccel.lattice.shift(model_id, idx[0])
    #     return model_id
