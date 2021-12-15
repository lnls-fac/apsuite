"""."""

from copy import deepcopy as _dcopy
import multiprocessing as mp_

import numpy as _np

from mathphys.functions import save_pickle as _save_pickle, \
    load_pickle as _load_pickle
import pyaccel as _pyaccel


class LOCOUtils:
    """LOCO utils."""

    @staticmethod
    def save_data(fname, jlocodict, overwrite=True):
        """."""
        _save_pickle(jlocodict, fname, overwrite=overwrite)

    @staticmethod
    def load_data(fname):
        """."""
        return _load_pickle(fname)

    @staticmethod
    def get_idx(indcs):
        """."""
        return _np.array([idx[0] for idx in indcs])

    @staticmethod
    def respm_calc(model, respm, use_dispersion):
        """."""
        respm.model = _dcopy(model)
        matrix = respm.get_respm()
        if not use_dispersion:
            matrix[:, -1] *= 0
        return matrix

    @staticmethod
    def apply_bpm_gain(matrix, gain):
        """."""
        return gain[:, None] * matrix

    @staticmethod
    def apply_bpm_roll(matrix, roll):
        """."""
        cos_mat = _np.diag(_np.cos(roll))
        sin_mat = _np.diag(_np.sin(roll))
        r_alpha = _np.hstack((cos_mat, sin_mat))
        r_alpha = _np.vstack((r_alpha, _np.hstack((-sin_mat, cos_mat))))
        return _np.dot(r_alpha, matrix)

    @staticmethod
    def apply_corr_gain(matrix, gain):
        """."""
        matrix[:, :-1] *= gain[None, :]
        return matrix

    @staticmethod
    def apply_all_gain(matrix, gain_bpm, roll_bpm, gain_corr):
        """."""
        matrix = LOCOUtils.apply_bpm_gain(matrix, gain_bpm)
        matrix = LOCOUtils.apply_bpm_roll(matrix, roll_bpm)
        matrix = LOCOUtils.apply_corr_gain(matrix, gain_corr)
        return matrix

    @staticmethod
    def apply_bpm_weight(matrix, weight_bpm):
        """."""
        return weight_bpm * matrix

    @staticmethod
    def apply_corr_weight(matrix, weight_corr):
        """."""
        return matrix * weight_corr[None, :]

    @staticmethod
    def apply_all_weight(matrix, weight_bpm, weight_corr):
        """."""
        matrix = LOCOUtils.apply_bpm_weight(matrix, weight_bpm)
        matrix = LOCOUtils.apply_corr_weight(matrix, weight_corr)
        return matrix

    @staticmethod
    def remove_offdiagonal(matrix_in, nr_bpm, nr_ch, nr_cv):
        """."""
        matrix_out = _np.zeros(matrix_in.shape)
        matrix_out[:nr_bpm, :nr_ch] = matrix_in[:nr_bpm, :nr_ch]
        matrix_out[nr_bpm:, nr_ch:nr_ch+nr_cv] = \
            matrix_in[nr_bpm:, nr_ch:nr_ch+nr_cv]
        matrix_out[:nr_bpm, -1] = matrix_in[:nr_bpm, -1]
        return matrix_out

    @staticmethod
    def remove_diagonal(matrix_in, nr_bpm, nr_ch):
        """."""
        matrix_out = _np.zeros(matrix_in.shape)
        matrix_out[:nr_bpm, nr_ch:-1] = matrix_in[:nr_bpm, nr_ch:-1]
        matrix_out[nr_bpm:, :nr_ch] = matrix_in[nr_bpm:, :nr_ch]
        matrix_out[nr_bpm:, -1] = matrix_in[nr_bpm:, -1]
        return matrix_out

    @staticmethod
    def add_dispersion_to_respm(matrix, energy_shift, dispersion):
        """."""
        matrix_out = _dcopy(matrix)
        matrix_out[:, :-1] += dispersion[:, None] * energy_shift[None, :]
        return matrix_out

    @staticmethod
    def get_quads_strengths(model, indices):
        """."""
        kquads = []
        for qidx in indices:
            kquads.append(_pyaccel.lattice.get_attribute(
                model, 'KL', qidx))
        return _np.array(kquads)

    @staticmethod
    def set_quadmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        for idx, idx_seg in enumerate(idx_mag):
            _pyaccel.lattice.set_attribute(
                model, 'KL', idx_seg, kvalues[idx] + kdelta/len(idx_mag))

    @staticmethod
    def set_quadset_kdelta(model, idx_set, kvalues, kdelta):
        """."""
        for idx, idx_mag in enumerate(idx_set):
            LOCOUtils.set_quadmag_kdelta(
                model, idx_mag, kvalues[idx], kdelta)

    @staticmethod
    def set_dipmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        ktotal = _np.sum(kvalues)
        if ktotal:
            newk = [kval*(1+kdelta/ktotal) for kval in kvalues]
            _pyaccel.lattice.set_attribute(
                model, 'KL', idx_mag, newk)
        else:
            newk = [kval + kdelta/len(idx_mag) for kval in kvalues]
            _pyaccel.lattice.set_attribute(
                model, 'KL', idx_mag, kvalues + kdelta/len(idx_mag))

    @staticmethod
    def set_dipset_kdelta(model, idx_set, kvalues, kdelta):
        """."""
        for idx, idx_mag in enumerate(idx_set):
            LOCOUtils.set_dipmag_kdelta(
                model, idx_mag, kvalues[idx], kdelta)

    @staticmethod
    def set_quadmag_ksdelta(model, idx_mag, ksvalues, ksdelta):
        """."""
        _pyaccel.lattice.set_attribute(
            model, 'KsL', idx_mag, ksvalues + ksdelta)

    @staticmethod
    def set_dipmag_ksdelta(model, idx_mag, ksvalues, ksdelta):
        """."""
        kstotal = _np.sum(ksvalues)
        if kstotal:
            newks = [ksval*(1+ksdelta/kstotal) for ksval in ksvalues]
            _pyaccel.lattice.set_attribute(
                model, 'KsL', idx_mag, newks)
        else:
            newks = [ksval + ksdelta/len(idx_mag) for ksval in ksvalues]
            _pyaccel.lattice.set_attribute(
                model, 'KsL', idx_mag, newks)

    @staticmethod
    def set_dipmag_kick(model, idx_mag, kick_values, kick_delta):
        """."""
        angle = _np.array(
            _pyaccel.lattice.get_attribute(model, 'angle', idx_mag))
        angle /= _np.sum(angle)
        _pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', idx_mag, kick_values + kick_delta * angle)

    @staticmethod
    def set_girders_long_shift(model, girders, ds_shift):
        """."""
        for i, inds in enumerate(girders):
            if ds_shift[i]:
                model[inds[0]-1].length += ds_shift[i]
                model[inds[1]+1].length -= ds_shift[i]
        return model

    @staticmethod
    def jloco_calc_linear(config, matrix):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        ncorr = nch + ncv
        shape0 = matrix.shape[0]
        shape1 = matrix.shape[1]

        if shape0 != 2*nbpm:
            raise Exception('Problem with BPM number in matrix')
        if shape1 not in (ncorr, ncorr + 1):
            raise Exception('Problem with correctors number in matrix')

        if shape1 < ncorr + 1 and config.use_dispersion:
            raise Exception('There is no dispersion line in the matrix')

        if config.gain_bpm is not None:
            g_bpm = config.gain_bpm
        else:
            g_bpm = _np.ones(2*nbpm)
        if config.roll_bpm is not None:
            alpha_bpm = config.roll_bpm
        else:
            alpha_bpm = _np.zeros(nbpm)
        cos_mat = _np.diag(_np.cos(alpha_bpm))
        sin_mat = _np.diag(_np.sin(alpha_bpm))

        r_alpha = _np.hstack((cos_mat, sin_mat))
        r_alpha = _np.vstack((r_alpha, _np.hstack((-sin_mat, cos_mat))))

        dr_alpha = _np.hstack((-sin_mat, cos_mat))
        dr_alpha = _np.vstack((dr_alpha, _np.hstack((-cos_mat, sin_mat))))

        dmdg_bpm = _np.zeros((shape0*shape1, 2*nbpm))
        for num in range(shape0):
            kron = LOCOUtils.kronecker(num, num, shape0)
            dbmat = _np.dot(r_alpha, kron)
            dmdg_bpm[:, num] = _np.dot(dbmat, matrix).ravel()

        dmdalpha_bpm = _np.zeros((shape0*shape1, nbpm))
        for idx in range(shape0//2):
            kron = LOCOUtils.kronecker(idx, idx, shape0//2)
            kron = _np.tile(kron, (2, 2))
            drmat = _np.dot(kron, dr_alpha)
            dbmat = drmat * g_bpm[:, None]
            dmdalpha_bpm[:, idx] = _np.dot(dbmat, matrix).ravel()

        dmdg_corr = _np.zeros((shape0*shape1, ncorr))
        for idx in range(ncorr):
            kron = LOCOUtils.kronecker(idx, idx, shape1)
            dmdg_corr[:, idx] = _np.dot(matrix, kron).ravel()
        return dmdg_bpm, dmdalpha_bpm, dmdg_corr

    @staticmethod
    def _parallel_base(config, model, indices, func, magtype=None):
        if not config.parallel:
            mat = func(config, model, indices)
        else:
            slcs = LOCOUtils._get_slices_multiprocessing(
                config.parallel, len(indices))
            with mp_.Pool(processes=len(slcs)) as pool:
                res = []
                for slc in slcs:
                    res.append(pool.apply_async(
                        func,
                        (config, model, indices[slc], magtype)))
                mat = [re.get() for re in res]
            mat = _np.concatenate(mat, axis=1)
        return mat

    @staticmethod
    def _get_slices_multiprocessing(parallel, npart):
        nrproc = mp_.cpu_count() - 3
        nrproc = nrproc if parallel is True else parallel
        nrproc = max(nrproc, 1)
        nrproc = min(nrproc, npart)

        np_proc = (npart // nrproc)*_np.ones(nrproc, dtype=int)
        np_proc[:(npart % nrproc)] += 1
        parts_proc = _np.r_[0, _np.cumsum(np_proc)]
        return [slice(parts_proc[i], parts_proc[i+1]) for i in range(nrproc)]

    @staticmethod
    def jloco_calc_k_dip(config, model):
        """."""
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        if config.use_dip_families:
            dip_indices = []
            for fam_name in config.famname_dipset:
                dip_indices.append(config.respm.fam_data[fam_name]['index'])
        else:
            dip_indices = config.respm.fam_data['BN']['index']
        magtype = 'dipole'
        dip_matrix = LOCOUtils._parallel_base(
            config, model, dip_indices,
            LOCOUtils._jloco_calc_k_matrix, magtype)
        return dip_matrix

    @staticmethod
    def jloco_calc_k_quad(config, model):
        """."""
        if config.use_quad_families:
            kindices = []
            for fam_name in config.famname_quadset:
                kindices.append(config.respm.fam_data[fam_name]['index'])
        else:
            kindices = config.respm.fam_data['QN']['index']
        magtype = 'quadrupole'
        kmatrix = LOCOUtils._parallel_base(
            config, model, kindices,
            LOCOUtils._jloco_calc_k_matrix, magtype)
        return kmatrix

    @staticmethod
    def jloco_calc_k_sext(config, model):
        """."""
        if config.use_sext_families:
            sindices = []
            for fam_name in config.famname_sextset:
                sindices.append(config.respm.fam_data[fam_name]['index'])
        else:
            sindices = config.respm.fam_data['SN']['index']
        magtype = 'sextupole'
        smatrix = LOCOUtils._parallel_base(
            config, model, sindices,
            LOCOUtils._jloco_calc_k_matrix, magtype)
        return smatrix

    @staticmethod
    def _jloco_calc_k_matrix(config, model, indices, magtype=None):
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        famtype = None
        if magtype == 'quadrupole':
            famtype = config.use_quad_families
        elif magtype == 'sextupole':
            famtype = config.use_sext_families
        elif magtype == 'dipole':
            famtype = config.use_dip_families

        if famtype:
            kvalues = LOCOUtils.get_quads_strengths(
                model, indices)
            if magtype == 'dipole':
                set_quad_kdelta = LOCOUtils.set_dipset_kdelta
            else:
                set_quad_kdelta = LOCOUtils.set_quadset_kdelta
        else:
            kvalues = _np.array(
                _pyaccel.lattice.get_attribute(model, 'KL', indices))
            if magtype == 'dipole':
                set_quad_kdelta = LOCOUtils.set_dipmag_kdelta
            else:
                set_quad_kdelta = LOCOUtils.set_quadmag_kdelta

        kmatrix = _np.zeros((matrix_nominal.size, len(indices)))
        model_this = _dcopy(model)
        for idx, idx_set in enumerate(indices):
            set_quad_kdelta(
                model_this, idx_set,
                kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            kmatrix[:, idx] = dmatrix.ravel()
            set_quad_kdelta(model_this, idx_set, kvalues[idx], 0)
        return kmatrix

    @staticmethod
    def _jloco_calc_ks_matrix(config, model, indices, magtype=None):
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        ksvalues = _np.array(
            _pyaccel.lattice.get_attribute(model, 'KsL', indices))

        if magtype == 'dipole':
            set_quad_ksdelta = LOCOUtils.set_dipmag_ksdelta
        else:
            set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta

        ksmatrix = _np.zeros((matrix_nominal.size, len(indices)))
        model_this = _dcopy(model)
        for idx, idx_set in enumerate(indices):
            set_quad_ksdelta(
                model_this, idx_set, ksvalues[idx], config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            ksmatrix[:, idx] = dmatrix.ravel()
            set_quad_ksdelta(model_this, idx_set, ksvalues[idx], 0)
        return ksmatrix

    @staticmethod
    def jloco_calc_ks_dipoles(config, model):
        """."""
        ksindices = config.respm.fam_data['BN']['index']

        ksmatrix = LOCOUtils._parallel_base(
            config, model, ksindices,
            LOCOUtils._jloco_calc_ks_matrix, magtype='dipole')
        return ksmatrix

    @staticmethod
    def jloco_calc_ks_quad(config, model):
        """."""
        ksindices = config.respm.fam_data['QN']['index']

        ksmatrix = LOCOUtils._parallel_base(
            config, model, ksindices,
            LOCOUtils._jloco_calc_ks_matrix)
        return ksmatrix

    @staticmethod
    def jloco_calc_ks_skewquad(config, model):
        """."""
        config.update_skew_quad_knobs()
        ksindices = config.skew_quad_indices
        ksmatrix = LOCOUtils._parallel_base(
            config, model, ksindices,
            LOCOUtils._jloco_calc_ks_matrix)
        return ksmatrix

    @staticmethod
    def jloco_calc_ks_sextupoles(config, model):
        """."""
        ksindices = config.respm.fam_data['SN']['index']
        ksmatrix = LOCOUtils._parallel_base(
            config, model, ksindices,
            LOCOUtils._jloco_calc_ks_matrix)
        return ksmatrix

    @staticmethod
    def jloco_calc_kick_dipoles(config, model):
        """."""
        dip_indices = config.respm.fam_data['BN']['index']
        kick_matrix = LOCOUtils._parallel_base(
            config, model, ksindices,
            LOCOUtils._jloco_calc_kick_dip)
        return kick_matrix

    @staticmethod
    def _jloco_calc_kick_dip(config, model, dip_indices, magtype=None):
        dip_kick_values = _np.array(_pyaccel.lattice.get_attribute(
            model, 'hkick_polynom', dip_indices))
        set_dip_kick = LOCOUtils.set_dipmag_kick
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        dip_kick_matrix = _np.zeros((matrix_nominal.size, 1))
        delta_kick = config.DEFAULT_DELTA_DIP_KICK

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_dip_kick(
                model_this, idx_set,
                dip_kick_values[idx], delta_kick)
            nmags = len(idx_set)
        matrix_this = LOCOUtils.respm_calc(
            model_this, config.respm, config.use_dispersion)
        dmatrix = (matrix_this - matrix_nominal) / delta_kick / nmags
        dip_kick_matrix[:, 0] = dmatrix.ravel()

        for idx, idx_set in enumerate(dip_indices):
            set_dip_kick(model_this, idx_set, dip_kick_values[idx], 0)
        return dip_kick_matrix

    @staticmethod
    def jloco_calc_energy_shift(config, model):
        """."""
        matrix0 = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)
        energy_shift = _np.zeros(config.nr_corr + 1)
        dm_energy_shift = _np.zeros((matrix0.size, config.nr_corr))
        for cnum in range(config.nr_corr):
            energy_shift[cnum] = 1
            matrix_shift = config.measured_dispersion[:, None] * \
                energy_shift[None, :]
            dm_energy_shift[:, cnum] = matrix_shift.ravel()
            energy_shift[cnum] = 0
        return dm_energy_shift

    @staticmethod
    def jloco_calc_girders(config, model):
        """."""
        gindices = config.gir_indices

        gir_matrix = LOCOUtils._parallel_base(
            config, model, gindices,
            LOCOUtils._jloco_girders_shift)
        return gir_matrix

    @staticmethod
    def _jloco_girders_shift(config, model, gindices, magtype=None):
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)
        gmatrix = _np.zeros((matrix_nominal.size, len(gindices)))

        model_this = _dcopy(model)
        ds_shift = _np.zeros(gindices.shape[0])
        for idx, _ in enumerate(gindices):
            ds_shift[idx] = config.DEFAULT_GIRDER_SHIFT
            LOCOUtils.set_girders_long_shift(
                model_this, gindices, ds_shift)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)
            dmatrix /= config.DEFAULT_GIRDER_SHIFT
            gmatrix[:, idx] = dmatrix.ravel()
            ds_shift[idx] = 0
            LOCOUtils.set_girders_long_shift(
                model_this, gindices, ds_shift)
        return gmatrix

    @staticmethod
    def jloco_merge_linear(
            config, km_quad, km_sext, km_dip,
            ksm_quad, ksm_sext, ksm_dip,
            dmdg_bpm, dmdalpha_bpm, dmdg_corr,
            kick_dip, energy_shift, ks_skewquad,
            girder_shift):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        knobs_k = 0
        knobs_ks = 0
        knobs_linear = 0
        knobs_skewquad = 0
        knobs_gir = 0

        if km_quad is not None:
            knobs_k += km_quad.shape[1]
        if km_sext is not None:
            knobs_k += km_sext.shape[1]
        if km_dip is not None:
            knobs_k += km_dip.shape[1]
        if ksm_quad is not None:
            knobs_ks += ksm_quad.shape[1]
        if ksm_sext is not None:
            knobs_ks += ksm_sext.shape[1]
        if ksm_dip is not None:
            knobs_ks += ksm_dip.shape[1]
        if ks_skewquad is not None:
            knobs_skewquad += ks_skewquad.shape[1]

        if config.fit_gain_bpm:
            knobs_linear += 2*nbpm
        if config.fit_roll_bpm:
            knobs_linear += nbpm
        if config.fit_gain_corr:
            knobs_linear += nch + ncv
        if config.fit_energy_shift:
            knobs_linear += nch + ncv
        if config.fit_dipoles_kick:
            knobs_linear += 3
        if config.fit_girder_shift:
            knobs_gir += girder_shift.shape[1]

        nknobs = knobs_k + knobs_ks + knobs_skewquad
        nknobs += knobs_linear
        nknobs += knobs_gir

        jloco = _np.zeros(
            (2*nbpm*(nch+ncv+1), nknobs))
        idx = 0
        if config.fit_quadrupoles:
            num = km_quad.shape[1]
            jloco[:, idx:idx+num] = km_quad
            idx += num
        if config.fit_sextupoles:
            num = km_sext.shape[1]
            jloco[:, idx:idx+num] = km_sext
            idx += num
        if config.fit_dipoles:
            num = km_dip.shape[1]
            jloco[:, idx:idx+num] = km_dip
            idx += num
        if config.fit_quadrupoles_coupling:
            num = ksm_quad.shape[1]
            jloco[:, idx:idx+num] = ksm_quad
            idx += num
        if config.fit_sextupoles_coupling:
            num = ksm_sext.shape[1]
            jloco[:, idx:idx+num] = ksm_sext
            idx += num
        if config.fit_dipoles_coupling:
            num = ksm_dip.shape[1]
            jloco[:, idx:idx+num] = ksm_dip
            idx += num
        if config.fit_gain_bpm:
            num = dmdg_bpm.shape[1]
            jloco[:, idx:idx+num] = dmdg_bpm
            idx += num
        if config.fit_roll_bpm:
            num = dmdalpha_bpm.shape[1]
            jloco[:, idx:idx+num] = dmdalpha_bpm
            idx += num
        if config.fit_gain_corr:
            num = dmdg_corr.shape[1]
            jloco[:, idx:idx+num] = dmdg_corr
            idx += num
        if config.fit_dipoles_kick:
            num = kick_dip.shape[1]
            jloco[:, idx:idx+num] = kick_dip
            idx += num
        if config.fit_energy_shift:
            num = energy_shift.shape[1]
            jloco[:, idx:idx+num] = energy_shift
            idx += num
        if config.fit_skew_quadrupoles:
            num = knobs_skewquad
            jloco[:, idx:idx+num] = ks_skewquad
            idx += num
        if config.fit_girder_shift:
            num = knobs_gir
            jloco[:, idx:idx+num] = girder_shift
            idx += num
        return jloco

    @staticmethod
    def jloco_apply_weight(jloco, weight_bpm, weight_corr):
        """."""
        weight = (weight_bpm * weight_corr[None, :]).ravel()
        return weight[:, None] * jloco

    @staticmethod
    def param_select(config, param):
        """."""
        idx = 0
        param_dict = dict()
        if config.fit_quadrupoles:
            size = len(config.quad_indices)
            param_dict['quadrupoles_gradient'] = param[idx:idx+size]
            idx += size
        if config.fit_sextupoles:
            size = len(config.sext_indices)
            param_dict['sextupoles_gradient'] = param[idx:idx+size]
            idx += size
        if config.fit_dipoles:
            size = len(config.dip_indices)
            param_dict['dipoles_gradient'] = param[idx:idx+size]
            idx += size
        if config.fit_quadrupoles_coupling:
            size = len(config.quad_indices_ks)
            param_dict['quadrupoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_sextupoles_coupling:
            size = len(config.sext_indices)
            param_dict['sextupoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_dipoles_coupling:
            size = len(config.dip_indices_ks)
            param_dict['dipoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_gain_bpm:
            size = 2*config.nr_bpm
            param_dict['gain_bpm'] = param[idx:idx+size]
            idx += size
        if config.fit_roll_bpm:
            size = config.nr_bpm
            param_dict['roll_bpm'] = param[idx:idx+size]
            idx += size
        if config.fit_gain_corr:
            size = config.nr_corr
            param_dict['gain_corr'] = param[idx:idx+size]
            idx += size
        if config.fit_dipoles_kick:
            size = len(config.dip_indices)
            param_dict['dipoles_kick'] = param[idx:idx+size]
            idx += size
        if config.fit_energy_shift:
            size = config.nr_corr
            param_dict['energy_shift'] = param[idx:idx+size]
            idx += size
        if config.fit_skew_quadrupoles:
            size = len(config.skew_quad_indices)
            param_dict['skew_quadrupoles'] = param[idx:idx+size]
            idx += size
        if config.fit_girder_shift:
            size = config.gir_indices.shape[0]
            param_dict['girders_shift'] = param[idx:idx+size]
            idx += size
        return param_dict

    @staticmethod
    def kronecker(i, j, size):
        """."""
        kron = _np.zeros((size, size))
        if i == j:
            kron[i, i] = 1
        else:
            kron[i, j] = 1
            kron[j, i] = 1
        return kron
