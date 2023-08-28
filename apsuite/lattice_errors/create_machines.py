#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
import time as _time
import copy as _copy

import pyaccel as _pyaccel
import pymodels as _pymodels
from apsuite.orbcorr import OrbitCorr, CorrParams
from apsuite.optics_analysis import TuneCorr, OpticsCorr, CouplingCorr
from apsuite.commisslib.measure_bba import BBAParams
from mathphys.functions import save_pickle, load_pickle
from mathphys import units


class MachinesParams:

    class CoupCorrParams:

        def __init__(self):
            self.nr_singval = 80
            self.weight_dispy = 1e5
            self.tolerance = 1e-8

    class OptCorrParams:

        def __init__(self):
            self.nr_singval = 150
            self.tolerance = 1e-6

    def __init__(self):

        self.acc = 'SI'
        self.orbcorr_params = CorrParams()  # Orbit corr params
        self.optcorr_params = self.OptCorrParams()  # Optics corr params
        self.coupcorr_params = self.CoupCorrParams()  # Coupling corr params
        self.ramp_with_ids = False
        self.do_bba = True
        self.force_orb_correction = True
        self.do_multipoles_corr = True
        self.do_optics_corr = True
        self.do_coupling_corr = True
        self.save_jacobians = False
        self.load_jacobians = True


class GenerateMachines:
    """Class to generate errors and create random machines with them."""

    def __init__(self, params):
        """Class attributes."""
        self._machines_data = None
        self._nominal_model = None
        self._famdata = None
        self._nr_mach = 20
        self._seed = 140699
        self._ids = None
        self._fam_errors_dict = None
        self._orbcorr = None
        self._models = []
        self._functions = {
            'posx': _pyaccel.lattice.add_error_misalignment_x,
            'posy': _pyaccel.lattice.add_error_misalignment_y,
            'roll': _pyaccel.lattice.add_error_rotation_roll,
            'pitch': _pyaccel.lattice.add_error_rotation_pitch,
            'yaw': _pyaccel.lattice.add_error_rotation_yaw,
            'excit': _pyaccel.lattice.add_error_excitation_main,
            'kdip': _pyaccel.lattice.add_error_excitation_kdip}
        self.params = params
        self._save_jacobians = False
        self._load_jacobians = True
        self._orbcorr_params = CorrParams()
        self._ramp_with_ids = False
        self._do_bba = True
        self._do_opt_corr = True
        self._do_singval_ramp = True
        self._corr_multipoles = True
        self._do_coupling_corr = True

    @property
    def machines_data(self):
        """Dictionary with the data of all machines.

        Returns:
            dictionary: The keys select the machine number and the parameter
                    to be analysed.
        """
        return self._machines_data

    @machines_data.setter
    def machines_data(self, value):
        self._machines_data = value

    @property
    def ids(self):
        """Dictionary with insertion devices information.

        Returns:
            dictionary: It contains informations as the ID name and its
                straight section.
        """
        return self._ids

    @ids.setter
    def ids(self, value):
        self._ids = value

    @property
    def nr_mach(self):
        """Number of random machines.

        Returns:
            Int: Number of machines
        """
        return self._nr_mach

    @nr_mach.setter
    def nr_mach(self, value):
        if isinstance(value, int):
            self._nr_mach = value
        else:
            raise ValueError('Number of machines must be an integer')

    @property
    def seed(self):
        """Seed to generate random errors.

        Returns:
            Int: Seed number
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def famdata(self):
        """Dictionary with information about families present in the model.

        Returns:
            dict: It contains information about the lattice families.

        """
        return self._famdata

    @famdata.setter
    def famdata(self, value):
        self._famdata = value

    @property
    def fam_errors_dict(self):
        """Dictionary containing the errors for all families.

        Returns:
            dictionary: The keys are: family_name, error_type and index.

        """
        return self._fam_errors_dict

    @fam_errors_dict.setter
    def fam_errors_dict(self, value):
        self._fam_errors_dict = value

    @property
    def models(self):
        """List if lattice models with errors.

        Returns:
            List of pymodels objects: list with nr_mach elements.

        """
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    @property
    def nominal_model(self):
        """Reference model to perform corrections.

        Returns:
            pymodels object: reference lattice.

        """
        return self._nominal_model

    @nominal_model.setter
    def nominal_model(self, value):
        self._nominal_model = value

    @property
    def save_jacobians(self):
        """Option to save the calculated jacobians.

        Returns:
            Boolean: If True the jacobians will be saved.

        """
        return self._save_jacobians

    @property
    def load_jacobians(self):
        """Option to load jacobians.

        Returns:
            Boolean: If True the jacobians will be loaded.

        """
        return self.params.load_jacobians

    @property
    def orbcorr(self):
        """Object of the class OrbitCorr.

        Returns:
            OrbitCorr object: Do orbit correction.

        """
        return self._orbcorr

    @orbcorr.setter
    def orbcorr(self, value):
        self._orbcorr = value

    @property
    def orbcorr_params(self):
        """Object of the class CorrParams.

        Returns:
            CorrParams object: Parameters used in orbit correction.

        """
        return self.params.orbcorr_params

    @property
    def optcorr_params(self):

        return self.params.optcorr_params

    @property
    def coupcorr_params(self):

        return self.params.coupcorr_params

    @property
    def ramp_with_ids(self):
        """Option to ramp errors with ids.

        Returns:
            Boolean: If true all ids will be inseted in the random machines.

        """
        return self.params.ramp_with_ids

    @property
    def do_opt_corr(self):
        """Option to correct optics.

        Returns:
            Boolean: If true optics will be corrected.

        """
        return self.params.do_optics_corr

    @property
    def do_bba(self):
        """Option to do bba.

        Returns:
            Boolean: If true bba will be executed.

        """
        return self.params.do_bba

    @property
    def corr_multipoles(self):
        """Option to correct machine after the insertion of multipoles errors.

        Returns:
            bool: If true orbit, optics, tune and coupling corrections will
                be done.

        """
        return self.params.do_multipoles_corr

    @property
    def do_coupling_corr(self):
        """Option to correct coupling.

        Returns:
            bool: If true coupling corrections will be done.

        """
        return self.params.do_coupling_corr

    def _create_models(self, nr_mach):
        """Create the models in which the errors will be applied."""

        accelerators = {
            'SI': _pymodels.si.create_accelerator,
            'BO': _pymodels.bo.create_accelerator,
            'TS': _pymodels.ts.create_accelerator,
            'TB': _pymodels.tb.create_accelerator,
            'LI': _pymodels.li.create_accelerator,
        }
        models_ = list()
        ids = None
        if self.ramp_with_ids:
            ids = self.ids
        for _ in range(nr_mach):
            if self.params.acc == 'SI':
                model = accelerators[self.params.acc](ids=ids)
            else:
                model = accelerators[self.params.acc]
            model.cavity_on = False
            model.radiation_on = 0
            model.vchamber_on = False
            models_.append(model)
        return models_

    def get_bba_quad_idcs(self):
        """Get the indices of the quadrupoles where the bba will be done."""
        quaddevnames = list(BBAParams.QUADNAMES)
        quads = [q for q in self.famdata.keys() if q[0] == 'Q' and q[1] != 'N']
        quads_idcs = list()
        for qfam in quads:
            qfam = self.famdata[qfam]
            for idx, devname in zip(qfam['index'], qfam['devnames']):
                if devname in quaddevnames:
                    quads_idcs.append(idx)
        bba_quad_idcs = _np.sort(_np.array(quads_idcs).ravel())
        return bba_quad_idcs

    def apply_errors(self, nr_steps, mach):
        """Apply errors from file to the models (except for multipole errors).

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            mach (int): Index of the machine.

        """
        print('Applying errors...', end='')
        for family in self.fam_errors_dict.values():
            inds = family['index']
            error_types = [err for err in family.keys() if err != 'index']
            for error_type in error_types:
                if error_type == 'multipoles':
                    continue
                errors = family[error_type]
                self._functions[error_type](
                    self.models[mach], inds, errors[mach]/nr_steps)
        print('Done!')

    def apply_multipoles_errors(self, nr_steps, mach):
        """Apply multipole errors.

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            mach (int): Index of the machine.

        """
        error_type = 'multipoles'
        for fam, family in self.fam_errors_dict.items():
            inds = family['index']
            if error_type not in family:
                continue
            main_monoms = {'B': 1, 'Q': 2, 'S': 3, 'QS': -2}
            mag_key = fam[0] if fam != 'QS' else fam
            main_monom = _np.ones(len(inds))*main_monoms[mag_key]
            r0 = family[error_type]['r0']
            polb_order = list(family[error_type]['normal'])
            pola_order = list(family[error_type]['skew'])
            Bn_norm = _np.zeros((len(inds), max(polb_order)+1))
            An_norm = _np.zeros((len(inds), max(pola_order)+1))
            for n in polb_order:
                Bn_norm[:, n] = family[error_type]['normal'][n][mach]/nr_steps
            for n in pola_order:
                An_norm[:, n] = family[error_type]['skew'][n][mach]/nr_steps
            _pyaccel.lattice.add_error_multipoles(
                self.models[mach], inds, r0, main_monom, Bn_norm, An_norm)

    def _get_girder_errors(self, nr_steps, step, idcs, mach):
        """Get a fraction of the girder errors.

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            step (int): number of the current step
            idcs (1D numpy array): List of indices to get the girder errors
            mach (int): Index of the machine.

        Returns:
            1D numpy array: Girder errors in the chosen indices.

        """
        gir_errx = list()
        gir_erry = list()
        fam_girs = self.fam_errors_dict['girder']
        for i, girder in enumerate(fam_girs['index']):
            for idx in girder:
                if not _np.any(idcs == idx):
                    continue
                gir_errx.append(fam_girs['posx'][mach][i])
                gir_erry.append(fam_girs['posy'][mach][i])
        return _np.array(gir_errx + gir_erry).ravel() * (step/nr_steps)

    def simulate_bba(self, bba_quad_idcs, nr_steps, step, mach):
        """Simulate the bba method.

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            step (int): number of the current step
            mach (int): Index of the machine.

        Returns:
            1D numpy array: Reference orbit
        """
        bpms = _np.array(self.famdata['BPM']['index']).ravel()
        orb_len = len(bpms)
        orb0 = _np.zeros(2*orb_len)
        orb0[:orb_len] += _pyaccel.lattice.get_error_misalignment_x(
                self.models[mach], bba_quad_idcs).ravel()
        orb0[:orb_len] += _pyaccel.lattice.get_error_misalignment_x(
                self.models[mach], bpms).ravel()
        orb0[orb_len:] += _pyaccel.lattice.get_error_misalignment_y(
                self.models[mach], bba_quad_idcs).ravel()
        orb0[orb_len:] += _pyaccel.lattice.get_error_misalignment_y(
                self.models[mach], bpms).ravel()

        # NOTE: The BPM errors will contain their girder errors. We need to
        # isolate the BBA errors from girder errors where the BPM is installed.
        bpm_girder_errors = self._get_girder_errors(nr_steps, step, bpms, mach)
        orb0 -= bpm_girder_errors
        return orb0

    def _config_orb_corr(self, jac=None):
        """Configure orbcorr object. This is an object of the class
             apsuite.orbcorr.orbit_correction.CorrParams.

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Orbit response matrix.
        """
        self.orbcorr = OrbitCorr(
            self.nominal_model, self.params.acc, params=self.orbcorr_params)
        if jac is None:
            jac = self.orbcorr.get_jacobian_matrix()
        self.orbmat = jac
        return jac

    def _correct_orbit_once(self, orb0, mach):
        """Do one orbit correction.

        Args:
            orb0 (1D numpy array): Reference orbit
            mach (int): Index of the machine.

        Returns:
            1D numpy array, 1D numpy array: Returns corrected orbit and kicks.

        """
        print('Correcting orbit...', end='')
        self.orbcorr.respm.model = self.models[mach]
        corr_status = self.orbcorr.correct_orbit(
            jacobian_matrix=self.orbmat, goal_orbit=orb0)
        if corr_status == 0:
            print('Done\n')
        elif corr_status == 1:
            print('Correction could not achieve RMS tolerance!\n')
        elif corr_status == 2:
            print('Correction could not converge!\n')
        else:
            print('Correctors are saturated!\n')

        return self.orbcorr.get_orbit(), self.orbcorr.get_kicks(), corr_status

    def _correct_orbit_iter(self, orb0, mach, numsingval, nriter=10):
        """Correct orbit iteratively.

        Args:
            orb0 (1D numpy array): Reference orbit
            mach (int): Index of the machine.

        Returns:
            1D numpy array: corrected orbit;
            1D numpy array: kicks applied;
            bool: correction status;
            int: initial minimum singular value.

        """
        kicks_before = self.orbcorr.get_kicks()

        self.orbcorr_params.numsingval = numsingval
        orb_t, kicks_t, corr_stts = self._correct_orbit_once(orb0, mach)

        i = 1
        while corr_stts == 1 or corr_stts == 3:
            self.orbcorr.set_kicks(kicks_before)
            self.orbcorr_params.numsingval -= 10
            if i > nriter:
                self.orbcorr_params.numsingval = numsingval
                return False
            print('Number of singular values: {:.0f}'.format(
                    self.orbcorr_params.numsingval))
            orb_t, kicks_t, corr_stts = self._correct_orbit_once(orb0, mach)
            i += 1

        self.orbf_, self.kicks_ = orb_t, kicks_t
        self.orbcorr_params.numsingval = numsingval
        return self.orbf_, self.kicks_, corr_stts

    def _config_tune_corr(self, jac=None):
        """Configure TuneCorr object.

        This is an object of the class:
            apsuite.optics_analysis.tune_correction.TuneCorr

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Tune response matrix.

        """
        self.tunecorr = TuneCorr(
            self.nominal_model, self.params.acc, method='Proportional',
            grouping='TwoKnobs')
        if jac is None:
            self.tunemat = self.tunecorr.calc_jacobian_matrix()
        self.tunemat = jac
        self.goal_tunes = self.tunecorr.get_tunes()
        print('Nominal tunes: {:.4f} {:.4f}'.format(*self.goal_tunes))
        return jac

    def _correct_tunes(self, mach):
        """Correct tunes.

        Args:
            mach (int): Index of the machine.

        """
        self.tunecorr.correct_parameters(
            model=self.models[mach], goal_parameters=self.goal_tunes,
            jacobian_matrix=self.tunemat)

    def _calc_coupling(self, mach):
        """Calc minimum tune separation.

        Args:
            mach (int): Index of the machine.

        Returns:
            float: minimum tune separation [no unit]

        """
        ed_teng, *_ = _pyaccel.optics.calc_edwards_teng(self.models[mach])
        min_tunesep, ratio =\
            _pyaccel.optics.estimate_coupling_parameters(ed_teng)

        return min_tunesep

    def _config_coupling_corr(self, jac=None):
        """Config CouplingCorr object.

        This is an object of the class
            apsuite.optics_analysis.coupling_correction.CouplingCorr.

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Coupling response matrix.

        """
        idcs = list()
        qs_fam = self.famdata['QS']
        for idx, sub in zip(qs_fam['index'], qs_fam['subsection']):
            if 'C2' not in sub:
                idcs.append(idx)
        self.coup_corr = CouplingCorr(self.nominal_model,
                                      self.params.acc, skew_list=idcs)
        if jac is None:
            weight_dispy = self.coupcorr_params.weight_dispy
            self.coupmat = self.coup_corr.calc_jacobian_matrix(
                model=self.nominal_model, weight_dispy=weight_dispy)
        self.coupmat = jac
        return jac

    def _correct_coupling(self, mach):
        """Correct coupling.

        Args:
            mach (int): Index of the machine.

        """
        self.coup_corr.model = self.models[mach]
        weight_dispy = self.coupcorr_params.weight_dispy
        nsv = self.coupcorr_params.nr_singval
        tol = self.coupcorr_params.tolerance
        self.coup_corr.coupling_correction(
            jacobian_matrix=self.coupmat, nsv=nsv, tol=tol,
            weight_dispy=weight_dispy)

    def _config_optics_corr(self, jac=None):
        """Config OpticsCorr object.

        This is an object of the class
            apsuite.optics_analysis.optics_correction.OpticsCorr.

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Optics response matrix.

        """
        self.opt_corr = OpticsCorr(self.nominal_model, self.params.acc)
        if jac is None:
            self.optmat = self.opt_corr.calc_jacobian_matrix()
        self.optmat = jac
        return jac

    def _correct_optics(self, mach):
        """Correct optics.

        Args:
            mach (int): Index of the machine.

        """
        self.opt_corr.model = self.models[mach]
        nsv = self.optcorr_params.nr_singval
        tol = self.optcorr_params.tolerance
        return self.opt_corr.optics_corr_loco(
            goal_model=self.nominal_model, nsv=nsv,
            jacobian_matrix=self.optmat, tol=tol)

    def _do_all_opt_corrections(self, mach):
        """Do all optics corrections - beta, tunes and coupling.

        Args:
            mach (int): Index of the machine.

        Returns:
            1D numpy array, 1D numpy array, 1D numpy array: twiss, ed tangs
                and twiss parameters of the nominal machine

        """
        # Symmetrize optics
        print('Correcting optics...')
        res = self._correct_optics(mach) == 1
        print('Optics correction tolerance achieved: ', res)
        print()

        # Correct tunes
        print('Correcting tunes:')
        tunes = self.tunecorr.get_tunes(model=self.models[mach])
        print('Old tunes: {:.4f} {:.4f}'.format(tunes[0], tunes[1]))
        self._correct_tunes(mach)
        tunes = self.tunecorr.get_tunes(model=self.models[mach])
        print('New tunes: {:.4f} {:.4f}'.format(*tunes))
        print()

        if self.do_coupling_corr:
            print('Correcting coupling:')
            mintune = self._calc_coupling(mach)
            print(f'Minimum tune separation before corr: {100*mintune:.3f} %')
            self._correct_coupling(mach)
            mintune = self._calc_coupling(mach)
            print(f'Minimum tune separation after corr: {100*mintune:.3f} %')
            print()

        return

    def configure_corrections(self):
        """Configure all corrections - orbit and optics."""
        orbmat, optmat, tunemat, coupmat = None, None, None, None
        if self.load_jacobians:
            respmats = load_pickle('respmats')
            orbmat = respmats['orbmat']
            optmat = respmats['optmat']
            tunemat = respmats['tunemat']
            coupmat = respmats['coupmat']

        # Config orbit correction
        print('Configuring orbit correction...')
        orbmat = self._config_orb_corr(orbmat)

        # Config optics correction
        print('Configuring optics correction...')
        optmat = self._config_optics_corr(optmat)

        # Config tune correction
        print('Configuring tune correction...')
        tunemat = self._config_tune_corr(tunemat)

        # Config coupling correction
        print('Configuring coupling correction...')
        coupmat = self._config_coupling_corr(coupmat)

        if self.save_jacobians:
            respmats = dict()
            respmats['orbmat'] = orbmat
            respmats['optmat'] = optmat
            respmats['tunemat'] = tunemat
            respmats['coupmat'] = coupmat
            save_pickle(respmats, 'respmats', overwrite=True)

    def save_machines(self, sufix=None):
        """Save all random machines in a pickle.

        Args:
            sufix (string, optional): sufix will be added in the filename.
                Defaults to None.

        """
        filename = self.params.acc + '_' + str(self.nr_mach)
        filename += '_machines_seed_' + str(self.seed)
        if self.ramp_with_ids:
            filename += '_' + self.ids[0].fam_name
        if sufix is not None:
            filename += sufix
        save_pickle(self.machines_data, filename, overwrite=True)

    def load_machines(self):
        """Load random machines.

        Returns:
            Dictionary: Contains all random machines and its data.

        """
        filename = self.params.acc + '_' + str(self.nr_mach)
        filename += '_machines_seed_' + str(self.seed)
        data = load_pickle(filename)
        print('loading ' + filename)
        return data

    def _corr_machines_ramping_sv(
            self, mach, nr_steps, bba_quad_idcs,
            step_data):

        mod = self.models[mach]
        # Save sextupoles values and set then to zero
        sx_idx = self.famdata['SN']['index']
        sx_stren = _pyaccel.lattice.get_attribute(mod, 'SL', sx_idx)
        _pyaccel.lattice.set_attribute(mod, 'SL', sx_idx, 0.0)

        corr_sucess = False
        init_numsingval = self.original_numsingval
        j = 1
        while not corr_sucess:
            print('Initial nr of singular values: {:.0f}'.format(
                init_numsingval))
            for step in range(nr_steps):
                print(f'Errors Ramping Step {step+1:d}/{nr_steps:d}')
                self.apply_errors(nr_steps, mach)

                # Orbit set by BBA or set to zero
                orb0 = _np.zeros(2*len(bba_quad_idcs))
                if self.do_bba:
                    orb0 = self.simulate_bba(
                        bba_quad_idcs, nr_steps, step+1, mach)

                if j > 10:
                    orb_t, kicks_t, corr_stts = self._correct_orbit_once(
                        orb0, mach)
                    print('Correcting optics...')
                    res = self._correct_optics(mach) == 1
                    print('Optics correction tolerance achieved: ', res)
                    print()
                    numsingval = self.original_numsingval
                    self.orbcorr_params.numsingval = numsingval

                # Correct orbit
                res = self._correct_orbit_iter(orb0, mach, init_numsingval)
                corr_sucess = True
                if res:
                    orbf, kicks_, corr_status = res
                else:
                    corr_sucess = False
                    break

                step_dict = dict()
                step_dict['orbcorr_status'] = corr_status
                step_dict['ref_orb'] = orb0
                step_dict['orbit'] = orbf
                step_dict['corr_kicks'] = kicks_
                step_data['step_' + str(step + 1)] = step_dict

                _pyaccel.lattice.set_attribute(
                    mod, 'SL', sx_idx, (step + 1)*sx_stren/nr_steps)

            if corr_sucess:
                # Perform one orbit correction after turning ON sextupoles
                res = self._correct_orbit_iter(orb0, mach, init_numsingval)
            if res:
                corr_sucess = True
                orbf, kicks_, corr_status = res
            else:
                mod = self._create_models(1)[0]
                self.orbcorr.respm.model = mod
                _pyaccel.lattice.set_attribute(mod, 'SL', sx_idx, 0.0)
                self.models[mach] = mod
                init_numsingval -= 10
                corr_sucess = False
                j += 1
        return step_data, orbf, orb0, kicks_, corr_status

    def _corr_machines_fix_sv(
            self, mach, nr_steps, bba_quad_idcs,
            step_data):

        mod = self.models[mach]
        # Save sextupoles values and set then to zero
        sx_idx = self.famdata['SN']['index']
        sx_stren = _pyaccel.lattice.get_attribute(mod, 'SL', sx_idx)
        _pyaccel.lattice.set_attribute(mod, 'SL', sx_idx, 0.0)

        for step in range(nr_steps):
            print(f'Errors Ramping Step {step+1:d}/{nr_steps:d}')
            self.apply_errors(nr_steps, mach)

            # Orbit set by BBA or set to zero
            orb0 = _np.zeros(2*len(bba_quad_idcs))
            if self.do_bba:
                orb0 = self.simulate_bba(
                    bba_quad_idcs, nr_steps, step+1, mach)

            orbf, kicks, corr_stts = self._correct_orbit_once(orb0, mach)

            step_dict = dict()
            step_dict['orbcorr_status'] = corr_stts
            step_dict['ref_orb'] = orb0
            step_dict['orbit'] = orbf
            step_dict['corr_kicks'] = kicks
            step_data['step_' + str(step + 1)] = step_dict

            _pyaccel.lattice.set_attribute(
                mod, 'SL', sx_idx, (step + 1)*sx_stren/nr_steps)

        # Perform one orbit correction after turning ON sextupoles
        orbf, kicks, corr_stts = self._correct_orbit_once(orb0, mach)

        return step_data, orbf, orb0, kicks, corr_stts

    def generate_machines(self, nr_steps=5):
        """Generate all random machines.

        Args:
            nr_steps (int, optional): Number of steps the ramp of errors and
                sextupoles will be done. Defaults to 5.

        Returns:
            Dictionary: Contains all random machines and its data.

        """
        # Get quadrupoles near BPMs indices
        bba_quad_idcs = self.get_bba_quad_idcs()

        # Create models
        self.models = self._create_models(self.nr_mach)

        data = dict()
        self.original_numsingval = _copy.copy(self.orbcorr_params.numsingval)
        for mach in range(self.nr_mach):
            print('Machine ', mach)
            step_data = dict()
            if self.params.do_singval_ramp:
                res = self._corr_machines_ramping_sv(
                    mach, nr_steps, bba_quad_idcs, step_data)
            else:
                res = self._corr_machines_fix_sv(
                    mach, nr_steps, bba_quad_idcs, step_data)
            step_data, orbf, orb0, kicks_, corr_status = res

            # Save last orbit corr data
            step_dict = dict()
            step_dict['orbcorr_status'] = corr_status
            step_dict['ref_orb'] = orb0
            step_dict['orbit'] = orbf
            step_dict['corr_kicks'] = kicks_
            step_data['step_' + str(nr_steps + 2)] = step_dict

            # Do optics corrections:
            step_dict = step_data['step_' + str(nr_steps + 2)]
            if self.do_opt_corr:
                self._do_all_opt_corrections(mach)

            # Apply multipoles errors
            self.apply_multipoles_errors(1, mach)

            # Correct multipoles errors
            if self.corr_multipoles:
                self._correct_orbit_once(orb0, mach)

                if self.do_opt_corr:
                    self._do_all_opt_corrections(mach)

            edteng, *_ = _pyaccel.optics.calc_edwards_teng(self.models[mach])
            twiss, *_ = _pyaccel.optics.calc_twiss(self.models[mach])
            twiss0, *_ = _pyaccel.optics.calc_twiss(self.nominal_model)

            dbetax = (twiss.betax - twiss0.betax)/twiss0.betax
            dbetay = (twiss.betay - twiss0.betay)/twiss0.betay
            step_dict['orbcorr_status'] = corr_status
            step_dict['ref_orb'] = orb0
            step_dict['orbit'] = orbf
            step_dict['corr_kicks'] = kicks_
            step_dict['twiss'] = twiss
            step_dict['edteng'] = edteng
            step_dict['betabeatingx'] = dbetax
            step_dict['betabeatingy'] = dbetay
            step_data['step_final'] = step_dict

            model_dict = dict()
            model_dict['model'] = self.models[mach]
            model_dict['data'] = step_data
            data['orbcorr_params'] = self.orbcorr_params
            data[mach] = model_dict
            self.machines_data = data
            self.save_machines()
        return data

    def insert_kickmap(self, model):
        """Insert a kickmap into the model

        Args:
            model (pymodels object): Lattice

        Returns:
            pymodels object: Lattice with kickmap

        """
        kickmaps = _pymodels.si.lattice.create_id_kickmaps_dict(
            self.ids, energy=3e9)
        twiss, *_ = _pyaccel.optics.calc_twiss(model, indices='closed')
        print('Model without ID:')
        print('length : {:.4f} m'.format(model.length))
        print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/_np.pi))
        print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/_np.pi))
        print()

        for id_ in self.ids:
            idcs = _np.array(self.famdata[id_.fam_name]['index']).ravel()
            for i, idc in enumerate(idcs):
                model[idc] = kickmaps[id_.subsec][i]

        twiss, *_ = _pyaccel.optics.calc_twiss(model, indices='closed')
        print('Model with ID:')
        print('length : {:.4f} m'.format(model.length))
        print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/_np.pi))
        print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/_np.pi))
        print()
        return model

    def corr_ids(self, sufix=None):
        """Do all corrections after the ID insertion.

        Args:
            sufix (string, optional): sufix will be added in the filename.
                Defaults to None.

        """
        data_mach = self.load_machines()
        self.orbcorr_params = data_mach['orbcorr_params']
        # insert ID in each machine
        models = list()
        for mach in range(self.nr_mach):
            model_ = data_mach[mach]['model']
            model = self.insert_kickmap(model_)
            models.append(model)
        self.models = models

        data = dict()
        for mach in range(self.nr_mach):
            step_data = dict()
            # get ref_orb
            ref_orb = data_mach[mach]['data']['step_final']['ref_orb']
            # correct orbit
            orbf_, kicks_ = self._correct_orbit_once(ref_orb, mach=mach)
            # do all optics corretions
            for i in range(1):
                twiss, edtang, twiss0 = self._do_all_opt_corrections(
                    mach)

            dbetax = (twiss.betax - twiss0.betax)/twiss0.betax
            dbetay = (twiss.betay - twiss0.betay)/twiss0.betay
            step_dict = dict()
            step_dict['twiss'] = twiss
            step_dict['edtang'] = edtang
            step_dict['betabeatingx'] = dbetax
            step_dict['betabeatingy'] = dbetay
            step_dict['ref_orb'] = ref_orb
            step_dict['orbit'] = orbf_
            step_dict['corr_kicks'] = kicks_
            step_data['step_final'] = step_dict

            model_dict = dict()
            model_dict['model'] = self.models[mach]
            model_dict['data'] = step_data
            data[mach] = model_dict
        self.machines_data = data
        sufix_ = '_' + self.ids[0].fam_name + '_symm'
        sufix = sufix_ + sufix
        self.save_machines(sufix=sufix)
