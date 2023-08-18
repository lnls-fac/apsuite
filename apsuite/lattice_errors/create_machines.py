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


class GenerateMachines():
    """Class to generate errors and create random machines with them."""

    def __init__(self):
        """Class attributes."""
        self._machines_data = None
        self._ids = None
        self._nr_mach = 20
        self._seed = 140699
        self._famdata = None
        self._fam_errors_dict = None
        self._bba_quad_idcs = None
        self._nominal_model = None
        self._models = []
        self._functions = {'posx': _pyaccel.lattice.add_error_misalignment_x,
                           'posy': _pyaccel.lattice.add_error_misalignment_y,
                           'roll': _pyaccel.lattice.add_error_rotation_roll,
                           'pitch': _pyaccel.lattice.add_error_rotation_pitch,
                           'yaw': _pyaccel.lattice.add_error_rotation_yaw,
                           'excit': _pyaccel.lattice.add_error_excitation_main,
                           'kdip': _pyaccel.lattice.add_error_excitation_kdip}
        self._orbcorr_params = CorrParams()
        self._orbcorr = None
        self._save_jacobians = False
        self._load_jacobians = True
        self._do_bba = True
        self._ramp_with_ids = False
        self._do_opt_corr = True
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
        """Dictionary with all information about the families present in the
            model lattice.

        Returns:
            Dictionary: It contains information about the lattice families.
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
    def bba_quad_idcs(self):
        """Quadrupoles indices where the bba will be performed.

        Returns:
            numpy array: array whose elements are the bba indices.
        """
        return self._bba_quad_idcs

    @bba_quad_idcs.setter
    def bba_quad_idcs(self, value):
        self._bba_quad_idcs = value

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

    @save_jacobians.setter
    def save_jacobians(self, value):
        if type(value) != bool:
            raise ValueError('Save jacobian must be bool type')
        else:
            self._save_jacobians = value

    @property
    def load_jacobians(self):
        """Option to load jacobians.

        Returns:
            Boolean: If True the jacobians will be loaded.
        """
        return self._load_jacobians

    @load_jacobians.setter
    def load_jacobians(self, value):
        if type(value) != bool:
            raise ValueError('Load jacobian must be bool type')
        else:
            self._load_jacobians = value

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
        return self._orbcorr_params

    @orbcorr_params.setter
    def orbcorr_params(self, value):
        self._orbcorr_params = value

    @property
    def ramp_with_ids(self):
        """Option to ramp errors with ids.

        Returns:
            Boolean: If true all ids will be inseted in the random machines.
        """
        return self._ramp_with_ids

    @ramp_with_ids.setter
    def ramp_with_ids(self, value):
        if type(value) != bool:
            raise ValueError('ramp_with_ids must be bool type')
        else:
            self._ramp_with_ids = value

    @property
    def do_opt_corr(self):
        """Option to correct optics.

        Returns:
            Boolean: If true optics will be corrected.
        """
        return self._do_opt_corr

    @do_opt_corr.setter
    def do_opt_corr(self, value):
        if type(value) != bool:
            raise ValueError('do_opt_corr must be bool type')
        else:
            self._do_opt_corr = value

    @property
    def do_bba(self):
        """Option to do bba.

        Returns:
            Boolean: If true bba will be executed.
        """
        return self._do_bba

    @do_bba.setter
    def do_bba(self, value):
        if type(value) != bool:
            raise ValueError('do_bba must be bool type')
        else:
            self._do_bba = value

    @property
    def corr_multipoles(self):
        """Option to correct the machine after the insertion of multipoles
                errors.

        Returns:
            bool: If true orbit, optics, tune and coupling corrections will
                 be done.
        """
        return self._corr_multipoles

    @corr_multipoles.setter
    def corr_multipoles(self, value):
        if type(value) != bool:
            raise ValueError('corr_multipoles must be bool type')
        else:
            self._corr_multipoles = value

    @property
    def do_coupling_corr(self):
        """Option to correct coupling.

        Returns:
            bool: If true coupling corrections will
                 be done.
        """
        return self._do_coupling_corr

    @do_coupling_corr.setter
    def do_coupling_corr(self, value):
        if type(value) != bool:
            raise ValueError('do_coupling_corr must be bool type')
        else:
            self._do_coupling_corr = value

    def _create_models(self):
        """Create the models in which the errors will be applied."""
        models_ = list()
        if self.ramp_with_ids:
            ids = self.ids
        else:
            ids = None
        for _ in range(self.nr_mach):
            model = _pymodels.si.create_accelerator(ids=ids)
            model.cavity_on = False
            model.radiation_on = 0
            model.vchamber_on = False
            models_.append(model)
        self.models = models_

    def _get_bba_quad_idcs(self):
        """Get the indices of the quadrupoles where the bba will be done."""
        quaddevnames = list(BBAParams.QUADNAMES)
        quads = [famname for famname in self.famdata.keys()
                 if famname[0] == 'Q' and famname[1] != 'N']
        quads_idcs = list()
        for quadfam in quads:
            for idx, devname in zip(self.famdata[quadfam]['index'],
                                    self.famdata[quadfam]['devnames']):
                if devname in quaddevnames:
                    quads_idcs.append(idx)
        bba_quad_idcs = _np.array(quads_idcs).ravel()
        self.bba_quad_idcs = _np.sort(bba_quad_idcs)

    def _apply_errors(self, nr_steps, mach):
        """Apply errors from a load file in the models (except for multipole
            errors).

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            mach (int): Index of the machine.
        """
        print('Applying errors...', end='')
        for fam, family in self.fam_errors_dict.items():
            inds = family['index']
            error_types = [err for err in family.keys() if err != 'index']
            for error_type in error_types:
                if error_type != 'multipoles':
                    errors = family[error_type]
                    self._functions[error_type](
                        self.models[mach], inds,
                        errors[mach]/nr_steps)

        print('Done!')

    def _restore_error(self, step, nr_steps, mach):
        """Restore one step fraction of the errors.

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            mach (int): Index of the machine.
        """
        print('Restoring machine...', end='')
        for fam, family in self.fam_errors_dict.items():

            inds = family['index']
            error_types = [err for err in family.keys() if err != 'index']
            for error_type in error_types:
                if error_type != 'multipoles':
                    errors = family[error_type]
                    self._functions[error_type](
                        self.models[mach], inds,
                        -step*errors[mach]/nr_steps)

        print('Done!')

    def _apply_multipoles_errors(self, nr_steps, mach):
        """Apply multipoles errors

        Args:
            nr_steps (int): Number of steps the ramp of errors and sextupoles
                will be done.
            mach (int): Index of the machine.
        """
        error_type = 'multipoles'
        for fam, family in self.fam_errors_dict.items():
            inds = family['index']
            if error_type in family.keys():
                main_monoms = {'B': 1, 'Q': 2, 'S': 3, 'QS': -2}
                mag_key = fam[0] if fam != 'QS' else fam
                main_monom = _np.ones(len(inds))*main_monoms[mag_key]
                r0 = family[error_type]['r0']
                polb_order = list(family[error_type]['normal'].keys())
                pola_order = list(family[error_type]['skew'].keys())
                Bn_norm = _np.zeros((len(inds), max(polb_order)+1))
                An_norm = _np.zeros((len(inds), max(pola_order)+1))
                for n in polb_order:
                    Bn_norm[:, n] =\
                        family[error_type]['normal'][n][mach]/nr_steps
                for n in pola_order:
                    An_norm[:, n] =\
                        family[error_type]['skew'][n][mach]/nr_steps
                _pyaccel.lattice.add_error_multipoles(
                    self.models[mach], inds, r0,
                    main_monom, Bn_norm, An_norm)

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
        girder_errorx = list()
        girder_errory = list()
        for i, girder in enumerate(self.fam_errors_dict['girder']['index']):
            for idx in girder:
                if _np.any(idcs == idx):
                    girder_errorx.append(
                        self.fam_errors_dict['girder']['posx'][mach][i])
                    girder_errory.append(
                        self.fam_errors_dict['girder']['posy'][mach][i])
        girder_errors_idcs = _np.array(girder_errorx + girder_errory).ravel()
        return step*girder_errors_idcs/nr_steps

    def _simulate_bba(self, nr_steps, step, mach):
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
                self.models[mach], self.bba_quad_idcs).ravel()
        orb0[:orb_len] += _pyaccel.lattice.get_error_misalignment_x(
                self.models[mach], bpms).ravel()
        orb0[orb_len:] += _pyaccel.lattice.get_error_misalignment_y(
                self.models[mach], self.bba_quad_idcs).ravel()
        orb0[orb_len:] += _pyaccel.lattice.get_error_misalignment_y(
                self.models[mach], bpms).ravel()

        bpm_girder_errors = self._get_girder_errors(
            nr_steps, step, bpms, mach)
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
                self.nominal_model, 'SI', '4d',
                params=self.orbcorr_params)
        if jac is not None:
            self.orbmat = jac
        else:
            self.orbmat = self.orbcorr.get_jacobian_matrix()
        return self.orbmat

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
            print('Could not achieve tolerance!\n')
        elif corr_status == 2:
            print('Correction could not converge!\n')
        else:
            print('Done!\n')

        return self.orbcorr.get_orbit(), self.orbcorr.get_kicks(), corr_status

    def _correct_orbit_iter(self, orb0, mach, init_minsingval):
        """Correct orbit iteratively

        Args:
            orb0 (1D numpy array): Reference orbit
            mach (int): Index of the machine.

        Returns:
            1D numpy array, 1D numpy array: Returns corrected orbit and kicks.
        """
        kicks_before = self.orbcorr.get_kicks()

        orb_temp, kicks_temp, corr_status = self._correct_orbit_once(
                                                                orb0, mach)
        cod_peak = _np.max(1e6*(orb_temp-orb0))
        while corr_status == 2 or cod_peak >= 100:
            self.orbcorr.set_kicks(kicks_before)
            self.orbcorr_params.minsingval += 0.05
            if init_minsingval > 0.7:
                print('Correcting optics...')
                res = self._correct_optics(mach)
                res = True if res == 1 else False
                print('Optics correction tolerance achieved: ', res)
                print()
                init_minsingval = self.original_minsingval
                self.orbcorr_params.minsingval = init_minsingval
            if self.orbcorr_params.minsingval > 0.8:
                self.orbcorr_params.minsingval = init_minsingval
                return False
            print('Minimum singular value: {:.2f}'.format(
                    self.orbcorr_params.minsingval))
            orb_temp, kicks_temp, corr_status = self._correct_orbit_once(
                orb0, mach)
            cod_peak = _np.max(1e6*(orb_temp-orb0))

        self.orbf_, self.kicks_ = orb_temp, kicks_temp
        self.orbcorr_params.minsingval = init_minsingval
        return self.orbf_, self.kicks_, corr_status, init_minsingval

    def _config_tune_corr(self, jac=None):
        """Configure TuneCorr object. This is an object of the class
             apsuite.optics_analysis.tune_correction.TuneCorr.

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Tune response matrix.
        """
        self.tunecorr = TuneCorr(
                            self.nominal_model,
                            'SI', method='Proportional',
                            grouping='TwoKnobs')
        if jac is not None:
            self.tunemat = jac
        else:
            self.tunemat = self.tunecorr.calc_jacobian_matrix()
        self.goal_tunes = self.tunecorr.get_tunes()
        print('Nominal tunes: {:.4f} {:.4f}'.format(
            self.goal_tunes[0], self.goal_tunes[1]))
        return self.tunemat

    def _correct_tunes(self, mach):
        """Correct tunes.

        Args:
            mach (int): Index of the machine.
        """
        self.tunecorr.correct_parameters(
            model=self.models[mach],
            goal_parameters=self.goal_tunes,
            jacobian_matrix=self.tunemat)

    def _calc_coupling(self, mach):
        """Calc minimum tune separation.

        Args:
            mach (int): Index of the machine.

        Returns:
            float: minimum tune separation [no unit]
        """
        ed_tang, *_ = _pyaccel.optics.calc_edwards_teng(self.models[mach])
        min_tunesep, ratio =\
            _pyaccel.optics.estimate_coupling_parameters(ed_tang)

        return min_tunesep

    def _config_coupling_corr(self, jac=None):
        """Config CouplingCorr object. This is an object of the class
             apsuite.optics_analysis.coupling_correction.CouplingCorr.

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Coupling response matrix.
        """
        idcs = list()
        for idx, sub in zip(
                self.famdata['QS']['index'],
                self.famdata['QS']['subsection']):
            if 'C2' not in sub:
                idcs.append(idx)
        self.coup_corr = CouplingCorr(self.nominal_model, 'SI',
                                      skew_list=idcs)
        if jac is not None:
            self.coupmat = jac
        else:
            self.coupmat = self.coup_corr.calc_jacobian_matrix(
                model=self.nominal_model, weight_dispy=1e5)
        return self.coupmat

    def _correct_coupling(self, mach):
        """Correct coupling.

        Args:
            mach (int): Index of the machine.
        """
        self.coup_corr.model = self.models[mach]
        self.coup_corr.coupling_correction(
                    jacobian_matrix=self.coupmat, nr_max=10,
                    nsv=80, tol=1e-8, weight_dispy=1e5)

    def _config_optics_corr(self, jac=None):
        """Config OpticsCorr object. This is an object of the class
             apsuite.optics_analysis.optics_correction.OpticsCorr.

        Args:
            jac (2D numpy array, optional): Loaded jacobian. Defaults to None.

        Returns:
            2D numpy array: Optics response matrix.
        """
        self.opt_corr = OpticsCorr(self.nominal_model, 'SI')
        if jac is not None:
            self.optmat = jac
        else:
            self.optmat = self.opt_corr.calc_jacobian_matrix()
        return self.optmat

    def _correct_optics(self, mach):
        """Correct optics.

        Args:
            mach (int): Index of the machine.
        """
        self.opt_corr.model = self.models[mach]
        return self.opt_corr.optics_corr_loco(goal_model=self.nominal_model,
                                              nr_max=10, nsv=150,
                                              jacobian_matrix=self.optmat)

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
        for i in range(1):
            res = self._correct_optics(mach)
        res = True if res == 1 else False
        print('Optics correction tolerance achieved: ', res)
        print()

        # Correct tunes
        print('Correcting tunes:')
        tunes = self.tunecorr.get_tunes(model=self.models[mach])
        print('Old tunes: {:.4f} {:.4f}'.format(tunes[0], tunes[1]))
        self._correct_tunes(mach)
        tunes = self.tunecorr.get_tunes(
                    model=self.models[mach])
        print('New tunes: {:.4f} {:.4f}'.format(
                        tunes[0], tunes[1]))
        print()

        if self.do_coupling_corr:
            # Correct coupling
            print('Correcting coupling:')
            mintune = self._calc_coupling(mach)
            print('Minimum tune separation before corr: {:.3f} %'.format(
                     100*mintune))
            self._correct_coupling(mach)
            mintune = self._calc_coupling(mach)
            print('Minimum tune separation after corr: {:.3f} %'.format(
                                100*mintune))
            print()

        ed_tang, *_ = _pyaccel.optics.calc_edwards_teng(self.models[mach])
        twiss, *_ = _pyaccel.optics.calc_twiss(self.models[mach])
        twiss0, *_ = _pyaccel.optics.calc_twiss(self.nominal_model)

        return twiss, ed_tang, twiss0

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
        filename = str(self.nr_mach) + '_machines_seed_' + str(self.seed)
        if self.ramp_with_ids:
            filename += '_'
            filename += self.ids[0].fam_name
        if sufix is not None:
            filename += sufix

        save_pickle(self.machines_data, filename, overwrite=True)

    def load_machines(self):
        """Load random machines.

        Returns:
            Dictionary: Contains all random machines and its data.
        """
        filename = str(self.nr_mach) + '_machines_seed_' + str(self.seed)
        data = load_pickle(filename)
        print('loading ' + filename)
        return data

    def generate_machines(self, nr_steps=5):
        """Generate all random machines.

        Args:
            nr_steps (int, optional): Number of steps the ramp of errors and
                sextupoles will be done. Defaults to 5.

        Returns:
            Dictionary: Contains all random machines and its data.
        """
        # Get quadrupoles near BPMs indices
        self._get_bba_quad_idcs()

        # Create SI models
        self._create_models()

        data = dict()
        self.original_minsingval = _copy.copy(self.orbcorr_params.minsingval)
        for mach in range(self.nr_mach):
            print('Machine ', mach)

            # Save sextupoles values and set then to zero
            index = self.famdata['SN']['index']
            values = _pyaccel.lattice.get_attribute(
                        self.models[mach], 'SL', index)
            zeros = _np.zeros(len(index))
            _pyaccel.lattice.set_attribute(
                self.models[mach], 'SL', index, zeros)

            step_data = dict()
            corr_sucess = False
            init_minsingval = self.original_minsingval
            while corr_sucess is not True:
                print('Initial singular value: {:.2f}'.format(init_minsingval))
                original_kicks = self.orbcorr.get_kicks()
                for step in range(nr_steps):
                    print('Step ', step+1)

                    self._apply_errors(nr_steps, mach)

                    # Orbit setted by BBA or setted to zero
                    if self.do_bba:
                        orb0_ = self._simulate_bba(nr_steps, step+1, mach)
                    else:
                        orb0_ = _np.zeros(2*len(self.bba_quad_idcs))

                    # Correct orbit
                    res = self._correct_orbit_iter(orb0_, mach,
                                                   init_minsingval)
                    if res is not False:
                        corr_sucess = True
                        orbf_, kicks_, corr_status, init_minsingval = res
                    else:
                        self._restore_error(step+1, nr_steps, mach)
                        self.orbcorr.set_kicks(original_kicks)
                        _pyaccel.lattice.set_attribute(
                            self.models[mach], 'SL', index, zeros)
                        init_minsingval += 0.05
                        corr_sucess = False
                        break

                    step_dict = dict()
                    step_dict['orbcorr_status'] = corr_status
                    step_dict['ref_orb'] = orb0_
                    step_dict['orbit'] = orbf_
                    step_dict['corr_kicks'] = kicks_
                    step_data['step_' + str(step + 1)] = step_dict

                    _pyaccel.lattice.set_attribute(
                        self.models[mach], 'SL', index,
                        (step + 1)*values/nr_steps)

                # Perform one last orbit correction after turning ON sextupoles
                res = self._correct_orbit_iter(orb0_, mach,
                                               init_minsingval)
                if res is not False:
                    corr_sucess = True
                    orbf_, kicks_, corr_status, init_minsingval = res
                else:
                    self._restore_error(step+1, nr_steps, mach)
                    self.orbcorr.set_kicks(original_kicks)
                    _pyaccel.lattice.set_attribute(
                            self.models[mach], 'SL', index, zeros)
                    init_minsingval += 0.05
                    corr_sucess = False

            # Save last orbit corr data
            step_dict = dict()
            step_dict['orbcorr_status'] = corr_status
            step_dict['ref_orb'] = orb0_
            step_dict['orbit'] = orbf_
            step_dict['corr_kicks'] = kicks_
            step_data['step_' + str(step + 2)] = step_dict

            # Do optics corrections:
            step_dict = step_data['step_' + str(step + 2)]
            if self.do_opt_corr:

                for i in range(1):
                    twiss, edtang, twiss0 = self._do_all_opt_corrections(
                        mach)

                dbetax = (twiss.betax - twiss0.betax)/twiss0.betax
                dbetay = (twiss.betay - twiss0.betay)/twiss0.betay
                step_dict['orbcorr_status'] = corr_status
                step_dict['ref_orb'] = orb0_
                step_dict['orbit'] = orbf_
                step_dict['corr_kicks'] = kicks_
                step_dict['twiss'] = twiss
                step_dict['edtang'] = edtang
                step_dict['betabeatingx'] = dbetax
                step_dict['betabeatingy'] = dbetay
                step_data['step_final'] = step_dict

            # Apply multipoles errors
            self._apply_multipoles_errors(1, mach)
            if self.corr_multipoles:

                twiss, edtang, twiss0 = self._do_all_opt_corrections(
                        mach)
                dbetax = (twiss.betax - twiss0.betax)/twiss0.betax
                dbetay = (twiss.betay - twiss0.betay)/twiss0.betay
                step_dict['orbcorr_status'] = corr_status
                step_dict['ref_orb'] = orb0_
                step_dict['orbit'] = orbf_
                step_dict['corr_kicks'] = kicks_
                step_dict['twiss'] = twiss
                step_dict['edtang'] = edtang
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
            if mach + 1 == self.nr_mach:
                break
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
            idcs = _np.array(
                self.famdata[id_.fam_name]['index']).ravel()
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
