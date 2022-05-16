"""Scripts to study and simulate the emittance exchange in the Sirius Booster
"""
import numpy as _np
import time as _time
import pyaccel as _pa
import pymodels as _pm
from numpy.random import default_rng as _default_rng
import matplotlib.pyplot as _plt


class EmittanceExchangeSimul:
    """Class used to simulate the emittance exchange process in the Sirius Booster
    """
    BO_REV_PERIOD = 1.657e-3  # [m s]

    def __init__(
            self, init_delta=None, c=0.01, radiation=True, s=1, indices=[0],
            quad='QF'):
        """."""
        self._model = _pm.bo.create_accelerator(energy=3e9)
        self._model.vchamber_on = True
        self._model.cavity_on = True
        self._model.radiation_on = radiation
        self._radiation = radiation
        self._tune_crossing_vel = s
        self._init_delta = init_delta
        self._indices = indices
        self._quad = quad
        self._coupling_coeff = None
        self._K_list = None
        self._tunes = None
        self._envelopes = None
        self._emittances = None
        self._exchange_quality = None
        self._bunch = None
        self._quad_idxs = None
        self._qs_idxs = None

        self._KL_default = _pa.lattice.get_attribute(
            lattice=self.model, attribute_name='KL',
            indices=self.quad_idxs[0])[0]

        self._KS_default = _pa.lattice.get_attribute(
            lattice=self.model, attribute_name='KsL',
            indices=self.qs_idxs[0])[0]

        if init_delta is not None:
            self._set_initial_delta(init_delta)
        else:
            self._init_delta = self._calc_delta()

        if c:
            self._set_coupling(c=c)

        self._init_env = _pa.optics.calc_beamenvelope(
            accelerator=self.model, indices=[0], full=False)

    @property
    def K_list(self):
        return self._K_list

    @property
    def quad_idxs(self):
        self._quad_idxs = _pa.lattice.find_indices(
            lattice=self.model, attribute_name='fam_name', value=self._quad)
        return self._quad_idxs

    @property
    def qs_idxs(self):
        self._qs_idxs = _pa.lattice.find_indices(
            lattice=self.model, attribute_name='fam_name', value='QS')
        return self._qs_idxs

    @property
    def model(self):
        return self._model

    @property
    def bunch(self):
        return self._bunch

    @property
    def coupling_coeff(self):
        ed_teng, _ = _pa.optics.calc_edwards_teng(self.model)
        self._coupling_coeff, _ = _pa.optics.estimate_coupling_parameters(
            ed_teng)
        return self._coupling_coeff

    @property
    def tune_crossing_vel(self):
        return self._tune_crossing_vel

    @property
    def tunes(self):
        return self._tunes

    @property
    def deltas(self):
        tunes_diff = self.tunes[0] - self.tunes[1]
        c = _np.min(_np.abs(tunes_diff))
        return _np.sign(tunes_diff) * _np.sqrt(tunes_diff**2 - c**2)

    @property
    def envelopes(self):
        return self._envelopes

    @property
    def emittances(self):
        return self._emittances

    @property
    def exchange_quality(self):
        if self.emittances is not None:
            self._exchange_quality = self.calc_exchange_quality()
        else:
            raise AttributeError("Emittance exchange wasn't simulated yet.")
        return self._exchange_quality

    def experimental_params(self, final_delta):
        """Estimate emittance exchange experimental parameters based on the
        Booster nominal model.

        Parameters
        ----------
        final_delta : float with opposite sign of self.initial_delta.
            Tune difference whose the beam is extracted.
        """
        c = self.coupling_coeff
        s = self.tune_crossing_vel
        Tr = self.BO_REV_PERIOD
        dtune_dt = s*c**2/Tr
        l_quad = 2*_pa.lattice.get_attribute(
            self.model, 'length', indices=self.quad_idxs[0])[0]
        dkl_dt = self._calc_dk(dtune_dt) * l_quad
        tt = (_np.abs(self._init_delta) + _np.abs(final_delta))/dtune_dt
        dc_dksL = self._ksl_to_c(1)

        ftmp = '{0:24s} = {1:9.5f}  {2:s}\n'.format

        stg = ''
        stg += ftmp('d(tune)/dt', dtune_dt, '[1/ms]')
        stg += ftmp('dKL/dt', dkl_dt, '[1/(m . ms)]')
        stg += ftmp('Exchange time', tt, '[ms]')
        stg += ftmp('dC/dKsL', dc_dksL, '[m]')
        print(stg)

    def generate_bunch(self, n_part=1e3):
        """Generates the bunch used in the particle simulations.
        Not necessary to run if you would use envelope tracking instead
        particle tracking.

        Parameters
        ----------
        n_part : int
            Number of particles, by default, 1e3.
        """
        init_env = self._init_env
        self._bunch = _pa.tracking.generate_bunch(
            n_part=n_part, envelope=init_env[0])

    def env_emit_exchange(self, verbose=True, indices=[0]):
        """Simulates the dynamic process of emittance exchange using beam
        envelope tracking.

        Parameters
        ----------
        verbose : bool, optional;
            If True, there will be printed the remaining steps to conclude the
            simulation, by default True.
        indices : list, optional;
            Indices in which there will be stored the envelopes, by default [0]
        """
        s = self.tune_crossing_vel
        c = self.coupling_coeff
        delta = self._calc_delta()
        n_turns = int(_np.abs(delta)/(s * c**2))
        print("---------------------Tracking particles----------------------\n"
              "Initial delta = {:.3f} \n N = {}\n".format(delta, 2*n_turns),
              "C={:.3f}[%], S={:.3f}".format(c*1e2, s))
        quad_idx = self.quad_idxs
        K_default = self.model[quad_idx[0]].K
        if self._quad == 'QF':
            dK = self._calc_dk(-delta)
        else:
            dK = self._calc_dk(delta)
        K_list = _np.linspace(K_default, K_default + 2*dK, 2*n_turns)
        self._K_list = K_list
        emittances = _np.zeros([2, K_list.size])
        envelopes = _np.zeros([K_list.size, 6, 6])
        tunes = emittances.copy()

        env0 = self._init_env

        for i, k in enumerate(K_list):

            _pa.lattice.set_attribute(
                lattice=self._model, attribute_name='K', values=k,
                indices=quad_idx)

            env0 = _pa.optics.calc_beamenvelope(
                accelerator=self._model, init_env=env0[-1],
                indices='closed', full=False)

            envelopes[i] = env0[indices]
            emittances[:, i] = self._calc_envelope_emittances(env0[-1])
            tunes[:, i] = self._calc_nm_tunes()

            if verbose:
                if i % 100 == 0:
                    print(f"step {i}", end='\t')
                if i % 500 == 0:
                    print('\n')
                if i == K_list.size-1:
                    print('Done!')

        self._envelopes = envelopes
        self._emittances = emittances
        self._tunes = tunes

    def part_emit_exchange(
            self, verbose=True):
        """Simulates the dynamic process of emittance exchange using particles
        tracking.

        Parameters
        ----------
        verbose : bool, optional;
            If True, there will be printed the remaining steps to conclude the
            simulation, by default True.
        """
        s = self.tune_crossing_vel
        c = self.coupling_coeff
        delta = self._calc_delta()
        n_turns = int(_np.abs(delta)/(s * c**2))    # Turns until the resonance
        rng = _default_rng()                        # crossing

        print("---------------------Tracking particles----------------------\n"
              "Initial delta = {:.3f} \n N = {}\n".format(delta, 2*n_turns),
              "C={:.3f}[%], S={:.3f}".format(c*1e2, s))

        quad_idx = self.quad_idxs
        K_default = self.model[quad_idx[0]].K
        if self._quad == 'QF':
            dK = self._calc_dk(-delta)
        else:
            dK = self._calc_dk(delta)
        K_list = _np.linspace(K_default, K_default + 2*dK, 2*n_turns)
        self._K_list = K_list

        emittances = _np.zeros([2, K_list.size])
        tunes = emittances.copy()

        bunch0 = self.bunch
        npart = bunch0.shape[1]
        env0 = self._init_env

        for i, K in enumerate(K_list):

            # Changing quadrupole forces
            _pa.lattice.set_attribute(
                lattice=self._model, attribute_name='K', values=K,
                indices=quad_idx)

            if self._radiation:
                # Tracking with quantum excitation
                env0, cum_mat, bdiff, _ = _pa.optics.calc_beamenvelope(
                    accelerator=self._model, init_env=env0[-1],
                    indices='closed', full=True)
                bunch0 = _np.dot(cum_mat[-1], bunch0)

                bunch_excit = rng.multivariate_normal(
                    mean=_np.zeros(6), cov=bdiff[-1],
                    method='cholesky', size=npart).T
                bunch0 += bunch_excit
            else:
                m = _pa.tracking.find_m66(self.model)
                bunch0 = _np.dot(m, bunch0)

            self._bunch = bunch0
            emittances[:, i] = self._calc_emittances()
            tunes[:, i] = self._calc_nm_tunes()

            if verbose:
                if i % 100 == 0:
                    print(f"step {i}", end='\t')
                if i % 500 == 0:
                    print('\n')
                if i == K_list.size-1:
                    print('Done!')

        self._emittances = emittances
        self._tunes = tunes

    def plot_exchange(self, save=False, fname=None):
        """."""
        r, best_r_idx = self._calc_exchange_quality()
        fig, ax = _plt.subplots(1, 1, figsize=(6, 4.5))
        emitx, emity = self.emittances[0], self.emittances[1]
        deltas = self.deltas

        ax.plot(deltas, emitx*1e9, lw=3, label=r'$\epsilon_x \;[nm]$')
        ax.plot(deltas, emity*1e9, lw=3, label=r'$\epsilon_y \;[nm]$')
        ax.axvline(
            deltas[best_r_idx],
            label=f'max(R) = {r[best_r_idx]*100:.2f} [\\%]', ls='--',
            c='k', lw='3')
        ax.set_xlabel(r'$\Delta$')
        ax.set_ylabel(r'$\epsilon \; [nm]$')
        secax = ax.secondary_xaxis(
            'top', functions=(self._delta2time, self._time2delta))
        secax.set_xlabel('Time [ms]')
        ax.legend(loc='best')
        fig.tight_layout()
        if save:
            if fname is None:
                date_string = _time.strftime("%Y-%m-%d-%H:%M")
                fname = 'emit_exch_simul_{}.png'.format(date_string)
            fig.savefig(fname, format='png', dpi=300)
        _plt.show()
        return fig, ax

    def _calc_delta(self):
        """."""
        twi, *_ = _pa.optics.calc_twiss(self.model)
        tunex = twi.mux[-1]/(2*_np.pi)
        tuney = twi.muy[-1]/(2*_np.pi)
        tunex %= 1
        tuney %= 1
        if tunex > 0.5:
            tunex = _np.abs(1-tunex)
        if tuney > 0.5:
            tuney = _np.abs(1-tuney)
        return tunex - tuney

    def _calc_dk(self, dtune):
        """."""
        if self._quad == 'QF':
            sign = 1
        elif self._quad == 'QD':
            sign = -1
        else:
            raise ValueError('Wrong quadrupole type')
        sum_beta_l, _ = self._calc_sum_beta_l()
        deltaK = sign*4*_np.pi*dtune/(sum_beta_l)
        return deltaK

    def _calc_nm_tunes(self):
        """Calculate normal modes tunes

        Returns
        -------
        np.array of size 2
            Vector with the tunes
        """
        ed_teng, _ = _pa.optics.calc_edwards_teng(self.model)
        tune1 = ed_teng.mu1[-1]/(2*_np.pi)
        tune2 = ed_teng.mu2[-1]/(2*_np.pi)
        tune1 %= 1
        tune2 %= 1
        if tune1 > 0.5:
            tune1 = _np.abs(1-tune1)
        if tune2 > 0.5:
            tune2 = _np.abs(1-tune2)
        return _np.array([tune1, tune2])

    def _calc_envelope_emittances(self, env):
        """."""
        twi, *_ = _pa.optics.calc_twiss(self.model)
        etax, etapx = twi.etax[0], twi.etapx[0]
        etay, etapy = twi.etay[0], twi.etapy[0]
        disp = _np.array([[etax], [etapx], [etay], [etapy], [0], [0]])
        env_nodisp = env - env[4, 4]*_np.dot(disp, disp.T)
        emitx = _np.sqrt(_np.linalg.det(env_nodisp[:2, :2]))
        emity = _np.sqrt(_np.linalg.det(env_nodisp[2:4, 2:4]))
        emittances_array = _np.array([emitx, emity])
        return emittances_array

    def _calc_emittances(self):
        """."""
        twi, *_ = _pa.optics.calc_twiss(self.model)
        bunch0 = self.bunch
        etax, etapx = twi.etax[0], twi.etapx[0]
        etay, etapy = twi.etay[0], twi.etapy[0]
        disp = _np.array([[etax], [etapx], [etay], [etapy], [0], [0]])
        bunch_nodisp = \
            bunch0 - bunch0[4, :]*disp - _np.mean(bunch0, axis=1)[:, None]
        emitx = _np.sqrt(_np.linalg.det(_np.cov(bunch_nodisp[:2, :])))
        emity = _np.sqrt(_np.linalg.det(_np.cov(bunch_nodisp[2:4, :])))
        emittances_array = _np.array([emitx, emity])
        return emittances_array

    def _calc_exchange_quality(self):
        """."""
        emit1_0 = self.emittances[0, 0]
        emit2_0 = self.emittances[1, 0]
        emit1 = self.emittances[0]
        r = 1 - (emit1 - emit2_0)/(emit1_0 - emit2_0)
        best_r_idx = _np.argmax(r)
        return r, best_r_idx

    def _set_coupling(self, c):
        """."""
        # NOTE: Change this method to set the coupling by minimization and
        # then set a scale factor to convert C to Ks and vice-versa.
        c0 = self.coupling_coeff
        dc = c - c0
        dksl = self._c_to_ksl(dc)

        ksl0 = _pa.lattice.get_attribute(
            lattice=self.model, attribute_name='KsL',
            indices=self.qs_idxs)

        _pa.lattice.set_attribute(
            lattice=self._model, attribute_name='KsL', values=ksl0+dksl,
            indices=self.qs_idxs)

    def _set_initial_delta(self, init_delta):
        "Sets delta to delta_f"
        delta = self._calc_delta()
        quad_idx = self.quad_idxs
        sum_beta_l, _ = self._calc_sum_beta_l()
        dv_x = init_delta - delta
        dk_x = 4*_np.pi*dv_x/(sum_beta_l)
        k_x = self.model[quad_idx[0]].K + dk_x
        _pa.lattice.set_attribute(
            lattice=self._model, attribute_name='K', indices=quad_idx,
            values=k_x)

    def _c_to_ksl(self, C):
        """."""
        fam_data = _pm.bo.get_family_data(self.model)
        qs_idx = fam_data['QS']['index']
        ed_tang, *_ = _pa.optics.calc_edwards_teng(accelerator=self.model)
        beta1 = ed_tang.beta1[qs_idx[0]]
        beta2 = ed_tang.beta2[qs_idx[0]]
        KsL = -2 * _np.pi * C / _np.sqrt(beta1 * beta2)

        return KsL[0]

    def _ksl_to_c(self, KsL):
        """."""
        fam_data = _pm.bo.get_family_data(self.model)
        qs_idx = fam_data['QS']['index']
        ed_tang, *_ = _pa.optics.calc_edwards_teng(accelerator=self.model)
        beta1 = ed_tang.beta1[qs_idx[0][0]]
        beta2 = ed_tang.beta2[qs_idx[0][0]]
        C = _np.abs(KsL * _np.sqrt(beta1 * beta2)/(2 * _np.pi))

        return C

    def _calc_sum_beta_l(self):
        """."""
        ed_teng, _ = _pa.optics.calc_edwards_teng(self.model)
        quad_idx = self.quad_idxs
        betax = ed_teng.beta1
        betay = ed_teng.beta2
        betasx = _np.zeros(len(quad_idx))
        betasy = betasx.copy()
        length = betasx.copy()

        if self._quad == 'QF':
            for i in range(0, len(quad_idx), 2):
                idx1, idx2 = quad_idx[i], quad_idx[i+1]
                betay_values = betay[[idx1, idx2, idx2+1]]
                betax_values = betax[[idx1, idx2, idx2+1]]
                length[i] = self.model[quad_idx[i]].length + \
                    self.model[quad_idx[i+1]].length
                betasx[i] = _np.mean(betax_values)
                betasy[i] = _np.mean(betay_values)
        else:
            for i in range(0, len(quad_idx)):
                idx = quad_idx[i]
                betay_values = betay[[idx, idx+1]]
                betax_values = betax[[idx, idx+1]]
                length[i] = self.model[quad_idx[i]].length
                betasx[i] = _np.mean(betax_values)
                betasy[i] = _np.mean(betay_values)

        sum_beta_l = _np.sum(length*(betasx + betasy))

        return sum_beta_l, length

    def _delta2time(self, delta):
        """."""
        s = self._tune_crossing_vel
        c = self.coupling_coeff
        tr = self.BO_REV_PERIOD
        time = tr*delta/(s*c**2)
        # time = time - _np.min(time)
        return time

    def _time2delta(self, time):
        """."""
        s = self._tune_crossing_vel
        c = self.coupling_coeff
        tr = self.BO_REV_PERIOD
        return time*s*c**2/tr
