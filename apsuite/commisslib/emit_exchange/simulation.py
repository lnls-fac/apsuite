"""Scripts to study and simulate the emittance exchange in the Sirius Booster
"""
import numpy as _np
import pyaccel as _pa
import pymodels as _pm
import matplotlib.pyplot as _plt


class EmittanceExchangeSimul:
    """Class used to simulate the emittance exchange process in the Sirius Booster
    """
    BO_REV_PERIOD = 1.657e-3  # [m s]

    def __init__(
            self, init_delta=None, c=0.01, radiation=True, s=1,
            quad='QF'):
        """."""
        self._model = _pm.bo.create_accelerator(energy=3e9)
        self._model.vchamber_on = True
        self._model.cavity_on = True
        self._model.radiation_on = radiation

        self._radiation = radiation
        self._tune_crossing_vel = s
        self._init_delta = init_delta
        self._quad = quad
        self._coupling_coeff = None
        self._K_list = None
        self._tunes = None
        self._envelopes = None
        self._emittances = None
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
        """."""
        return self._K_list

    @property
    def quad_idxs(self):
        """."""
        self._quad_idxs = _pa.lattice.find_indices(
            lattice=self.model, attribute_name='fam_name', value=self._quad)
        return self._quad_idxs

    @property
    def qs_idxs(self):
        """."""
        self._qs_idxs = _pa.lattice.find_indices(
            lattice=self.model, attribute_name='fam_name', value='QS')
        return self._qs_idxs

    @property
    def model(self):
        """."""
        return self._model

    @property
    def coupling_coeff(self):
        """."""
        self._coupling_coeff = self.calc_coupling(self.model)
        return self._coupling_coeff

    @property
    def tune_crossing_vel(self):
        """."""
        return self._tune_crossing_vel

    @property
    def tunes(self):
        """."""
        return self._tunes

    @property
    def deltas(self):
        """."""
        tunes_diff = self.tunes[0] - self.tunes[1]
        c = _np.min(_np.abs(tunes_diff))
        return _np.sign(tunes_diff) * _np.sqrt(tunes_diff**2 - c**2)

    @property
    def envelopes(self):
        """."""
        return self._envelopes

    @property
    def emittances(self):
        """."""
        return self._emittances

    def env_emit_exchange(self, verbose=True, store_env=False, indices=[0]):
        """Simulates the dynamic process of emittance exchange using beam
        envelope tracking.

        Parameters
        ----------
        verbose : bool, optional;
            If True, there will be printed the remaining steps to conclude the
            simulation, by default True.
        store_env : bool, optional;
            If True, stores the beam envelopes in self.envelopes.
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

            if store_env:
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

    def plot_exchange(self, fname=None):
        """."""
        r, best_r_idx = self.calc_exchange_quality()
        fig, ax = _plt.subplots(figsize=(6, 3))
        emitx, emity = self.emittances[0], self.emittances[1]
        deltas = self.deltas

        ax.plot(deltas, emitx*1e9, lw=3, label=r'$\epsilon_x \;[nm]$')
        ax.plot(deltas, emity*1e9, lw=3, label=r'$\epsilon_y \;[nm]$')
        ax.axvline(
            deltas[best_r_idx],
            label=f'max(R) = {r[best_r_idx]*100:.2f} [%]', ls='--',
            c='k', lw='3')
        ax.set_xlabel(r'$\Delta$')
        ax.set_ylabel(r'$\epsilon \; [nm]$')
        secax = ax.secondary_xaxis(
            'top', functions=(self._delta2time, self._time2delta))
        secax.set_xlabel('Time [ms]')
        ax.legend(loc='best')
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname, format='png', dpi=300, facecolor='white',
                        transparent=False)
        _plt.show()
        return fig, ax

    def calc_exchange_quality(self):
        """."""
        emit1_0 = self.emittances[0, 0]
        emit2_0 = self.emittances[1, 0]
        emit1 = self.emittances[0]
        r = 1 - (emit1 - emit2_0)/(emit1_0 - emit2_0)
        best_r_idx = _np.argmax(r)
        return r, best_r_idx

    @staticmethod
    def calc_coupling(model):
        """."""
        ed_teng, _ = _pa.optics.calc_edwards_teng(model)
        coupling_coeff, _ = _pa.optics.estimate_coupling_parameters(ed_teng)
        return coupling_coeff

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
        """Calculate normal modes fractional tunes

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

    def _set_coupling(self, c):
        """."""
        ksl = self._c_to_ksl(c)

        _pa.lattice.set_attribute(
            lattice=self._model, attribute_name='KsL', values=ksl,
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

    def _find_ksl_response(self):
        mod = self.model[:]
        ksl = 0.01
        _pa.lattice.set_attribute(
            lattice=mod, attribute_name='KsL', values=ksl,
            indices=self.qs_idxs)
        c = self.calc_coupling(mod)
        return c/ksl

    def _c_to_ksl(self, C):
        """."""
        ksl2c = self._find_ksl_response()
        KsL = C/ksl2c
        return KsL

    def _ksl_to_c(self, KsL):
        """."""
        ksl2c = self._find_ksl_response()
        c = KsL*ksl2c
        return c

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
