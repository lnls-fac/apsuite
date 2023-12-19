"""Coupling Measurement from Minimal Tune Separation."""

import sys as _sys
import os as _os
import time as _time
import logging as _log

import numpy as _np
from scipy.optimize import least_squares
import matplotlib.pyplot as _plt

from siriuspy.devices import PowerSupply, Tune
from siriuspy.search import PSSearch as _PSSearch

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass

# _log.basicConfig(format=)
root = _log.getLogger()
root.setLevel(_log.INFO)

handler = _log.StreamHandler(_sys.stdout)
handler.setLevel(_log.INFO)
formatter = _log.Formatter('%(levelname)7s ::: %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


class CouplingParams(_ParamsBaseClass):
    """."""

    QUADS = (
        'QFB', 'QFA', 'QDB1', 'QDB2', 'QFP',
        'QDB1', 'QDB2', 'QDP1', 'QDP2', 'QDA',
        'Q1', 'Q2', 'Q3', 'Q4')

    def __init__(self):
        """."""
        super().__init__()
        self._quadfam_name = 'Q3'
        self.nr_points = 21
        self.time_wait = 5  # [s]
        self._lower_percent = 0.0/100
        self._upper_percent = 0.15/100
        self.coupling_resolution = 0.02/100

    @property
    def quadfam_name(self):
        """."""
        return self._quadfam_name

    @quadfam_name.setter
    def quadfam_name(self, val):
        """."""
        if isinstance(val, str) and val.upper() in self.QUADS:
            self._quadfam_name = val.upper()

    @property
    def lower_percent(self):
        """."""
        return self._lower_percent

    @lower_percent.setter
    def lower_percent(self, val):
        """."""
        self._lower_percent = val

    @property
    def upper_percent(self):
        """."""
        return self._upper_percent

    @upper_percent.setter
    def upper_percent(self, val):
        """."""
        self._upper_percent = val

    def __str__(self):
        """."""
        stmp = '{0:22s} = {1:4s}  {2:s}\n'.format
        ftmp = '{0:22s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:22s} = {1:9d}  {2:s}\n'.format
        stg = stmp('quadfam_name', self.quadfam_name, '')
        stg += dtmp('nr_points', self.nr_points, '')
        stg += ftmp('time_wait [s]', self.time_wait, '')
        stg += ftmp('lower_percent', self.lower_percent, '')
        stg += ftmp('upper_percent', self.upper_percent, '')
        stg += ftmp('coupling_resolution', self.coupling_resolution, '')
        return stg


class MeasCoupling(_BaseClass):
    """Coupling measurement and fitting.

    tunex = coeff1 * quad_parameter + offset1
    tuney = coeff2 * quad_parameter + offset2

    tune1, tune2 = Eigenvalues([[tunex, coupling/2], [coupling/2, tuney]])

    fit parameters: coeff1, offset1, coeff2, offset2, coupling

    NOTE: It maybe necessary to add a quadratic quadrupole strength
          dependency for tunes!
    """

    # Achromatic skew quadrupole variations to adjust betatron coupling,
    # it is -(v1 + v4)/np.sqrt(2) where v1 and v4 are the first and fourth
    # left-singular vectors of jacobian matrix calculated with
    # apsuite.optics.coupling_correction.calc_jacobian_matrix()
    ACHROM_QS_ADJ = 1e-4 * _np.array(
        [0.53393553, -1.83813959, -1.98690077, -1.75364868, -1.56156276,
         0.88701644,  1.14972317,  2.1831073,  0.53393553, -1.83813959,
         -1.98690077, -1.75364868, -1.56156276,  0.88701644,  1.14972317,
         2.1831073,  0.53393553, -1.83813959, -1.98690077, -1.75364868,
         -1.56156276,  0.88701644,  1.14972317,  2.1831073,  0.53393553,
         -1.83813959, -1.98690077, -1.75364868, -1.56156276,  0.88701644,
         1.14972317,  2.1831073,  0.53393553, -1.83813959, -1.98690077,
         -1.75364868, -1.56156276,  0.88701644,  1.14972317,  2.1831073])

    # Tune variation matrix for a relative change in quadrupole family strength
    # The values correspond to beta*KL/4\pi
    REL_DELTATUNE_QUADFAM = {
        'QFA': (10.298780521514438, -4.514923875173607),
        'QDA': (-1.0348691702091177, 3.5708035590970915),
        'QFB': (32.033111510608165, -17.107818311318557),
        'QDB1': (-1.8347622371355026, 9.987845128027336),
        'QDB2': (-7.368996560722064, 6.66243378337252),
        'QFP': (16.018344106829602, -8.554003388210424),
        'QDP1': (-0.9174526696494403, 4.993981381524282),
        'QDP2': (-3.68498307753575, 3.3312470390351985),
        'Q1': (12.603211153382881, -17.641037941914114),
        'Q2': (30.31621119877737, -21.69158623632497),
        'Q3': (11.519837506608473, -13.38406923471411),
        'Q4': (19.108756971042833, -7.102168831998462)}

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=CouplingParams(), target=self._do_meas, isonline=isonline)
        self.qs_names = None
        self.apply_factor = 0
        self.initial_strengths = None
        if self.isonline:
            self._create_devices()

    def get_initial_strengths(self):
        """."""
        ini_stren = [self.devices[name].strength for name in self.qs_names]
        return _np.array(ini_stren)

    def _create_skews(self):
        qs_names = _PSSearch.get_psnames({'sec': 'SI', 'dis': 'PS', 'dev': 'QS'})
        self.qs_names = [n for n in qs_names if n.sub.endswith('M1', 'M2')]
        for name in self.qs_names:
            self.devices[name] = PowerSupply(name)

    def _create_devices(self):
        self.devices['quad'] = PowerSupply(
                'SI-Fam:PS-' + self.params.quadfam_name)
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self._create_skews()

    def load_and_apply_old_data(self, fname):
        """Load and apply `data` and `params` from pickle or HDF5 file.

        Args:
            fname (str): name of the pickle file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be '.pickle' for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

        """
        data = self.load_data(fname)
        if 'data' in data:
            self.load_and_apply(fname)
            self._rename_old_params()
            return
        data['timestamp'] = _os.path.getmtime(fname)
        self.data = data

    def _rename_old_params(self):
        for attr, new_attr in \
            [('neg_percent', 'lower_percent'),
             ('pos_percent', 'upper_percent')]:
            if hasattr(self.params, attr):
                setattr(self.params, new_attr, getattr(self.params, attr))
                delattr(self.params, attr)

    def _do_meas(self):
        if not self.isonline:
            _log.error(
                'Cannot measure. Object is configured for offline studies.')
            return
        if self.devices['quad'].devname.dev != self.params.quadfam_name:
            self.devices['quad'] = PowerSupply(
                'SI-Fam:PS-' + self.params.quadfam_name)

        quad = self.devices['quad']
        tunes = self.devices['tune']
        quad.wait_for_connection()
        tunes.wait_for_connection()

        curr0 = quad.current
        curr_vec = curr0 * _np.linspace(
            1+self.params.lower_percent,
            1+self.params.upper_percent,
            self.params.nr_points)
        tunes_vec = _np.zeros((len(curr_vec), 2))

        _log.info(f'{quad.devname:s} current:')
        for idx, curr in enumerate(curr_vec):
            if self._stopevt.is_set():
                break
            quad.current = curr
            _time.sleep(self.params.time_wait)
            tunes_vec[idx, :] = tunes.tunex, tunes.tuney
            _log.info(
                f'{idx+1:02d}/{curr_vec.size:02d} --> ' +
                f'{curr:8.4f} A ({(curr/curr0-1)*100:+6.3f} %), ' +
                f'nux={tunes_vec[idx, 0]:6.4f}, nuy={tunes_vec[idx, 0]:6.4f}'
                )
        _log.info('Finished!')
        quad.current = curr0
        self.data['timestamp'] = _time.time()
        self.data['qname'] = quad.devname
        self.data['initial_current'] = curr0
        self.data['current'] = curr_vec
        self.data['tunes'] = tunes_vec

    def process_data(self):
        """."""
        self.analysis = dict()
        if not self.data:
            _log.error('There is no data to process.')
            return
        qcurr, tune1, tune2 = self._filter_data()
        self.analysis['qcurr'] = qcurr
        self.analysis['tune1'] = tune1
        self.analysis['tune2'] = tune2

        ini_param = self._calc_init_parms(qcurr, tune1, tune2)
        self.analysis['initial_param'] = ini_param

        # least squares using Levenberg-Marquardt minimization algorithm
        fit_param = least_squares(
            fun=self._err_func, x0=ini_param,
            args=(qcurr, tune1, tune2), method='lm')
        self.analysis['fitted_param'] = fit_param
        fit_error = self._calc_fitting_error()
        self.analysis['fitting_error'] = fit_error

    def plot_fitting(
            self, oversampling=10, save=False, fname=None):
        """."""
        anl = self.analysis
        fit_vec = anl['fitted_param']['x']
        qcurr, tune1, tune2 = anl['qcurr'], anl['tune1'], anl['tune2']

        fittune1, fittune2, qcurr_interp = self.get_normal_modes(
            params=fit_vec, curr=qcurr, oversampling=oversampling)

        fig, axi = _plt.subplots(1, 1, figsize=(8, 6))

        axi.set_xlabel(f'{self.data["qname"]} Current [A]')
        axi.set_ylabel('Transverse Tunes')
        fig.suptitle('Transverse Linear Coupling: ({:.2f} ± {:.2f}) %'.format(
            fit_vec[-1]*100, anl['fitting_error'][-1] * 100))

        # plot meas data
        axi.plot(qcurr, tune1, 'o', color='C0', label=r'$\nu_1$')
        axi.plot(qcurr, tune2, 'o', color='C1', label=r'$\nu_2$')

        # plot fitting
        axi.plot(qcurr_interp, fittune1, color='tab:gray', label='fitting')
        axi.plot(qcurr_interp, fittune2, color='tab:gray')
        axi.legend(loc='best')
        if save:
            if fname is None:
                date_string = _time.strftime("%Y-%m-%d-%H:%M")
                fname = 'coupling_fitting_{}.png'.format(date_string)
            fig.savefig(fname, format='png', dpi=300)
        fig.tight_layout()
        return fig, axi

    def apply_achromatic_delta_ksl(self, factor=0):
        """Change machine betatron coupling with achromatic QS.

        The variation is at the direction -(v1 + v4)/np.sqrt(2) where v1 and v4
        are the first and fourth left-singular vectors of the coupling
        jacobian matrix.

        Args:
            factor (int, optional): Scalar factor that sets the strenght of
            variation in the direction. The relation is typically 1:1 with
            respect to betatron coupling. Defaults to None.

        """
        dksl = factor * MeasCoupling.ACHROM_QS_ADJ

        if self.initial_strengths is None:
            self.initial_strengths = self.get_initial_strengths()

        for idx, name in enumerate(self.qs_names):
            ksl0 = self.initial_strengths[idx]
            self.devices[name].strength = ksl0 + dksl[idx]

    @staticmethod
    def calc_expected_delta_tunes(relative_dkl, quadfam='Q3'):
        """."""
        rel_dnux, rel_dnuy = MeasCoupling.REL_DELTATUNE_QUADFAM[quadfam]
        return (rel_dnux*relative_dkl, rel_dnuy*relative_dkl)

    @staticmethod
    def get_normal_modes(params, curr, oversampling=1):
        """Calculate the tune normal modes."""
        curr = MeasCoupling._oversample_vector(curr, oversampling)

        coeff1, offset1, coeff2, offset2, coupling = params
        fx_ = coeff1 * curr + offset1
        fy_ = coeff2 * curr + offset2
        coupvec = _np.ones(curr.size) * coupling/2
        mat = _np.array([[fx_, coupvec], [coupvec, fy_]])
        mat = mat.transpose((2, 0, 1))
        tune1, tune2 = _np.linalg.eigvalsh(mat).T
        sel = tune1 <= tune2
        tune1[sel], tune2[sel] = tune2[sel], tune1[sel]
        return tune1, tune2, curr

    # ------ Auxiliary methods --------
    def _filter_data(self):
        qcurr = _np.asarray(self.data['current'])
        tune1, tune2 = self.data['tunes'].T
        tune1 = _np.asarray(tune1)
        tune2 = _np.asarray(tune2)

        # sort tune1 > tune2 at each point
        sel = tune1 <= tune2
        if sel.any():
            tune1[sel], tune2[sel] = tune2[sel], tune1[sel]

        # remove nearly degenerate measurement points
        dtunes = _np.abs(tune1 - tune2)
        sel = _np.where(dtunes < self.params.coupling_resolution)[0]
        qcurr = _np.delete(qcurr, sel)
        tune1 = _np.delete(tune1, sel)
        tune2 = _np.delete(tune2, sel)
        return qcurr, tune1, tune2

    def _calc_fitting_error(self):
        # based on fitting error calculation of scipy.optimization.curve_fit
        # do Moore-Penrose inverse discarding zero singular values.
        fit_params = self.analysis['fitted_param']
        _, smat, vhmat = _np.linalg.svd(
            fit_params['jac'], full_matrices=False)
        thre = _np.finfo(float).eps * max(fit_params['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[:smat.size]
        pcov = _np.dot(vhmat.T / (smat*smat), vhmat)

        # multiply covariance matrix by residue 2-norm
        ysize = len(fit_params['fun'])
        cost = 2 * fit_params['cost']  # res.cost is half sum of squares!
        popt = fit_params['x']
        if ysize > popt.size:
            # normalized by degrees of freedom
            s_sq = cost / (ysize - popt.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(_np.nan)
            _log.warning(
                '# of fitting parameters larger than # of data points!')
        return _np.sqrt(_np.diag(pcov))

    @staticmethod
    def _err_func(params, qcurr, tune1, tune2):
        tune1f, tune2f, _ = MeasCoupling.get_normal_modes(params, qcurr)
        return _np.sqrt((tune1f - tune1)**2 + (tune2f - tune2)**2)

    @staticmethod
    def _calc_init_parms(curr, tune1, tune2):
        nu_beg = tune1[0], tune2[0]
        nu_end = tune1[-1], tune2[-1]
        dcurr = curr[-1] - curr[0]
        # estimative based on
        # nux = coeff1 * curr + offset1
        # nuy = coeff2 * curr + offset2
        coeff1 = (min(nu_end) - max(nu_beg)) / dcurr
        offset1 = max(nu_beg) - coeff1 * curr[0]
        coeff2 = (max(nu_end) - min(nu_beg)) / dcurr
        offset2 = min(nu_beg) - coeff2 * curr[0]
        coupling = min(_np.abs(tune1 - tune2))
        return [coeff1, offset1, coeff2, offset2, coupling]

    @staticmethod
    def _oversample_vector(vec, oversampling):
        oversampling = int(oversampling)
        if oversampling <= 1:
            return vec
        siz = len(vec)
        x_over = _np.linspace(0, siz-1, (siz-1)*oversampling+1)
        x_data = _np.arange(siz)
        return _np.interp(x_over, x_data, vec)
