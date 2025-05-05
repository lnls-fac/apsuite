"""."""

from copy import deepcopy as _dcopy
from mathphys.functions import get_namedtuple as _get_namedtuple
import numpy as _np

import pyaccel

from .calc_orbcorr_mat import OrbRespmat


class CorrParams:
    """."""

    RESPMAT_MODE = _get_namedtuple('RespMatMode', ['Full', 'Mxx', 'Myy'])

    def __init__(self):
        """."""
        # The most restrictive of the two below will be the limiting factor:
        self.minsingval = 0.2
        self.numsingval = 500
        self.tikhonovregconst = 0  # Tikhonov regularization constant
        self.respmatmode = self.RESPMAT_MODE.Full
        self.respmatrflinemult = 1e6  # Mult. factor of RF line in SVD.
        self.enblrf = True
        self.enbllistbpm = None
        self.enbllistch = None
        self.enbllistcv = None

        self.maxkickch = 300e-6  # rad
        self.maxkickcv = 300e-6  # rad
        self.maxkickrf = 500e6  # Hz
        self.maxdeltakickch = 300e-6  # rad
        self.maxdeltakickcv = 300e-6  # rad
        self.maxdeltakickrf = 1000  # Hz
        self.corrgainch = 1.0  # Gains applied to correctors delta kicks.
        self.corrgaincv = 1.0
        self.corrgainrf = 1.0

        self.maxnriters = 10
        self.convergencetol = 0.5e-6  # m  Define the convergence criteria
        self.orbrmswarnthres = 1e-6  # m  Orb. threshold to trigger warning.
        self.useglobalcoef = False  # Use same kick factor for all correctors.
        self.updatejacobian = False  # jacobian should update in all iterations

        self.use6dorb = False


class OrbitCorr:
    """."""

    CORR_STATUS = _get_namedtuple(
        'CorrStatus',
        ['Sucess', 'OrbRMSWarning', 'ConvergenceFail', 'SaturationFail'])

    def __init__(self, model, acc, params=None, corr_system='SOFB'):
        """."""
        self.acc = acc
        self.params = params or CorrParams()
        dim = '6d' if self.params.use6dorb else '4d'
        self.respm = OrbRespmat(
            model=model, acc=self.acc, dim=dim, corr_system=corr_system)
        self.params.enbllistbpm = _np.ones(
            self.respm.bpm_idx.size*2, dtype=bool)
        if corr_system == 'FOFB':
            enbllistbpm = self.params.enbllistbpm.reshape(40, -1)
            enbllistbpm[:, [1, 2, 5, 6]] = False
            self.params.enbllistbpm = enbllistbpm.ravel()
        elif corr_system == 'SOFB':
            pass
        else:
            raise ValueError('Corretion system must be "SOFB" or "FOFB"')
        if self.params.enblrf and model.cavity_on is False:
            raise Exception("It is necessary to turn the cavity on if it is"
                            " to be used in correction..")

        self.params.enbllistch = _np.ones(
            self.respm.ch_idx.size, dtype=bool)
        self.params.enbllistcv = _np.ones(
            self.respm.cv_idx.size, dtype=bool)

    def get_jacobian_matrix(self):
        """."""
        return self.respm.get_respm()

    def get_kicks(self):
        """Return corrector kicks."""
        model = self.respm.model
        chidx = self.respm.ch_idx
        cvidx = self.respm.cv_idx
        rfidx = self.respm.rf_idx
        kickch = pyaccel.lattice.get_attribute(model, 'hkick_polynom', chidx)
        kickcv = pyaccel.lattice.get_attribute(model, 'vkick_polynom', cvidx)
        kickrf = pyaccel.lattice.get_attribute(model, 'frequency', rfidx)
        return _np.r_[kickch, kickcv, kickrf]

    @staticmethod
    def get_figm(res):
        """Calculate figure of merit from residue vector."""
        return _np.sqrt(_np.sum(res*res)/res.size)

    def get_inverse_matrix(self, jacobian_matrix, full=False):
        """Calculate the pseudo-inverse of jacobian_matrix.

        Args:
            jacobian_matrix (numpy.ndarray, (N, M)): Jacobian matrix.
            full (bool, optional): Whether or not to return the SVD
                decomposition with the processed singular values.
                Defaults to False.

        Raises:
            ValueError: If there is some problem with the calculation.

        Returns:
            inv_mat (numpy.ndarray, (M, N)): pseudo inverse.
            u (numpy.ndarray, (N, min(N, M)), optional): U matrix.
            s (numpy.ndarray, (min(N, M), ), optional): Processed sing. values.
            vt (numpy.ndarray, (min(N, M), M), optional): Transpose of V matrix

        """
        par = self.params

        enblbpms = par.enbllistbpm
        enblcorrs = _np.r_[par.enbllistch, par.enbllistcv, par.enblrf]

        if not _np.any(enblbpms):
            raise ValueError('No BPM selected in enbllistbpm')
        if not _np.any(enblcorrs):
            raise ValueError('No Corrector selected in enbllist')

        sel_mat = enblbpms[:, None] * enblcorrs[None, :]
        if sel_mat.size != jacobian_matrix.size:
            raise ValueError('Incompatible size between jacobian and enbllist')

        mat = jacobian_matrix.copy()
        mat[:, -1] *= par.respmatrflinemult
        nr_bpms = len(par.enbllistbpm)
        nr_ch = len(par.enbllistch)
        nr_chcv = nr_ch + len(par.enbllistcv)
        if par.respmatmode != par.RESPMAT_MODE.Full:
            mat[:nr_bpms, nr_ch:nr_chcv] = 0
            mat[nr_bpms:, :nr_ch] = 0
            mat[nr_bpms:, nr_chcv:] = 0
        if par.respmatmode == par.RESPMAT_MODE.Mxx:
            mat[nr_bpms:] = 0
        elif par.respmatmode == par.RESPMAT_MODE.Myy:
            mat[:nr_bpms] = 0

        mat = mat[sel_mat]
        mat = _np.reshape(mat, [_np.sum(enblbpms), _np.sum(enblcorrs)])
        uuu, sing, vvv = self.calc_svd(mat)

        idcs = sing > par.minsingval
        idcs[par.numsingval:] = False
        singr = sing[idcs]
        nr_sv = _np.sum(idcs)
        if not nr_sv:
            raise ValueError('All Singular Values below minimum.')

        # Apply Tikhonov regularization:
        regc = par.tikhonovregconst
        inv_s = _np.zeros(sing.size, dtype=float)
        inv_s[idcs] = singr/(singr*singr + regc*regc)

        # Calculate Inverse
        imat = _np.dot(vvv.T*inv_s, uuu.T)
        is_nan = _np.any(_np.isnan(imat))
        is_inf = _np.any(_np.isinf(imat))
        if is_nan or is_inf:
            raise ValueError('Inverse contains nan or inf.')
        inv_mat = _np.zeros(jacobian_matrix.shape[::-1], dtype=float)
        inv_mat[sel_mat.T] = imat.ravel()
        inv_mat[-1, :] *= par.respmatrflinemult
        if not full:
            return inv_mat

        # Calculate processed singular values
        singp = _np.zeros(sing.size, dtype=float)
        singp[idcs] = 1/inv_s[idcs]
        sing_vals = _np.zeros(min(*inv_mat.shape), dtype=float)
        sing_vals[:singp.size] = singp
        return inv_mat, uuu, singp, vvv

    def calc_svd(self, mat):
        """Calculate truncated SVD of a matrix.

        Args:
            mat (numpy.ndarray, (N, M)): 2D array to decompose.

        Raises:
            ValueError: When SVD decomposition is not successful.

        Returns:
            u (numpy.ndarray, (N, min(N, M))): U matrix of decomposition
            s (numpy.ndarray, (min(N, M), )): Singular values
            vt (numpy.ndarray, (min(N, M), M)): Transpose of V matrix

        """
        try:
            return _np.linalg.svd(mat, full_matrices=False)
        except _np.linalg.LinAlgError():
            raise ValueError('Could not calculate SVD of jacobian')

    def correct_orbit(self, jacobian_matrix=None, goal_orbit=None):
        """Orbit correction.

        Calculates the pseudo-inverse of orbit correction matrix via SVD
        and minimizes the residue vector [CODx@BPM, CODy@BPM].
        """
        if goal_orbit is None:
            nbpm = len(self.respm.bpm_idx)
            goal_orbit = _np.zeros(2 * nbpm, dtype=float)

        jmat = jacobian_matrix
        if jmat is None:
            jmat = self.get_jacobian_matrix()

        ismat = self.get_inverse_matrix(jmat)

        orb = self.get_orbit()
        dorb = orb - goal_orbit
        bestfigm = OrbitCorr.get_figm(dorb)
        for _ in range(self.params.maxnriters):
            dkicks = -1*_np.dot(ismat, dorb)
            kicks, saturation_flag = self._process_kicks(dkicks)
            if saturation_flag:
                return OrbitCorr.CORR_STATUS.SaturationFail
            self.set_kicks(kicks)
            orb = self.get_orbit()
            dorb = orb - goal_orbit
            figm = OrbitCorr.get_figm(dorb)
            diff_figm = _np.abs(bestfigm - figm)
            if figm < bestfigm:
                bestfigm = figm
            if diff_figm < self.params.convergencetol:
                if bestfigm <= self.params.orbrmswarnthres:
                    return OrbitCorr.CORR_STATUS.Sucess
                else:
                    return OrbitCorr.CORR_STATUS.OrbRMSWarning
            if self.params.updatejacobian:
                jmat = self.get_jacobian_matrix()
                ismat = self.get_inverse_matrix(jmat)
        return OrbitCorr.CORR_STATUS.ConvergenceFail

    def get_orbit(self):
        """."""
        if self.params.use6dorb:
            cod = pyaccel.tracking.find_orbit6(
                self.respm.model, indices='open')
        else:
            cod = pyaccel.tracking.find_orbit4(
                self.respm.model, indices='open')
        codx = cod[0, self.respm.bpm_idx].ravel()
        cody = cod[2, self.respm.bpm_idx].ravel()
        res = _np.r_[codx, cody]
        return res

    def set_delta_kicks(self, dkicks):
        """."""
        kicks = self.get_kicks()
        kicks += dkicks
        self.set_kicks(kicks)

    def set_kicks(self, kicks):
        """."""
        model = self.respm.model
        nch = len(self.respm.ch_idx)
        ncv = len(self.respm.cv_idx)

        kickch, kickcv, kickrf = kicks[:nch], kicks[nch:nch+ncv], kicks[-1]

        for i, idx in enumerate(self.respm.ch_idx):
            model[idx].hkick_polynom = kickch[i]
        for i, idx in enumerate(self.respm.cv_idx):
            model[idx].vkick_polynom = kickcv[i]
        if self.params.enblrf:
            model[self.respm.rf_idx[0]].frequency = kickrf

    def _process_kicks(self, dkicks):
        chidx = self.respm.ch_idx
        cvidx = self.respm.cv_idx

        par = self.params

        # if kicks are larger the maximum tolerated raise error
        kicks = self.get_kicks()
        nch = len(chidx)
        ncv = len(cvidx)
        kickch, kickcv, kickrf = kicks[:nch], kicks[nch:nch+ncv], kicks[-1]
        cond = _np.any(_np.abs(kickch) >= par.maxkickch)
        cond &= _np.any(_np.abs(kickcv) >= par.maxkickcv)
        if par.enblrf:
            cond &= kickrf >= par.maxkickrf
        if cond:
            raise ValueError('Kicks above maximum allowed value')

        dkickch, dkickcv = dkicks[:nch], dkicks[nch:nch+ncv]
        dkickrf = dkicks[-1]

        # apply factor to dkicks in case they are larger than maximum delta:
        coef_ch = min(
            par.corrgainch, par.maxdeltakickch/_np.abs(dkickch).max())
        coef_cv = min(
            par.corrgaincv, par.maxdeltakickcv/_np.abs(dkickcv).max())
        coef_rf = 1.0
        if par.enblrf and dkickrf != 0:
            coef_rf = min(
                par.corrgainrf, par.maxdeltakickrf/_np.abs(dkickrf))

        # Do not allow kicks to be larger than maximum after application
        # Algorithm:
        # perform the modulus inequality and then
        # since we know that any initial kick is lesser than max_kick
        # from the previous comparison, at this point each column of 'que'
        # has a positive and a negative value. We must consider only
        # the positive one and take the minimum value along the columns
        # to be the multiplicative factor:
        que = [(-par.maxkickch - kickch) / dkickch, ]
        que.append((par.maxkickch - kickch) / dkickch)
        que = _np.max(que, axis=0)
        coef_ch = max(min(_np.min(que), coef_ch), 0)

        que = [(-par.maxkickcv - kickcv) / dkickcv, ]
        que.append((par.maxkickcv - kickcv) / dkickcv)
        que = _np.max(que, axis=0)
        coef_cv = max(min(_np.min(que), coef_cv), 0)

        if self.params.enblrf and dkickrf != 0:
            que = [(-par.maxkickrf - kickrf) / dkickrf, ]
            que.append((par.maxkickrf - kickrf) / dkickrf)
            que = _np.max(que, axis=0)
            coef_rf = max(min(_np.min(que), coef_rf), 0)

        min_coef = min(coef_ch, coef_cv, coef_rf)

        if self.params.useglobalcoef:
            dkickch *= min_coef
            dkickcv *= min_coef
            dkickrf *= min_coef
        else:
            dkickch *= coef_ch
            dkickcv *= coef_cv
            dkickrf *= coef_rf
        saturation_flag = _np.isclose(min_coef, 0)

        kicks[:nch] += dkickch
        kicks[nch:nch+ncv] += dkickcv
        kicks[-1] += dkickrf
        return kicks, saturation_flag
