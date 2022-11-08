"""."""

from copy import deepcopy as _dcopy
from mathphys.functions import get_namedtuple as _get_namedtuple
import numpy as _np

import pyaccel

from .calc_orbcorr_mat import OrbRespmat


class CorrParams:
    """."""

    def __init__(self):
        """."""
        self.minsingval = 0.2
        self.maxkickch = 300e-6  # rad
        self.maxkickcv = 300e-6  # rad
        self.maxkickrf = 500e6  # Hz
        self.maxdeltakickch = 300e-6  # rad
        self.maxdeltakickcv = 300e-6  # rad
        self.maxdeltakickrf = 1000  # Hz
        self.maxnriters = 10
        self.tolerance = 0.5e-6  # m
        self.enblrf = True
        self.enbllistbpm = None
        self.enbllistch = None
        self.enbllistcv = None


class OrbitCorr:
    """."""

    CORR_STATUS = _get_namedtuple('CorrStatus', ['Fail', 'Sucess'])

    def __init__(self, model, acc, params=None, corr_system='SOFB'):
        """."""
        self.acc = acc
        self.params = params or CorrParams()
        self.respm = OrbRespmat(
            model=model, acc=self.acc, dim='6d', corr_system=corr_system)
        self.respm.model.cavity_on = True
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
        """."""
        jmat = _dcopy(jacobian_matrix)
        jmat[_np.logical_not(self.params.enbllistbpm), :] = 0
        enblcorrs = _np.r_[
            self.params.enbllistch, self.params.enbllistcv, self.params.enblrf]
        jmat[:, ~enblcorrs] = 0

        umat, smat, vmat = _np.linalg.svd(jmat, full_matrices=False)
        idx = smat > self.params.minsingval
        ismat = _np.zeros(smat.shape, dtype=float)
        ismat[idx] = 1/smat[idx]

        ismat = _np.dot(vmat.T*ismat, umat.T)
        if full:
            return ismat, umat, smat, vmat
        else:
            return ismat

    def correct_orbit(self, jacobian_matrix=None, goal_orbit=None):
        """Orbit correction.

        Calculates the pseudo-inverse of orbit correction matrix via SVD
        and minimizes the residue vector [CODx@BPM, CODy@BPM].
        """
        if goal_orbit is None:
            nbpm = len(self.respm.bpm_idx)
            goal_orbit = _np.zeros(2 * nbpm)

        if jacobian_matrix is None:
            jmat = self.get_jacobian_matrix()
        else:
            jmat = jacobian_matrix

        ismat = self.get_inverse_matrix(jmat)

        orb = self.get_orbit()
        dorb = orb - goal_orbit
        bestfigm = OrbitCorr.get_figm(dorb)
        if bestfigm < self.params.tolerance:
            return OrbitCorr.CORR_STATUS.Sucess

        for _ in range(self.params.maxnriters):
            dkicks = -1*_np.dot(ismat, dorb)
            kicks = self._process_kicks(dkicks)
            self.set_kicks(kicks)
            orb = self.get_orbit()
            dorb = orb - goal_orbit
            figm = OrbitCorr.get_figm(dorb)
            diff_figm = _np.abs(bestfigm - figm)
            if figm < bestfigm:
                bestfigm = figm
            if diff_figm < self.params.tolerance:
                break
        else:
            return OrbitCorr.CORR_STATUS.Fail
        return OrbitCorr.CORR_STATUS.Sucess

    def get_orbit(self):
        """."""
        cod = pyaccel.tracking.find_orbit6(self.respm.model, indices='open')
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
        dkickch *= min(1, par.maxdeltakickch / _np.abs(dkickch).max())
        dkickcv *= min(1, par.maxdeltakickcv / _np.abs(dkickcv).max())
        if par.enblrf and dkickrf != 0:
            dkickrf *= min(1, par.maxdeltakickrf / _np.abs(dkickrf))

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
        dkickch *= min(_np.min(que), 1.0)

        que = [(-par.maxkickcv - kickcv) / dkickcv, ]
        que.append((par.maxkickcv - kickcv) / dkickcv)
        que = _np.max(que, axis=0)
        dkickcv *= min(_np.min(que), 1.0)

        if self.params.enblrf and dkickrf != 0:
            que = [(-par.maxkickrf - kickrf) / dkickrf, ]
            que.append((par.maxkickrf - kickrf) / dkickrf)
            que = _np.max(que, axis=0)
            dkickrf *= min(_np.min(que), 1.0)

        kicks[:nch] += dkickch
        kicks[nch:nch+ncv] += dkickcv
        kicks[-1] += dkickrf
        return kicks
