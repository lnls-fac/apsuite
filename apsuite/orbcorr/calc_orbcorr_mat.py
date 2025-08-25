"""."""

import numpy as _np
from pymodels import tb, bo, ts, si
import pyaccel


class OrbRespmat:
    """."""

    _FREQ_DELTA = 10
    _ENERGY_DELTA = 1e-5

    def __init__(self, model, acc='SI', use6dorb=False, corr_system='SOFB'):
        """."""
        self.model = model
        self.use6dorb = use6dorb

        acc = acc.upper()
        if acc not in {'BO', 'SI'}:
            raise ValueError('Variable "acc" must be "BO" or "SI"')

        self.fam_data = (bo if acc == 'BO' else si).get_family_data(self.model)
        cav_famname = 'SRFCav' if acc == 'SI' else 'P5Cav'
        self.rf_idx = self._get_idx(self.fam_data[cav_famname]['index'])

        self.bpm_idx = self._get_idx(self.fam_data['BPM']['index'])

        if corr_system.upper() == 'SOFB':
            self.ch_idx = self._get_idx(self.fam_data['CH']['index'])
            self.cv_idx = self._get_idx(self.fam_data['CV']['index'])
        elif corr_system == 'FOFB':
            self.ch_idx = self._get_idx(self.fam_data['FCH']['index'])
            self.cv_idx = self._get_idx(self.fam_data['FCV']['index'])
        else:
            raise ValueError('Correction system must be "SOFB" or "FOFB"')

    def get_respm(self, add_rfline=True):
        """."""
        cav = self.model.cavity_on
        self.model.cavity_on = self.use6dorb

        find_m = pyaccel.tracking
        find_m = find_m.find_m66 if self.use6dorb else find_m.find_m44
        m_mat, t_mat = find_m(self.model, indices='open')

        nch = len(self.ch_idx)
        respmat = []
        corrs = _np.hstack([self.ch_idx, self.cv_idx])

        for idx, corr in enumerate(corrs):
            rc_mat = t_mat[corr]
            rb_mat = t_mat[self.bpm_idx]
            corr_len = self.model[corr].length
            kl_stren = self.model[corr].KL
            ksl_stren = self.model[corr].KsL
            respx, respy = self._get_respmat_line(
                rc_mat,
                rb_mat,
                m_mat,
                corr,
                corr_len,
                kxl=kl_stren,
                kyl=-kl_stren,
                ksxl=ksl_stren,
                ksyl=ksl_stren,
            )
            respmat.append(respx if idx < nch else respy)

        if add_rfline:
            respmat.append(self._get_rfline())  # [m/Hz]

        respmat = _np.array(respmat).T

        self.model.cavity_on = cav
        return respmat

    def _get_respmat_line(
        self, rc_mat, rb_mat, m_mat, corr, length, kxl=0, kyl=0, ksxl=0, ksyl=0
    ):
        # NOTE: The method for constructing "half_cor" is not the most general
        # possible, as it only considers quadrupolar elements. An attempt was
        # made to generalize the method. Initially, it was tried to calculate
        # "half_cor" by taking the square root of the corrector transfer
        # matrix. However, the overall performance penalty was too high (23%)
        # and it couldn't handle correctors with rotation errors properly.
        # Another possibility is to obtain "half_cor" directly by tracking,
        # defining a component with the same properties as the corrector,
        # but with half the length.

        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = _np.eye(rc_mat.shape[0], dtype=float)
        drift[0, 1] = length / 2 / 2
        drift[2, 3] = length / 2 / 2
        quad = _np.eye(rc_mat.shape[0], dtype=float)
        quad[1, 0] = -kxl / 2
        quad[3, 2] = -kyl / 2
        quad[1, 2] = -ksxl / 2
        quad[3, 0] = -ksyl / 2
        half_cor = drift @ quad @ drift

        # transfer matrix from 0 to the middle of corrector, Rc:
        rc_mat = half_cor @ rc_mat

        # one tune matrix at the middle of corrector, Mc = Rc * M0 * Rc^-1:
        mc_mat = _np.linalg.solve(rc_mat.T, (rc_mat @ m_mat).T).T

        # inverse: (1-Mc)^-1
        mci_mat = _np.eye(mc_mat.shape[0], dtype=float) - mc_mat

        small = self.bpm_idx < corr
        large = _np.logical_not(small)

        # transfer matrix from corrector to BPM, Rc->b:
        # first, calculate Rb * Rc^-1
        rcbl_mat = _np.linalg.solve(rc_mat.T, rb_mat.transpose((0, 2, 1)))
        rcbl_mat = rcbl_mat.transpose((0, 2, 1))

        # if bpm is before corrector, Rc->b = Rb * Rc^-1 * Mc = Rb * M0 * Rc^-1
        rcbs_mat = rcbl_mat[small] @ mc_mat

        # if bpm is after corrector, Rc->b = Rb * Rc^-1
        rcbl_mat = rcbl_mat[large]

        # closed orbit condition from corrector to bpm Rc->b * (1 - Mc)^(-1)
        rcbl_mat = _np.linalg.solve(mci_mat.T, rcbl_mat.transpose((0, 2, 1)))
        rcbl_mat = rcbl_mat.transpose((0, 2, 1))
        rcbs_mat = _np.linalg.solve(mci_mat.T, rcbs_mat.transpose((0, 2, 1)))
        rcbs_mat = rcbs_mat.transpose((0, 2, 1))

        respxx = _np.zeros(len(self.bpm_idx))
        respyx = _np.zeros(len(self.bpm_idx))
        respxy = _np.zeros(len(self.bpm_idx))
        respyy = _np.zeros(len(self.bpm_idx))

        respxx[large] = rcbl_mat[:, 0, 1]
        respyx[large] = rcbl_mat[:, 2, 1]
        respxx[small] = rcbs_mat[:, 0, 1]
        respyx[small] = rcbs_mat[:, 2, 1]
        respx = _np.hstack([respxx, respyx])

        respxy[large] = rcbl_mat[:, 0, 3]
        respyy[large] = rcbl_mat[:, 2, 3]
        respxy[small] = rcbs_mat[:, 0, 3]
        respyy[small] = rcbs_mat[:, 2, 3]
        respy = _np.hstack([respxy, respyy])
        return respx, respy

    def _get_rfline(self):
        idx = self.rf_idx[0]
        rffreq = self.model[idx].frequency
        if self.use6dorb:
            dfreq = OrbRespmat._FREQ_DELTA
            self.model[idx].frequency = rffreq + dfreq
            orbp = pyaccel.tracking.find_orbit6(self.model, indices='open')
            self.model[idx].frequency = rffreq - dfreq
            orbn = pyaccel.tracking.find_orbit6(self.model, indices='open')
            self.model[idx].frequency = rffreq
            rfline = (orbp[[0, 2], :] - orbn[[0, 2], :]) / 2 / dfreq
            rfline = rfline[:, self.bpm_idx].ravel()
        else:
            denergy = OrbRespmat._ENERGY_DELTA
            orbp = pyaccel.tracking.find_orbit4(
                self.model, energy_offset=denergy, indices='open'
            )
            orbn = pyaccel.tracking.find_orbit4(
                self.model, energy_offset=-denergy, indices='open'
            )
            dispbpm = (orbp[[0, 2], :] - orbn[[0, 2], :]) / 2 / denergy
            dispbpm = dispbpm[:, self.bpm_idx].ravel()

            rin = _np.zeros((6, 2))
            rin[4, :] = [denergy, -denergy]
            rout, *_ = pyaccel.tracking.ring_pass(self.model, rin)
            leng = self.model.length
            alpha = -1 * _np.diff(rout[5, :]) / 2 / denergy / leng
            # Convert dispersion to deltax/deltaf:
            rfline = -dispbpm / rffreq / alpha
        return rfline

    @staticmethod
    def _get_idx(indcs):
        return _np.array([idx[0] for idx in indcs])


class TrajRespmat:
    """."""

    def __init__(self, model, acc, nturns=1):
        """."""
        self.model = model
        self.acc = acc
        self.nturns = nturns
        if acc == 'TB':
            self.fam_data = tb.get_family_data(self.model)
        elif acc == 'BO':
            self.fam_data = bo.get_family_data(self.model)
        elif acc == 'TS':
            self.fam_data = ts.get_family_data(self.model)
        elif acc == 'SI':
            self.fam_data = si.get_family_data(self.model)

        self.ch_idx = self.fam_data['CH']['index']
        if acc == 'TS':
            ejesept = pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'EjeSeptG'
            )
            segs = len(ejesept)
            self.ch_idx.append([ejesept[segs // 2]])
            self.ch_idx = sorted(self.ch_idx)

        self.ch_idx = self._get_idx(self.ch_idx)

        if acc == 'TS':
            self.cv_idx = pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'CV'
            )
        else:
            self.cv_idx = self._get_idx(self.fam_data['CV']['index'])

        if self.nturns > 1 and acc in ('BO', 'SI'):
            self.model = self.nturns * model
            bpm_idx = pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'BPM'
            )
        else:
            bpm_idx = self._get_idx(self.fam_data['BPM']['index'])

        # Remove BPM from Linac
        self.bpm_idx = bpm_idx[1:] if acc == 'TB' else bpm_idx

    @staticmethod
    def _get_idx(indcs):
        return _np.array([idx[0] for idx in indcs])

    def get_respm(self):
        """."""
        _, cumulmat = pyaccel.tracking.find_m44(
            self.model, indices='open', fixed_point=[0, 0, 0, 0]
        )

        respmat = []
        corrs = _np.hstack([self.ch_idx, self.cv_idx])
        for idx, corr in enumerate(corrs):
            rc_mat = cumulmat[corr]
            rb_mat = cumulmat[self.bpm_idx]
            corr_len = self.model[corr].length
            kl_stren = self.model[corr].KL
            ksl_stren = self.model[corr].KsL
            respx, respy = self._get_respmat_line(
                rc_mat,
                rb_mat,
                corr,
                length=corr_len,
                kxl=kl_stren,
                kyl=-kl_stren,
                ksxl=ksl_stren,
                ksyl=ksl_stren,
            )
            if idx < len(self.ch_idx):
                respmat.append(respx)
            else:
                respmat.append(respy)

        if self.acc == 'SI':
            # RF column set as zero for trajectory correction in SI
            respmat.append(_np.zeros(2 * len(self.bpm_idx)))
        respmat = _np.array(respmat).T

        if self.nturns > 1:
            respmat = self._calc_nturns_respm(respmat)
        return respmat

    def _get_respmat_line(
        self, rc_mat, rb_mat, corr, length, kxl=0, kyl=0, ksxl=0, ksyl=0
    ):
        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = _np.eye(4, dtype=float)
        drift[0, 1] = length / 2 / 2
        drift[2, 3] = length / 2 / 2
        quad = _np.eye(4, dtype=float)
        quad[1, 0] = -kxl / 2
        quad[3, 2] = -kyl / 2
        quad[1, 2] = -ksxl / 2
        quad[3, 0] = -ksyl / 2
        half_cor = drift @ quad @ drift

        rc_mat = half_cor @ rc_mat

        large = self.bpm_idx > corr

        rb_mat = rb_mat[large, :, :]
        rcb_mat = _np.linalg.solve(rc_mat.T, rb_mat.transpose((0, 2, 1)))
        rcb_mat = rcb_mat.transpose(0, 2, 1)

        respxx = _np.zeros(len(self.bpm_idx), dtype=float)
        respyx = _np.zeros(len(self.bpm_idx), dtype=float)
        respxy = _np.zeros(len(self.bpm_idx), dtype=float)
        respyy = _np.zeros(len(self.bpm_idx), dtype=float)

        respxx[large] = rcb_mat[:, 0, 1]
        respyx[large] = rcb_mat[:, 2, 1]
        respx = _np.hstack([respxx, respyx])

        respxy[large] = rcb_mat[:, 0, 3]
        respyy[large] = rcb_mat[:, 2, 3]
        respy = _np.hstack([respxy, respyy])
        return respx, respy

    def _calc_nturns_respm(self, respmat):
        """."""
        hshape = respmat.shape[0] // 2
        respmx = respmat[:hshape, :]
        respmy = respmat[hshape:, :]
        matx = respmx.copy()
        maty = respmy.copy()
        nturns = self.nturns
        nbpm = len(self.bpm_idx) // nturns
        for iturn in range(1, nturns):
            matx[iturn * nbpm :, :] += respmx[: (nturns - iturn) * nbpm, :]
            maty[iturn * nbpm :, :] += respmy[: (nturns - iturn) * nbpm, :]
        return _np.vstack((matx, maty))
