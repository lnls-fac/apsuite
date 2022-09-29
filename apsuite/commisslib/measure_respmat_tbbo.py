"""."""

import numpy as np

import pyaccel
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import SOFB

from ..optimization import SimulAnneal
from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class Params(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.deltas = {
            'CH': 0.3e-3, 'CV': 0.15e-3, 'InjSept': 0.3e-3, 'InjKckr': 0.3e-3}
        self.wait_time = 2
        self.timeout_orb = 10
        self.num_points = 10


class MeasureRespMatTBBO(_BaseClass):
    """."""

    def __init__(self, all_corrs):
        """."""
        super().__init__(params=Params(), target=self._measure_matrix_thread)
        self.devices = {
            'bo_sofb': SOFB(SOFB.DEVICES.BO),
            'tb_sofb': SOFB(SOFB.DEVICES.TB),
            }
        self._all_corrs = all_corrs
        self._matrix = dict()
        self._corrs_to_measure = []

    @property
    def trajx(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajx, self.devices['bo_sofb'].trajx])

    @property
    def trajy(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajy, self.devices['bo_sofb'].trajy])

    def wait(self, timeout=10):
        """."""
        self.devices['tb_sofb'].wait_buffer(timeout=timeout)
        self.devices['bo_sofb'].wait_buffer(timeout=timeout)

    def reset(self, wait=0):
        """."""
        if self._stopevt.wait(wait):
            return False
        self.devices['tb_sofb'].cmd_reset()
        self.devices['bo_sofb'].cmd_reset()
        if self._stopevt.wait(1):
            return False
        return True

    @property
    def corr_names(self):
        """."""
        corrs = sorted([
            c for c in self._all_corrs if not c.dev.startswith('CV')])
        corrs.extend(sorted([
            c for c in self._all_corrs if c.dev.startswith('CV')]))
        return corrs

    @property
    def corrs_to_measure(self):
        """."""
        if not self._corrs_to_measure:
            return sorted(self._all_corrs.keys() - self._matrix.keys())
        return self._corrs_to_measure

    @corrs_to_measure.setter
    def corrs_to_measure(self, value):
        """."""
        self._corrs_to_measure = sorted([_PVName(n) for n in value])

    @property
    def matrix(self):
        """."""
        mat = np.zeros([len(self._all_corrs), 2*self.trajx.size], dtype=float)
        for i, cor in enumerate(self.corr_names):
            line = self._matrix.get(cor)
            if line is not None:
                mat[i, :] = line
        return mat

    @property
    def nr_points(self):
        """."""
        return min(
            self.devices['tb_sofb'].nr_points,
            self.devices['bo_sofb'].nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.devices['tb_sofb'].nr_points = int(value)
        self.devices['bo_sofb'].nr_points = int(value)

    def _measure_matrix_thread(self):
        self.nr_points = self.params.num_points
        corrs = self.corrs_to_measure
        print('Starting...')
        for i, cor in enumerate(corrs):
            print('{0:2d}|{1:2d}: {2:20s}'.format(i, len(corrs), cor), end='')
            orb = []
            delta = self.params.deltas[cor.dev]
            origkick = self._all_corrs[cor].strength
            print('orig ', end='')
            if not self.reset(self.params.wait_time):
                break
            self.wait(self.params.timeout_orb)
            orb.append(-np.hstack([self.trajx, self.trajy]))

            sig = -2*int(origkick > 0) + 1
            print('pos' if sig > 0 else 'neg')
            self._all_corrs[cor].strength = origkick + sig*delta
            if not self.reset(self.params.wait_time):
                break
            self.wait(self.params.timeout_orb)
            orb.append(np.hstack([self.trajx, self.trajy]))

            self._all_corrs[cor].strength = origkick
            if self._stopevt.is_set():
                print('Stopped!')
                break
            else:
                self._matrix[cor] = np.array(orb).sum(axis=0)/(sig*delta)
        else:
            print('Finished!')


def calc_model_respmatTBBO(
        tb_mod, model, corr_names, elems, meth='middle', ishor=True):
    """."""
    bpms = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'BPM'))[1:]
    _, cumulmat = pyaccel.tracking.find_m44(
        model, indices='open', fixed_point=[0, 0, 0, 0])

    matrix = np.zeros((len(corr_names), 2*bpms.size))
    for idx, corr in enumerate(corr_names):
        elem = elems[corr]
        indcs = np.array(elem.model_indices)
        if corr.sec == 'BO':
            print('Booster ', corr)
            indcs += len(tb_mod)
        cortype = elem.magnet_type
        kxl = kyl = ksxl = ksyl = 0
        if corr.dev == 'InjSept':
            # kxl = tb_mod[indcs[0][1]].KxL
            # kyl = tb_mod[indcs[0][1]].KyL
            # ksxl = tb_mod[indcs[0][1]].KsxL
            # ksyl = tb_mod[indcs[0][1]].KsyL
            midx = pyaccel.lattice.find_indices(
                tb_mod, 'fam_name', 'InjSeptM66')
            for m in midx:
                kxl += tb_mod[m].KxL
                kyl += tb_mod[m].KyL
                ksxl += tb_mod[m].KsxL
                ksyl += tb_mod[m].KsyL
        if not ishor and corr.dev in {'InjSept', 'InjKckr'}:
            cortype = 'vertical'
        matrix[idx, :] = _get_respmat_line(
            cumulmat, indcs, bpms, length=elem.model_length,
            kxl=kxl, kyl=kyl, ksxl=ksxl, ksyl=ksyl,
            cortype=cortype, meth=meth)
    return matrix


def _get_respmat_line(
    cumul_mat, indcs, bpms, length, kxl=0, kyl=0, ksxl=0, ksyl=0,
    cortype='vertical', meth='middle'):

    idx = 3 if cortype.startswith('vertical') else 1
    cor = indcs[0]
    if meth.lower().startswith('end'):
        cor = indcs[-1]+1
    elif meth.lower().startswith('mid'):
        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = np.eye(4, dtype=float)
        drift[0, 1] = length/2 / 2
        drift[2, 3] = length/2 / 2
        quad = np.eye(4, dtype=float)
        quad[1, 0] = -kxl/2
        quad[3, 2] = -kyl/2
        quad[1, 2] = -ksxl/2
        quad[3, 0] = -ksyl/2
        half_cor = np.dot(np.dot(drift, quad), drift)

    m0c = cumul_mat[cor]
    if meth.lower().startswith('mid'):
        m0c = np.linalg.solve(half_cor, m0c)
    mat = np.linalg.solve(m0c.T, cumul_mat[bpms].transpose((0, 2, 1)))
    mat = mat.transpose(0, 2, 1)
    # if meth.lower().startswith('mid'):
    #     mat = np.dot(mat, half_cor)
    respx = mat[:, 0, idx]
    respy = mat[:, 2, idx]
    respx[bpms < indcs[0]] = 0
    respy[bpms < indcs[0]] = 0
    return np.hstack([respx, respy])


class FindSeptQuad(SimulAnneal):
    """."""

    def __init__(self, tb_model, bo_model, corr_names, elems,
                 respmat, nturns=5, save=False, in_sept=True):
        """."""
        super().__init__(save=save)
        self.tb_model = tb_model
        self.bo_model = bo_model
        self.corr_names = corr_names
        self.elems = elems
        self.nturns = nturns
        self.respmat = respmat
        self.in_sept = in_sept

    def initialization(self):
        """."""
        return

    def calc_obj_fun(self):
        """."""
        if self.in_sept:
            sept_idx = pyaccel.lattice.find_indices(
                self.tb_model, 'fam_name', 'InjSept')
        else:
            sept_idx = self.elems['TB-04:MA-CV-2'].model_indices
        k, ks = self._position
        pyaccel.lattice.set_attribute(self.tb_model, 'K', sept_idx, k)
        pyaccel.lattice.set_attribute(self.tb_model, 'Ks', sept_idx, ks)
        respmat = calc_model_respmatTBBO(
            self.tb_model, self.bo_model, self.corr_names, self.elems)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))
