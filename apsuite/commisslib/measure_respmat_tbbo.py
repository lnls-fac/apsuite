"""."""

import time as _time
import numpy as np

import pyaccel
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply
from siriuspy.devices import EVG as _EVG
from siriuspy.devices import SOFB as _SOFB

from ..optimization import SimulAnneal
from ..utils import (
    ThreadedMeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)


class Params(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.corr_nrpts = 5
        self.corr_range = {
            "CH": [-4, 4],  # [urad]
            "CV": [-4, 4],  # [urad]
            "InjSept": [-1, 1],  # [mrad]
            "InjKckr": [-0.3, 0.3],  # [mrad]
        }
        self.corr_wait = 0.5  # [s]
        self.injection_interval = 3  # [s]
        # self.wait_time = 2  # [s]

        self.timeout_orb = 10  # [s]
        self.nr_points = 10

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        ttmp = '{0:24s} = {1}\n'.format
        stg = dtmp('corr_nrpts', self.corr_nrpts, '')
        stg += ttmp('corr_range', self.corr_range, '')
        stg += ftmp('corr_wait', self.corr_wait, '[s]')
        stg += ftmp('injection_interval', self.injection_interval, '[s]')
        stg += ftmp('timeout_orb', self.timeout_orb, '[s]')
        stg += dtmp('nr_points', self.nr_points, '')
        return stg


class MeasureRespMatTBBO(_BaseClass):
    """."""

    def __init__(self, corr_names, isonline=True):
        """."""
        super().__init__(params=Params(), target=self.measure_respmat)
        self.isonline = isonline
        self.devices = dict()
        self._corr_names = [_PVName(corr) for corr in corr_names]
        self._matrix = dict()
        self._corrs_to_measure = []
        if isonline:
            self._create_devices()

    @property
    def trajx(self):
        """."""
        return np.hstack(
            [self.devices["tb_sofb"].trajx, self.devices["bo_sofb"].trajx]
        )

    @property
    def trajy(self):
        """."""
        return np.hstack(
            [self.devices["tb_sofb"].trajy, self.devices["bo_sofb"].trajy]
        )

    @property
    def trajs(self):
        """."""
        return np.hstack(
            [self.devices["tb_sofb"].sum, self.devices["bo_sofb"].sum]
        )

    @property
    def traj(self):
        """."""
        traj_xy = np.hstack([self.trajx, self.trajy])
        return traj_xy

    def wait_new_traj(self, traj_xy_0=None, timeout_orb=None):
        """."""
        timeout_orb = timeout_orb or self.params.timeout_orb
        if traj_xy_0 is None:
            traj_xy_0 = self.traj
        for _ in range(50):
            traj_xy = self.traj
            if not np.any(np.isclose(traj_xy_0, traj_xy)):
                return True
            _time.sleep(timeout_orb / 50)
        return False

    def inject_and_get_data(self, corr_name):
        """."""
        evg = self.devices['evg']
        corr_strn0 = self.devices[corr_name].strength
        traj_xy = list()
        traj_sum = list()
        timestamp = list()
        corr_strn = list()
        for i in range(self.params.nr_points):
            traj_xy_0 = self.traj
            evg.cmd_turn_on_injection()
            # traj_xy_new, traj_sum_new = self.get_new_traj(traj_xy_0)

            t0_ = _time.time()
            stg = f'    {i + 1:02d}/{self.params.nr_points:02d} -> '
            stg += 'Getting trajectory...'
            print(stg, end='\r', flush=True)
            if not self.wait_new_traj(traj_xy_0):
                stg += ' timed out waiting traj to update.'
            print(stg + '  done!')

            traj_xy_new = self.traj
            traj_sum_new = self.trajs
            timestamp.append(_time.time())
            traj_xy.append(traj_xy_new)
            traj_sum.append(traj_sum_new)
            corr_strn.append(corr_strn0)
            dtim = max(
                0, self.params.injection_interval - (_time.time() - t0_)
            )
            if i < self.params.nr_points - 1:
                _time.sleep(dtim)
        return dict(
            traj_xy=traj_xy,
            traj_sum=traj_sum,
            timestamp=timestamp,
            corr_strn=corr_strn,
        )

    @property
    def corr_names(self):
        """."""
        return self._corr_names[:]
        # corrs = sorted(
        #     [c for c in self._corr_names if not c.dev.startswith("CV")]
        # )
        # corrs.extend(
        #     sorted([c for c in self._corr_names if c.dev.startswith("CV")])
        # )
        # return corrs

    @property
    def corrs_to_measure(self):
        """."""
        if not self._corrs_to_measure:
            return self._corr_names[:]
        else:
            return self._corrs_to_measure
        # if not self._corrs_to_measure:
        #     return sorted(self._corr_names.keys() - self._matrix.keys())
        # return self._corrs_to_measure

    @corrs_to_measure.setter
    def corrs_to_measure(self, value):
        """."""
        for corr_name in self._corr_names:
            if corr_name in value:
                self._corrs_to_measure.append(corr_name)

    def measure_respmat_corr(self, corr_name):
        """."""
        nrpts = self.params.corr_nrpts
        kick_min, kick_max = self.params.corr_range[corr_name.dev]
        delta_strength = np.linspace(kick_min, kick_max, nrpts)

        corr_dev = self.devices[corr_name]
        orig_strn = corr_dev.strength

        stopped = False
        data = []
        try:
            for i, delta_strn in enumerate(delta_strength):
                print(
                    f'  {corr_name} {i + 1:02d}/{nrpts:02d} --> '
                    f'delta_strength: {delta_strn:.3f}'
                )
                new_strn = orig_strn + delta_strn
                self._set_device_corrector(corr_name, new_strn)
                _time.sleep(self.params.corr_wait)

                orb_data = self.inject_and_get_data(corr_name)
                data.append(orb_data)

                if self._stopevt.is_set():
                    stopped = True
                    break
        finally:
            print(f'  restoring {corr_name} strength...')
            self._set_device_corrector(corr_name, orig_strn)
            # _time.sleep(1)
            print(f'  {corr_name} strength: {corr_dev.strength:.3f}')
            if stopped:
                print(f'  {corr_name} interrupted!')
            else:
                print(f'  {corr_name} finished!')
        return stopped, data

    def measure_respmat(self):
        """."""
        self.nr_points = self.params.nr_points
        corrs = self.corrs_to_measure

        self.data = dict()
        print("Starting...")

        for idx, corr_name in enumerate(corrs):
            print(f'Varrying {corr_name:<20s} ({idx+1:02d}/{len(corrs):02d})')
            stopped, data = self.measure_respmat_corr(corr_name)
            if not stopped:
                self.data[corr_name] = data
            else:
                break

        print("Finished.")

    def process_data_corr(self, corr_name, fit_order=1):
        """."""
        data = self.data[corr_name]
        if not data:
            raise ValueError("No data to process. Run measure first.")

        fit_results = []
        nr_bpms = len(data[0]["traj_xy"][0])

        respmat_meas = np.zeros((len(self.data), nr_bpms), dtype=float)

        for i, datum in enumerate(self.data):

            xfit = np.array(datum["delta_strength"])
            traj_xy = np.array(datum["traj_xy"])

            coefs, _ = np.polynomial.polynomial.polyfit(
                xfit, traj_xy, deg=fit_order, full=True
            )

            ress = [(traj_xy**2).sum(axis=0)]

            for order in range(1, fit_order + 2):
                fit = np.polynomial.polynomial.polyval(xfit, coefs[:order])
                ress.append(((traj_xy - fit.T) ** 2).sum(axis=0))

            ress = np.array(ress)
            ratio = ress / ress[1][None, :]

            fit_results.append(
                {
                    "corr": datum["corr_name"],
                    "orig_strength": datum["orig_strength"],
                    "fit_x": xfit,
                    "fit_coefs": coefs,
                    "fit_residue_order": ress,
                    "fit_rel_residue": ratio,
                    "traj_xy": traj_xy,
                }
            )

            respmat_meas[i] = coefs[1]

        self.analysis = {
            "fit_order": fit_order,
            "fit_results": fit_results,
            "respmat_meas": respmat_meas,
        }

    def _create_devices(self):
        """."""
        self.devices = dict(
            evg=_EVG(),
            tb_sofb=_SOFB(_SOFB.DEVICES.TB),
            bo_sofb=_SOFB(_SOFB.DEVICES.BO),
        )
        for corr_name in self._corr_names:
            self.devices[corr_name] = _PowerSupply(corr_name)

    def _set_device_corrector(self, devname, value):
        # self.devices[devname].strength = value
        pass


def calc_model_respmatTBBO(
    tb_mod, model, corr_names, elems, meth="middle", ishor=True
):
    """."""
    bpms = np.array(pyaccel.lattice.find_indices(model, "fam_name", "BPM"))[1:]
    _, cumulmat = pyaccel.tracking.find_m44(
        model, indices="open", fixed_point=[0, 0, 0, 0]
    )

    matrix = np.zeros((len(corr_names), 2 * bpms.size))
    for idx, corr in enumerate(corr_names):
        elem = elems[corr]
        indcs = np.array(elem.model_indices)
        if corr.sec == "BO":
            print("Booster ", corr)
            indcs += len(tb_mod)
        cortype = elem.magnet_type
        kxl = kyl = ksxl = ksyl = 0
        if corr.dev == "InjSept":
            # kxl = tb_mod[indcs[0][1]].KxL
            # kyl = tb_mod[indcs[0][1]].KyL
            # ksxl = tb_mod[indcs[0][1]].KsxL
            # ksyl = tb_mod[indcs[0][1]].KsyL
            midx = pyaccel.lattice.find_indices(
                tb_mod, "fam_name", "InjSeptM66"
            )
            for m in midx:
                kxl += tb_mod[m].KxL
                kyl += tb_mod[m].KyL
                ksxl += tb_mod[m].KsxL
                ksyl += tb_mod[m].KsyL
        if not ishor and corr.dev in {"InjSept", "InjKckr"}:
            cortype = "vertical"
        matrix[idx, :] = _get_respmat_line(
            cumulmat,
            indcs,
            bpms,
            length=elem.model_length,
            kxl=kxl,
            kyl=kyl,
            ksxl=ksxl,
            ksyl=ksyl,
            cortype=cortype,
            meth=meth,
        )
    return matrix


def _get_respmat_line(
    cumul_mat,
    indcs,
    bpms,
    length,
    kxl=0,
    kyl=0,
    ksxl=0,
    ksyl=0,
    cortype="vertical",
    meth="middle",
):

    idx = 3 if cortype.startswith("vertical") else 1
    cor = indcs[0]
    if meth.lower().startswith("end"):
        cor = indcs[-1] + 1
    elif meth.lower().startswith("mid"):
        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = np.eye(4, dtype=float)
        drift[0, 1] = length / 2 / 2
        drift[2, 3] = length / 2 / 2
        quad = np.eye(4, dtype=float)
        quad[1, 0] = -kxl / 2
        quad[3, 2] = -kyl / 2
        quad[1, 2] = -ksxl / 2
        quad[3, 0] = -ksyl / 2
        half_cor = np.dot(np.dot(drift, quad), drift)

    m0c = cumul_mat[cor]
    if meth.lower().startswith("mid"):
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

    def __init__(
        self,
        tb_model,
        bo_model,
        corr_names,
        elems,
        respmat,
        nturns=5,
        save=False,
        in_sept=True,
    ):
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
                self.tb_model, "fam_name", "InjSept"
            )
        else:
            sept_idx = self.elems["TB-04:MA-CV-2"].model_indices
        k, ks = self._position
        pyaccel.lattice.set_attribute(self.tb_model, "K", sept_idx, k)
        pyaccel.lattice.set_attribute(self.tb_model, "Ks", sept_idx, ks)
        respmat = calc_model_respmatTBBO(
            self.tb_model, self.bo_model, self.corr_names, self.elems
        )
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat * respmat))
