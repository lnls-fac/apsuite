"""Main module."""

import datetime as _datetime
import time as _time
from copy import deepcopy as _dcopy

import numpy as _np
import pyaccel as _pyacc
from mathphys.functions import get_namedtuple as _get_namedtuple
from pymodels import si as _si
from siriuspy.clientconfigdb import ConfigDBClient
from siriuspy.devices import (
    CurrInfoSI as _CurrInfoSI,
    PowerSupply as _PowerSupply,
    SOFB as _SOFB,
)
from siriuspy.namesys import SiriusPVName as _PVName

from ..orbcorr.orbit_correction import OrbitCorr as _OrbitCorr
from ..utils import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)


class ParallelBBAParams(_ParamsBaseClass):
    """."""

    BPMNAMES = (
        'SI-01M2:DI-BPM',
        'SI-01C1:DI-BPM-1',
        'SI-01C1:DI-BPM-2',
        'SI-01C2:DI-BPM',
        'SI-01C3:DI-BPM-1',
        'SI-01C3:DI-BPM-2',
        'SI-01C4:DI-BPM',
        'SI-02M1:DI-BPM',
        'SI-02M2:DI-BPM',
        'SI-02C1:DI-BPM-1',
        'SI-02C1:DI-BPM-2',
        'SI-02C2:DI-BPM',
        'SI-02C3:DI-BPM-1',
        'SI-02C3:DI-BPM-2',
        'SI-02C4:DI-BPM',
        'SI-03M1:DI-BPM',
        'SI-03M2:DI-BPM',
        'SI-03C1:DI-BPM-1',
        'SI-03C1:DI-BPM-2',
        'SI-03C2:DI-BPM',
        'SI-03C3:DI-BPM-1',
        'SI-03C3:DI-BPM-2',
        'SI-03C4:DI-BPM',
        'SI-04M1:DI-BPM',
        'SI-04M2:DI-BPM',
        'SI-04C1:DI-BPM-1',
        'SI-04C1:DI-BPM-2',
        'SI-04C2:DI-BPM',
        'SI-04C3:DI-BPM-1',
        'SI-04C3:DI-BPM-2',
        'SI-04C4:DI-BPM',
        'SI-05M1:DI-BPM',
        'SI-05M2:DI-BPM',
        'SI-05C1:DI-BPM-1',
        'SI-05C1:DI-BPM-2',
        'SI-05C2:DI-BPM',
        'SI-05C3:DI-BPM-1',
        'SI-05C3:DI-BPM-2',
        'SI-05C4:DI-BPM',
        'SI-06M1:DI-BPM',
        'SI-06M2:DI-BPM',
        'SI-06C1:DI-BPM-1',
        'SI-06C1:DI-BPM-2',
        'SI-06C2:DI-BPM',
        'SI-06C3:DI-BPM-1',
        'SI-06C3:DI-BPM-2',
        'SI-06C4:DI-BPM',
        'SI-07M1:DI-BPM',
        'SI-07M2:DI-BPM',
        'SI-07C1:DI-BPM-1',
        'SI-07C1:DI-BPM-2',
        'SI-07C2:DI-BPM',
        'SI-07C3:DI-BPM-1',
        'SI-07C3:DI-BPM-2',
        'SI-07C4:DI-BPM',
        'SI-08M1:DI-BPM',
        'SI-08M2:DI-BPM',
        'SI-08C1:DI-BPM-1',
        'SI-08C1:DI-BPM-2',
        'SI-08C2:DI-BPM',
        'SI-08C3:DI-BPM-1',
        'SI-08C3:DI-BPM-2',
        'SI-08C4:DI-BPM',
        'SI-09M1:DI-BPM',
        'SI-09M2:DI-BPM',
        'SI-09C1:DI-BPM-1',
        'SI-09C1:DI-BPM-2',
        'SI-09C2:DI-BPM',
        'SI-09C3:DI-BPM-1',
        'SI-09C3:DI-BPM-2',
        'SI-09C4:DI-BPM',
        'SI-10M1:DI-BPM',
        'SI-10M2:DI-BPM',
        'SI-10C1:DI-BPM-1',
        'SI-10C1:DI-BPM-2',
        'SI-10C2:DI-BPM',
        'SI-10C3:DI-BPM-1',
        'SI-10C3:DI-BPM-2',
        'SI-10C4:DI-BPM',
        'SI-11M1:DI-BPM',
        'SI-11M2:DI-BPM',
        'SI-11C1:DI-BPM-1',
        'SI-11C1:DI-BPM-2',
        'SI-11C2:DI-BPM',
        'SI-11C3:DI-BPM-1',
        'SI-11C3:DI-BPM-2',
        'SI-11C4:DI-BPM',
        'SI-12M1:DI-BPM',
        'SI-12M2:DI-BPM',
        'SI-12C1:DI-BPM-1',
        'SI-12C1:DI-BPM-2',
        'SI-12C2:DI-BPM',
        'SI-12C3:DI-BPM-1',
        'SI-12C3:DI-BPM-2',
        'SI-12C4:DI-BPM',
        'SI-13M1:DI-BPM',
        'SI-13M2:DI-BPM',
        'SI-13C1:DI-BPM-1',
        'SI-13C1:DI-BPM-2',
        'SI-13C2:DI-BPM',
        'SI-13C3:DI-BPM-1',
        'SI-13C3:DI-BPM-2',
        'SI-13C4:DI-BPM',
        'SI-14M1:DI-BPM',
        'SI-14M2:DI-BPM',
        'SI-14C1:DI-BPM-1',
        'SI-14C1:DI-BPM-2',
        'SI-14C2:DI-BPM',
        'SI-14C3:DI-BPM-1',
        'SI-14C3:DI-BPM-2',
        'SI-14C4:DI-BPM',
        'SI-15M1:DI-BPM',
        'SI-15M2:DI-BPM',
        'SI-15C1:DI-BPM-1',
        'SI-15C1:DI-BPM-2',
        'SI-15C2:DI-BPM',
        'SI-15C3:DI-BPM-1',
        'SI-15C3:DI-BPM-2',
        'SI-15C4:DI-BPM',
        'SI-16M1:DI-BPM',
        'SI-16M2:DI-BPM',
        'SI-16C1:DI-BPM-1',
        'SI-16C1:DI-BPM-2',
        'SI-16C2:DI-BPM',
        'SI-16C3:DI-BPM-1',
        'SI-16C3:DI-BPM-2',
        'SI-16C4:DI-BPM',
        'SI-17M1:DI-BPM',
        'SI-17M2:DI-BPM',
        'SI-17C1:DI-BPM-1',
        'SI-17C1:DI-BPM-2',
        'SI-17C2:DI-BPM',
        'SI-17C3:DI-BPM-1',
        'SI-17C3:DI-BPM-2',
        'SI-17C4:DI-BPM',
        'SI-18M1:DI-BPM',
        'SI-18M2:DI-BPM',
        'SI-18C1:DI-BPM-1',
        'SI-18C1:DI-BPM-2',
        'SI-18C2:DI-BPM',
        'SI-18C3:DI-BPM-1',
        'SI-18C3:DI-BPM-2',
        'SI-18C4:DI-BPM',
        'SI-19M1:DI-BPM',
        'SI-19M2:DI-BPM',
        'SI-19C1:DI-BPM-1',
        'SI-19C1:DI-BPM-2',
        'SI-19C2:DI-BPM',
        'SI-19C3:DI-BPM-1',
        'SI-19C3:DI-BPM-2',
        'SI-19C4:DI-BPM',
        'SI-20M1:DI-BPM',
        'SI-20M2:DI-BPM',
        'SI-20C1:DI-BPM-1',
        'SI-20C1:DI-BPM-2',
        'SI-20C2:DI-BPM',
        'SI-20C3:DI-BPM-1',
        'SI-20C3:DI-BPM-2',
        'SI-20C4:DI-BPM',
        'SI-01M1:DI-BPM',
    )
    QUADNAMES = (
        'SI-01M2:PS-QS',
        'SI-01C1:PS-Q1',
        'SI-01C1:PS-QS',
        'SI-01C2:PS-QS',
        'SI-01C3:PS-Q4',
        'SI-01C3:PS-QS',
        'SI-01C4:PS-Q1',
        'SI-02M1:PS-QDB2',
        'SI-02M2:PS-QDB2',
        'SI-02C1:PS-Q1',
        'SI-02C1:PS-QS',
        'SI-02C2:PS-QS',
        'SI-02C3:PS-Q4',
        'SI-02C3:PS-QS',
        'SI-02C4:PS-Q1',
        'SI-03M1:PS-QDP2',
        'SI-03M2:PS-QDP2',
        'SI-03C1:PS-Q1',
        'SI-03C1:PS-QS',
        'SI-03C2:PS-QS',
        'SI-03C3:PS-Q4',
        'SI-03C3:PS-QS',
        'SI-03C4:PS-Q1',
        'SI-04M1:PS-QDB2',
        'SI-04M2:PS-QDB2',
        'SI-04C1:PS-Q1',
        'SI-04C1:PS-QS',
        'SI-04C2:PS-QS',
        'SI-04C3:PS-Q4',
        'SI-04C3:PS-QS',
        'SI-04C4:PS-Q1',
        'SI-05M1:PS-QS',
        'SI-05M2:PS-QS',
        'SI-05C1:PS-Q1',
        'SI-05C1:PS-QS',
        'SI-05C2:PS-QS',
        'SI-05C3:PS-Q4',
        'SI-05C3:PS-QS',
        'SI-05C4:PS-Q1',
        'SI-06M1:PS-QDB2',
        'SI-06M2:PS-QDB2',
        'SI-06C1:PS-Q1',
        'SI-06C1:PS-QS',
        'SI-06C2:PS-QS',
        'SI-06C3:PS-Q4',
        'SI-06C3:PS-QS',
        'SI-06C4:PS-Q1',
        'SI-07M1:PS-QDP2',
        'SI-07M2:PS-QDP2',
        'SI-07C1:PS-Q1',
        'SI-07C1:PS-QS',
        'SI-07C2:PS-QS',
        'SI-07C3:PS-Q4',
        'SI-07C3:PS-QS',
        'SI-07C4:PS-Q1',
        'SI-08M1:PS-QDB2',
        'SI-08M2:PS-QDB2',
        'SI-08C1:PS-Q1',
        'SI-08C1:PS-QS',
        'SI-08C2:PS-QS',
        'SI-08C3:PS-Q4',
        'SI-08C3:PS-QS',
        'SI-08C4:PS-Q1',
        'SI-09M1:PS-QS',
        'SI-09M2:PS-QS',
        'SI-09C1:PS-Q1',
        'SI-09C1:PS-QS',
        'SI-09C2:PS-QS',
        'SI-09C3:PS-Q4',
        'SI-09C3:PS-QS',
        'SI-09C4:PS-Q1',
        'SI-10M1:PS-QDB2',
        'SI-10M2:PS-QDB2',
        'SI-10C1:PS-Q1',
        'SI-10C1:PS-QS',
        'SI-10C2:PS-QS',
        'SI-10C3:PS-Q4',
        'SI-10C3:PS-QS',
        'SI-10C4:PS-Q1',
        'SI-11M1:PS-QDP2',
        'SI-11M2:PS-QDP2',
        'SI-11C1:PS-Q1',
        'SI-11C1:PS-QS',
        'SI-11C2:PS-QS',
        'SI-11C3:PS-Q4',
        'SI-11C3:PS-QS',
        'SI-11C4:PS-Q1',
        'SI-12M1:PS-QDB2',
        'SI-12M2:PS-QDB2',
        'SI-12C1:PS-Q1',
        'SI-12C1:PS-QS',
        'SI-12C2:PS-QS',
        'SI-12C3:PS-Q4',
        'SI-12C3:PS-QS',
        'SI-12C4:PS-Q1',
        'SI-13M1:PS-QS',
        'SI-13M2:PS-QS',
        'SI-13C1:PS-Q1',
        'SI-13C1:PS-QS',
        'SI-13C2:PS-QS',
        'SI-13C3:PS-Q4',
        'SI-13C3:PS-QS',
        'SI-13C4:PS-Q1',
        'SI-14M1:PS-QDB2',
        'SI-14M2:PS-QDB2',
        'SI-14C1:PS-Q1',
        'SI-14C1:PS-QS',
        'SI-14C2:PS-QS',
        'SI-14C3:PS-Q4',
        'SI-14C3:PS-QS',
        'SI-14C4:PS-Q1',
        'SI-15M1:PS-QDP2',
        'SI-15M2:PS-QDP2',
        'SI-15C1:PS-Q1',
        'SI-15C1:PS-QS',
        'SI-15C2:PS-QS',
        'SI-15C3:PS-Q4',
        'SI-15C3:PS-QS',
        'SI-15C4:PS-Q1',
        'SI-16M1:PS-QDB2',
        'SI-16M2:PS-QDB2',
        'SI-16C1:PS-Q1',
        'SI-16C1:PS-QS',
        'SI-16C2:PS-QS',
        'SI-16C3:PS-Q4',
        'SI-16C3:PS-QS',
        'SI-16C4:PS-Q1',
        'SI-17M1:PS-QS',
        'SI-17M2:PS-QS',
        'SI-17C1:PS-Q1',
        'SI-17C1:PS-QS',
        'SI-17C2:PS-QS',
        'SI-17C3:PS-Q4',
        'SI-17C3:PS-QS',
        'SI-17C4:PS-Q1',
        'SI-18M1:PS-QDB2',
        'SI-18M2:PS-QDB2',
        'SI-18C1:PS-Q1',
        'SI-18C1:PS-QS',
        'SI-18C2:PS-QS',
        'SI-18C3:PS-Q4',
        'SI-18C3:PS-QS',
        'SI-18C4:PS-Q1',
        'SI-19M1:PS-QDP2',
        'SI-19M2:PS-QDP2',
        'SI-19C1:PS-Q1',
        'SI-19C1:PS-QS',
        'SI-19C2:PS-QS',
        'SI-19C3:PS-Q4',
        'SI-19C3:PS-QS',
        'SI-19C4:PS-Q1',
        'SI-20M1:PS-QDB2',
        'SI-20M2:PS-QDB2',
        'SI-20C1:PS-Q1',
        'SI-20C1:PS-QS',
        'SI-20C2:PS-QS',
        'SI-20C3:PS-Q4',
        'SI-20C3:PS-QS',
        'SI-20C4:PS-Q1',
        'SI-01M1:PS-QS',
    )

    BPMNAMES = tuple([_PVName(bpm) for bpm in BPMNAMES])
    QUADNAMES = tuple([_PVName(quad) for quad in QUADNAMES])

    def __init__(self):
        """."""
        super().__init__()

        self.quad_deltakl = 0.02  # [1/m]

        self.wait_correctors = 0.3  # [s]
        self.wait_quadrupole = 0.3  # [s]
        self.timeout_wait_orbit = 3  # [s]

        self.corr_nr_iters = 6
        self.inv_jac_rcond = 1e-5

        self.sofb_nrpoints = 20
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]

    def __str__(self):
        """."""
        stg = ''
        stg += f'quad_deltakl = {self.quad_deltakl:.3f}'
        stg += f'wait_correctors = {self.wait_correctors:.3f}'
        stg += f'wait_quadrupole = {self.wait_quadrupole:.3f}'
        stg += f'timeout_wait_orbit = {self.timeout_wait_orbit:.3f}'
        stg += f'corr_nr_iters = {self.corr_nr_iters:.3f}'
        stg += f'inv_jac_rcond = {self.inv_jac_rcond:.3f}'
        stg += f'sofb_nrpoints = {self.sofb_nrpoints:.3f}'
        stg += f'sofb_maxcorriter = {self.sofb_maxcorriter:.3f}'
        stg += f'sofb_maxorberr = {self.sofb_maxorberr:.3f}'
        return stg

    @staticmethod
    def get_default_groups():
        """."""
        return [
            sorted(
                [
                    e
                    for e in ParallelBBAParams.BPMNAMES
                    if (e.sub[2:], e.idx) in k
                ],
                key=lambda x: {
                    'Q4': 0,
                    'Q1': 1,
                    'QDB2': 2,
                    'QDP2': 3,
                    'QS': 4,
                }.get(
                    ParallelBBAParams.QUADNAMES[
                        ParallelBBAParams.BPMNAMES.index(x)
                    ].dev,
                    999,
                ),
            )
            for k in [
                [('M2', ''), ('C3', '1')],
                [('C1', '1'), ('C3', '2')],
                [('C1', '2'), ('C4', '')],
                [('C2', ''), ('M1', '')],
            ]
        ]

    def get_default_dkl(self, groups=None):
        """."""
        groups = (
            ParallelBBAParams.get_default_groups()
            if groups is None
            else groups
        )
        dkl = [_np.ones(len(g)) * self.quad_deltakl for g in groups]
        for d in dkl:
            d[::2] *= -1
        return dkl


class BBAPairPVName(_PVName):
    """."""

    def __new__(
        cls,
        pv_name,
        associated_quad_str=None,
        associated_dkl=0.0,
        elements=None,
    ):
        """."""
        if 'BPM' in pv_name:
            obj = super().__new__(cls, pv_name, elements)
            if associated_quad_str is None:
                associated_quad_str = ParallelBBAParams.QUADNAMES[
                    ParallelBBAParams.BPMNAMES.index(pv_name)
                ]
            obj.associated_quad = _PVName(str(associated_quad_str))
            obj.associated_dkl = associated_dkl
            return obj
        else:
            raise ValueError('Creation restricted for BPMs')


class PBBAGroup(list[BBAPairPVName]):
    """Lista de SiriusPVNames (BPMs + metadados)."""

    def __init__(self, iterable=()):
        """."""
        if all(isinstance(_, BBAPairPVName) for _ in iterable):
            super().__init__(iterable)
        elif all(isinstance(_, str) for _ in iterable):
            iterable = [
                BBAPairPVName(
                    bname,
                    ParallelBBAParams.QUADNAMES[
                        ParallelBBAParams.BPMNAMES.index(bname)
                    ],
                )
                for bname in iterable
                if bname in ParallelBBAParams.BPMNAMES
            ]
            super().__init__(iterable)
        else:
            raise ValueError('')

    @property
    def delta_kl(self):
        """."""
        return _np.array([_.associated_dkl for _ in self])

    @delta_kl.setter
    def delta_kl(self, value):
        if len(value) != len(self):
            raise ValueError
        for i, val in enumerate(value):
            self[i].associated_dkl = val


class DoParallelBBA(_BaseClass):
    """."""

    STATUS = _get_namedtuple('Status', ['Fail', 'Success'])

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=ParallelBBAParams(), target=self._do_pbba, isonline=isonline
        )
        self.clt_confdb = ConfigDBClient(config_type='si_bbadata')
        self.clt_confdb._TIMEOUT_DEFAULT = 20
        self.data['bpmnames'] = list(ParallelBBAParams.BPMNAMES)
        self.data['quadnames'] = list(ParallelBBAParams.QUADNAMES)
        self.data['measure'] = list()
        self.data['groups2dopbba'] = ParallelBBAParams.get_default_groups()
        self.data['delta_kl'] = self.params.get_default_dkl(
            self.data['groups2dopbba']
        )
        self.data['jacobians'] = list()
        self._model = None
        self._fam_data = None

        if self.isonline:
            self.devices['sofb'] = _SOFB(_SOFB.DEVICES.SI)
            self.devices['currinfosi'] = _CurrInfoSI()
            self.connect_to_quadrupoles()

    def __str__(self):
        """."""
        return str(self.params)

    @property
    def havebeam(self):
        """."""
        haveb = self.devices['currinfosi']
        return haveb.connected and haveb.storedbeam

    @property
    def groups2dopbba(self):
        """."""
        return _dcopy(self.data['groups2dopbba'])

    @groups2dopbba.setter
    def groups2dopbba(self, groups):
        self.data['groups2dopbba'] = [
            [_PVName(bpm) for bpm in group if isinstance(bpm, str)]
            for group in groups
            if isinstance(group, (list, tuple, _np.ndarray))
        ]

    @property
    def delta_kl(self):
        """."""
        return _dcopy(self.data['delta_kl'])

    @delta_kl.setter
    def delta_kl(self, value):
        _max = self.params.quad_deltakl
        for i, group in enumerate(self.data['groups2dopbba']):
            if len(value[i]) != len(group):
                raise ValueError(
                    f'size mismatch between group {i} and given delta_kl'
                )
            if any([abs(v) > _max for v in value[i]]):
                raise ValueError(f"values for delta kl can't exceed {_max}")
        self.data['delta_kl'] = _dcopy(value)

    @property
    def jacobians(self):
        """."""
        return _dcopy(self.data['jacobians'])

    @jacobians.setter
    def jacobians(self, jacs):
        """."""
        if len(jacs) != len(self.data['groups2dopbba']):
            raise ValueError('Size not compatible.')
        self.data['jacobians'] = _dcopy(jacs)

    @property
    def model(self):
        """."""
        if self._model is None:
            print('\n     Undefined model... setting a default one')
            self._model = _si.create_accelerator()
            self._model.cavity_on = True
            self._model.radiation_on = 1
            self._model = _si.fitted_models.vertical_dispersion_and_coupling(
                self._model
            )
            self._fam_data = _si.families.get_family_data(self._model)
        return self._model

    @model.setter
    def model(self, value):
        if not value.cavity_on and value.radiation_on != 1:
            raise ValueError(
                'cavity_on must be True and radiation_on must be 1'
            )
        self._model = value
        self._fam_data = _si.families.get_family_data(self._model)

    @property
    def fam_data(self):
        """."""
        return self._fam_data

    def connect_to_quadrupoles(self):
        """."""
        for qname in self.data['quadnames']:
            if qname in self.devices:
                continue
            self.devices[qname] = _PowerSupply(
                qname, props2init=('PwrState-Sts', 'KL-SP', 'KL-RB')
            )

    def get_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer(self.params.timeout_wait_orbit)
        return _np.hstack([sofb.orbx, sofb.orby])

    def correct_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices['sofb']
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

    def get_kicks(self):
        """."""
        sofb = self.devices['sofb']
        return _np._r[sofb.kickch, sofb.kickcv, sofb.kickrf]

    def set_delta_kicks(self, dkicks):
        """."""
        sofb = self.devices['sofb']
        nch, ncv, nrf = sofb._data.nr_ch, sofb._data.nr_cv, 1
        if len(dkicks) != nch + ncv + nrf:
            raise ValueError(
                f'invalid dim for dkicks, must have shape=({nch + ncv + nrf},)'
            )
        dch, dcv, drf = dkicks[:nch], dkicks[nch : nch + ncv], dkicks[-1]

        factch, factcv, factrf = (
            sofb.mancorrgainch,
            sofb.mancorrgaincv,
            sofb.mancorrgainrf,
        )
        sofb.deltakickch, sofb.deltakickcv, sofb.deltakickrf = dch, dcv, drf
        nrsteps = _np.ceil(max(_np.abs(dch).max(), _np.abs(dcv).max()) / 1.0)
        for i in range(int(nrsteps)):
            sofb.mancorrgainch = (i + 1) / nrsteps * 100
            sofb.mancorrgaincv = (i + 1) / nrsteps * 100
            sofb.mancorrgainrf = (i + 1) / nrsteps * 100
            sofb.cmd_applycorr_all()
            _time.sleep(self.params.wait_correctors)
        sofb.deltakickch, sofb.deltakickcv, sofb.deltakickrf = (
            dch * 0,
            dcv * 0,
            drf * 0,
        )
        sofb.mancorrgainch, sofb.mancorrgaincv = factch, factcv, factrf

    # #### pbba utils #####

    def set_quad_strengths(self, group_id, strengths, ignore_timeout=False):
        """."""
        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']
        for strength, bpmname in zip(strengths, bpms):
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            quad.strength = strength

        if ignore_timeout:
            return DoParallelBBA.STATUS.Success

        for strength, bpmname in zip(strengths, bpms):
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            if not quad._wait_float(
                'KLRef-Mon', strength, timeout=self.params.wait_quadrupole
            ):
                return DoParallelBBA.STATUS.Fail
        return DoParallelBBA.STATUS.Success

    def get_quad_strengths(self, group_id):
        """."""
        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        strengths = []
        for bpmname in bpms:
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            strengths.append(quad.strength)
        return _np.array(strengths)

    def meas_ios(self, group_id):
        """."""
        delta_strens = self.data['delta_kl'][group_id]
        strens_orig = self.get_quad_strengths(group_id)
        if not self.set_quad_strengths(
            group_id, strens_orig + delta_strens / 2
        ):
            return None, DoParallelBBA.STATUS.Fail

        orb_pos = self.get_orbit()

        if not self.set_quad_strengths(
            group_id, strens_orig - delta_strens / 2
        ):
            return None, DoParallelBBA.STATUS.Fail

        orb_neg = self.get_orbit()

        if not self.set_quad_strengths(group_id, strens_orig):
            return None, DoParallelBBA.STATUS.Fail

        return orb_pos - orb_neg, DoParallelBBA.STATUS.Success

    def calc_ios_jacobians(self, groups_to_calc=None):
        """Calculate the IOS Response Matrices for all groups."""
        model = self.model
        _orbcorr = _OrbitCorr(
            model=model, acc='SI', corr_system='SOFB', use6dtrack=True
        )
        quadindices = self._get_quads_indices_in_model(self.data['quadnames'])

        def _get_or_set_kl(bname, value=None):
            _do = getattr if value is None else setattr
            bidx = self.data['bpmnames'].index(bname)
            qidx = quadindices[bidx]
            if 'QS' == self.data['quadnames'][bidx].dev:
                return _do(model[qidx], 'KsL', value)
            else:
                return _do(model[qidx], 'KL', value)

        def _get_quad_strengths(group):
            strens = []
            for bname in group:
                strens.append(_get_or_set_kl(bname))
            return _np.array(strens)

        def _set_quad_strengths(group, strengths):
            for strength, bname in zip(strengths, group):
                _get_or_set_kl(bname, strength)

        jacobians = []
        groups_to_calc = (
            _np.arange(len(self.data['groups2dopbba']))
            if groups_to_calc is None
            else groups_to_calc
        )
        for group_id in groups_to_calc:
            group = self.data['groups2dopbba'][group_id]
            try:
                delta_strens = self.data['delta_kl'][group_id]
            except Exception as e:
                str_msg = 'undefined or empty "delta_kl"'
                str_msg += f' of group {group_id}'
                raise IndexError(str_msg) from e
            strens_orig = _get_quad_strengths(group)

            _set_quad_strengths(group, strens_orig + delta_strens / 2)
            try:
                jac_pos = _orbcorr.get_jacobian_matrix()
            except Exception as err:
                _set_quad_strengths(group, strens_orig)
                raise err

            _set_quad_strengths(group, strens_orig - delta_strens / 2)
            try:
                jac_neg = _orbcorr.get_jacobian_matrix()
            except Exception as err:
                _set_quad_strengths(group, strens_orig)
                raise err

            _set_quad_strengths(group, strens_orig)
            jacobians.append(jac_pos - jac_neg)
        return jacobians

    def analyze_groups(self):
        """Helper function to analyze the groups' properties."""
        if not self.data['jacobians']:
            raise ValueError('Please calculate and set jacobians first.')

        anl = []
        for group_id in range(len(self.data['groups2dopbba'])):
            print(f'Analyzing group: {group_id:d}')
            anl.append(self.analyze_group(group_id))
        return anl

    def analyze_group(self, group_id):
        """Helper function to analyze group's properties."""
        jacobian = self.data['jacobians'][group_id]
        u_mat, svals, vt_mat = _np.linalg.svd(jacobian)

        model = self.model
        quadindices = self._get_quads_indices_in_model(self.data['quadnames'])
        delta_strens = self.data['delta_kl'][group_id]
        group = self.data['groups2dopbba'][group_id]

        tune_variation = [_pyacc.optics.get_frac_tunes(model)[:2]]

        for fac in [1, -2, 1]:
            for dkl, bpm in zip(delta_strens, group):
                _id = self.data['bpmnames'].index(bpm)
                qname = self.data['quadnames'][_id]
                qidx = quadindices[_id]
                if 'QS' in qname:
                    model[qidx].KsL += fac * dkl / 2
                else:
                    model[qidx].KL += fac * dkl / 2
                tune_variation.append(_pyacc.optics.get_frac_tunes(model)[:2])

        return {
            'u_matrix': u_mat,
            'vt_matrix': vt_mat,
            'svals': svals,
            'tune_variation': _np.array(tune_variation),
        }

    # #### private methods ####
    def _get_quads_indices_in_model(self, quadnames):
        """."""
        fam_data = self.fam_data
        quadindices = []
        for qname in quadnames:
            key = qname.dev
            idx = fam_data[key]['devnames'].index(qname)
            qindex = fam_data[key]['index'][idx]
            qindex = qindex[0] if len(qindex) == 1 else qindex
            quadindices.append(qindex)
        return quadindices

    def _do_pbba(self):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        print(
            'Starting measurement at {:s}'.format(
                tini.strftime('%Y-%m-%d %Hh%Mm%Ss')
            )
        )

        self.data['jacobians'] = self.calc_ios_jacobians()

        sofb = self.devices['sofb']
        if sofb.autocorrsts:
            print('SOFB feedback is enabled. Please desable it first.')
            return

        for group_id in range(len(self.data['groups2dopbba'])):
            if self._stopevt.is_set():
                print('Stopped!')
                break
            if not self.havebeam:
                print('Beam was Lost')
                break
            print('\nCorrecting Orbit... ', end='')
            self.correct_orbit()
            print('Ok!')
            if self._dopbba_single_group(group_id):
                break

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        print('Finished! Elapsed time {:s}'.format(dtime))

    def _dopbba_single_group(self, group_id):
        """."""
        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime('%Hh%Mm%Ss')
        print(f'{strtini:s} --> Doing PBBA for Group {group_id:d}')

        jac = self.data['jacobians'][group_id]
        inv_jac = _np.linalg.pinv(jac, self.params.inv_jac_rcond)

        group_data = {
            'bpms': self.data['groups2dopbba'][group_id],
            'orbit_init': self.get_orbit(),
            'kicks_init': self.get_kicks(),
            'strengths_init': self.get_quad_strengths(group_id),
        }
        ios_iter, dkicks_iter = [], []
        nr_iters = self.params.corr_nr_iters
        print('    correcting IOS:', end='')
        sts = self.STATUS.Fail
        for i in range(nr_iters):
            print('    {:02d}/{:02d} --> '.format(i + 1, nr_iters), end='')
            if self._stopevt.is_set() or not self.havebeam:
                self._restore_init_conditions(
                    group_id, init_strengths=group_data['strengths_init']
                )
                print('   exiting...')
                break
            ios, sts = self.meas_ios(group_id)
            if not sts:
                self._restore_init_conditions(
                    group_id, init_strengths=group_data['strengths_init']
                )
                break
            ios_iter.append(ios)
            dkicks = list(-1 * _np.dot(inv_jac, ios))
            dkicks_iter.append(dkicks)
            self.set_delta_kicks(dkicks)
            print('Done.')
        else:
            sts = self.STATUS.Success

        # final ios
        ios, sts = self.meas_ios(group_id)
        if not sts:
            self._restore_init_conditions(
                group_id, init_strengths=group_data['strengths_init']
            )
        ios_iter.append(ios)

        group_data['kicks_end'] = self.get_kicks()
        group_data['ios_iter'] = ios_iter
        group_data['dkicks_iter'] = dkicks_iter
        group_data['orbit_end'] = self.get_orbit()
        group_data['delta_kl'] = self.data['delta_kl'][group_id]
        self.data['measure'].append(group_data)

        self.correct_orbit()

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        if sts:
            print('Done! Elapsed time: {:s}\n'.format(dtime))
        print('Fail! Elapsed time: {:s}\n'.format(dtime))
        return sts

    def _restore_init_conditions(self, group_id, init_strengths):
        """."""
        self.set_quad_strengths(group_id, init_strengths, ignore_timeout=True)
        _time.sleep(self.params.wait_quadrupole)
        self.correct_orbit()
