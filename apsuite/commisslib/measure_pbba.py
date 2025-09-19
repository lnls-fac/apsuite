"""Main module."""

import datetime as _datetime
import time as _time
from copy import deepcopy as _dcopy

import matplotlib.cm as _cmap
import matplotlib.gridspec as _mpl_gs
import matplotlib.pyplot as _plt
import numpy as _np
import pyaccel as _pyacc
from scipy.optimize import least_squares as _least_squares
from siriuspy.clientconfigdb import ConfigDBClient
from siriuspy.devices import CurrInfoSI as _CurrInfoSI, \
    PowerSupply as _PowerSupply, SOFB as _SOFB
from siriuspy.namesys import SiriusPVName as _PVName

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _BaseClass
from .measure_bba import DoBBA as _DoBBA
from ..orbcorr.orbit_correction import OrbitCorr as _OrbitCorr

class ParallelBBAParams(_ParamsBaseClass):
    """."""

    BPMNAMES = (
        "SI-01M2:DI-BPM",
        "SI-01C1:DI-BPM-1",
        "SI-01C1:DI-BPM-2",
        "SI-01C2:DI-BPM",
        "SI-01C3:DI-BPM-1",
        "SI-01C3:DI-BPM-2",
        "SI-01C4:DI-BPM",
        "SI-02M1:DI-BPM",
        "SI-02M2:DI-BPM",
        "SI-02C1:DI-BPM-1",
        "SI-02C1:DI-BPM-2",
        "SI-02C2:DI-BPM",
        "SI-02C3:DI-BPM-1",
        "SI-02C3:DI-BPM-2",
        "SI-02C4:DI-BPM",
        "SI-03M1:DI-BPM",
        "SI-03M2:DI-BPM",
        "SI-03C1:DI-BPM-1",
        "SI-03C1:DI-BPM-2",
        "SI-03C2:DI-BPM",
        "SI-03C3:DI-BPM-1",
        "SI-03C3:DI-BPM-2",
        "SI-03C4:DI-BPM",
        "SI-04M1:DI-BPM",
        "SI-04M2:DI-BPM",
        "SI-04C1:DI-BPM-1",
        "SI-04C1:DI-BPM-2",
        "SI-04C2:DI-BPM",
        "SI-04C3:DI-BPM-1",
        "SI-04C3:DI-BPM-2",
        "SI-04C4:DI-BPM",
        "SI-05M1:DI-BPM",
        "SI-05M2:DI-BPM",
        "SI-05C1:DI-BPM-1",
        "SI-05C1:DI-BPM-2",
        "SI-05C2:DI-BPM",
        "SI-05C3:DI-BPM-1",
        "SI-05C3:DI-BPM-2",
        "SI-05C4:DI-BPM",
        "SI-06M1:DI-BPM",
        "SI-06M2:DI-BPM",
        "SI-06C1:DI-BPM-1",
        "SI-06C1:DI-BPM-2",
        "SI-06C2:DI-BPM",
        "SI-06C3:DI-BPM-1",
        "SI-06C3:DI-BPM-2",
        "SI-06C4:DI-BPM",
        "SI-07M1:DI-BPM",
        "SI-07M2:DI-BPM",
        "SI-07C1:DI-BPM-1",
        "SI-07C1:DI-BPM-2",
        "SI-07C2:DI-BPM",
        "SI-07C3:DI-BPM-1",
        "SI-07C3:DI-BPM-2",
        "SI-07C4:DI-BPM",
        "SI-08M1:DI-BPM",
        "SI-08M2:DI-BPM",
        "SI-08C1:DI-BPM-1",
        "SI-08C1:DI-BPM-2",
        "SI-08C2:DI-BPM",
        "SI-08C3:DI-BPM-1",
        "SI-08C3:DI-BPM-2",
        "SI-08C4:DI-BPM",
        "SI-09M1:DI-BPM",
        "SI-09M2:DI-BPM",
        "SI-09C1:DI-BPM-1",
        "SI-09C1:DI-BPM-2",
        "SI-09C2:DI-BPM",
        "SI-09C3:DI-BPM-1",
        "SI-09C3:DI-BPM-2",
        "SI-09C4:DI-BPM",
        "SI-10M1:DI-BPM",
        "SI-10M2:DI-BPM",
        "SI-10C1:DI-BPM-1",
        "SI-10C1:DI-BPM-2",
        "SI-10C2:DI-BPM",
        "SI-10C3:DI-BPM-1",
        "SI-10C3:DI-BPM-2",
        "SI-10C4:DI-BPM",
        "SI-11M1:DI-BPM",
        "SI-11M2:DI-BPM",
        "SI-11C1:DI-BPM-1",
        "SI-11C1:DI-BPM-2",
        "SI-11C2:DI-BPM",
        "SI-11C3:DI-BPM-1",
        "SI-11C3:DI-BPM-2",
        "SI-11C4:DI-BPM",
        "SI-12M1:DI-BPM",
        "SI-12M2:DI-BPM",
        "SI-12C1:DI-BPM-1",
        "SI-12C1:DI-BPM-2",
        "SI-12C2:DI-BPM",
        "SI-12C3:DI-BPM-1",
        "SI-12C3:DI-BPM-2",
        "SI-12C4:DI-BPM",
        "SI-13M1:DI-BPM",
        "SI-13M2:DI-BPM",
        "SI-13C1:DI-BPM-1",
        "SI-13C1:DI-BPM-2",
        "SI-13C2:DI-BPM",
        "SI-13C3:DI-BPM-1",
        "SI-13C3:DI-BPM-2",
        "SI-13C4:DI-BPM",
        "SI-14M1:DI-BPM",
        "SI-14M2:DI-BPM",
        "SI-14C1:DI-BPM-1",
        "SI-14C1:DI-BPM-2",
        "SI-14C2:DI-BPM",
        "SI-14C3:DI-BPM-1",
        "SI-14C3:DI-BPM-2",
        "SI-14C4:DI-BPM",
        "SI-15M1:DI-BPM",
        "SI-15M2:DI-BPM",
        "SI-15C1:DI-BPM-1",
        "SI-15C1:DI-BPM-2",
        "SI-15C2:DI-BPM",
        "SI-15C3:DI-BPM-1",
        "SI-15C3:DI-BPM-2",
        "SI-15C4:DI-BPM",
        "SI-16M1:DI-BPM",
        "SI-16M2:DI-BPM",
        "SI-16C1:DI-BPM-1",
        "SI-16C1:DI-BPM-2",
        "SI-16C2:DI-BPM",
        "SI-16C3:DI-BPM-1",
        "SI-16C3:DI-BPM-2",
        "SI-16C4:DI-BPM",
        "SI-17M1:DI-BPM",
        "SI-17M2:DI-BPM",
        "SI-17C1:DI-BPM-1",
        "SI-17C1:DI-BPM-2",
        "SI-17C2:DI-BPM",
        "SI-17C3:DI-BPM-1",
        "SI-17C3:DI-BPM-2",
        "SI-17C4:DI-BPM",
        "SI-18M1:DI-BPM",
        "SI-18M2:DI-BPM",
        "SI-18C1:DI-BPM-1",
        "SI-18C1:DI-BPM-2",
        "SI-18C2:DI-BPM",
        "SI-18C3:DI-BPM-1",
        "SI-18C3:DI-BPM-2",
        "SI-18C4:DI-BPM",
        "SI-19M1:DI-BPM",
        "SI-19M2:DI-BPM",
        "SI-19C1:DI-BPM-1",
        "SI-19C1:DI-BPM-2",
        "SI-19C2:DI-BPM",
        "SI-19C3:DI-BPM-1",
        "SI-19C3:DI-BPM-2",
        "SI-19C4:DI-BPM",
        "SI-20M1:DI-BPM",
        "SI-20M2:DI-BPM",
        "SI-20C1:DI-BPM-1",
        "SI-20C1:DI-BPM-2",
        "SI-20C2:DI-BPM",
        "SI-20C3:DI-BPM-1",
        "SI-20C3:DI-BPM-2",
        "SI-20C4:DI-BPM",
        "SI-01M1:DI-BPM",
    )
    QUADNAMES = (
        "SI-01M2:PS-QS",
        "SI-01C1:PS-Q1",
        "SI-01C1:PS-QS",
        "SI-01C2:PS-QS",
        "SI-01C3:PS-Q4",
        "SI-01C3:PS-QS",
        "SI-01C4:PS-Q1",
        "SI-02M1:PS-QDB2",
        "SI-02M2:PS-QDB2",
        "SI-02C1:PS-Q1",
        "SI-02C1:PS-QS",
        "SI-02C2:PS-QS",
        "SI-02C3:PS-Q4",
        "SI-02C3:PS-QS",
        "SI-02C4:PS-Q1",
        "SI-03M1:PS-QDP2",
        "SI-03M2:PS-QDP2",
        "SI-03C1:PS-Q1",
        "SI-03C1:PS-QS",
        "SI-03C2:PS-QS",
        "SI-03C3:PS-Q4",
        "SI-03C3:PS-QS",
        "SI-03C4:PS-Q1",
        "SI-04M1:PS-QDB2",
        "SI-04M2:PS-QDB2",
        "SI-04C1:PS-Q1",
        "SI-04C1:PS-QS",
        "SI-04C2:PS-QS",
        "SI-04C3:PS-Q4",
        "SI-04C3:PS-QS",
        "SI-04C4:PS-Q1",
        "SI-05M1:PS-QS",
        "SI-05M2:PS-QS",
        "SI-05C1:PS-Q1",
        "SI-05C1:PS-QS",
        "SI-05C2:PS-QS",
        "SI-05C3:PS-Q4",
        "SI-05C3:PS-QS",
        "SI-05C4:PS-Q1",
        "SI-06M1:PS-QDB2",
        "SI-06M2:PS-QDB2",
        "SI-06C1:PS-Q1",
        "SI-06C1:PS-QS",
        "SI-06C2:PS-QS",
        "SI-06C3:PS-Q4",
        "SI-06C3:PS-QS",
        "SI-06C4:PS-Q1",
        "SI-07M1:PS-QDP2",
        "SI-07M2:PS-QDP2",
        "SI-07C1:PS-Q1",
        "SI-07C1:PS-QS",
        "SI-07C2:PS-QS",
        "SI-07C3:PS-Q4",
        "SI-07C3:PS-QS",
        "SI-07C4:PS-Q1",
        "SI-08M1:PS-QDB2",
        "SI-08M2:PS-QDB2",
        "SI-08C1:PS-Q1",
        "SI-08C1:PS-QS",
        "SI-08C2:PS-QS",
        "SI-08C3:PS-Q4",
        "SI-08C3:PS-QS",
        "SI-08C4:PS-Q1",
        "SI-09M1:PS-QS",
        "SI-09M2:PS-QS",
        "SI-09C1:PS-Q1",
        "SI-09C1:PS-QS",
        "SI-09C2:PS-QS",
        "SI-09C3:PS-Q4",
        "SI-09C3:PS-QS",
        "SI-09C4:PS-Q1",
        "SI-10M1:PS-QDB2",
        "SI-10M2:PS-QDB2",
        "SI-10C1:PS-Q1",
        "SI-10C1:PS-QS",
        "SI-10C2:PS-QS",
        "SI-10C3:PS-Q4",
        "SI-10C3:PS-QS",
        "SI-10C4:PS-Q1",
        "SI-11M1:PS-QDP2",
        "SI-11M2:PS-QDP2",
        "SI-11C1:PS-Q1",
        "SI-11C1:PS-QS",
        "SI-11C2:PS-QS",
        "SI-11C3:PS-Q4",
        "SI-11C3:PS-QS",
        "SI-11C4:PS-Q1",
        "SI-12M1:PS-QDB2",
        "SI-12M2:PS-QDB2",
        "SI-12C1:PS-Q1",
        "SI-12C1:PS-QS",
        "SI-12C2:PS-QS",
        "SI-12C3:PS-Q4",
        "SI-12C3:PS-QS",
        "SI-12C4:PS-Q1",
        "SI-13M1:PS-QS",
        "SI-13M2:PS-QS",
        "SI-13C1:PS-Q1",
        "SI-13C1:PS-QS",
        "SI-13C2:PS-QS",
        "SI-13C3:PS-Q4",
        "SI-13C3:PS-QS",
        "SI-13C4:PS-Q1",
        "SI-14M1:PS-QDB2",
        "SI-14M2:PS-QDB2",
        "SI-14C1:PS-Q1",
        "SI-14C1:PS-QS",
        "SI-14C2:PS-QS",
        "SI-14C3:PS-Q4",
        "SI-14C3:PS-QS",
        "SI-14C4:PS-Q1",
        "SI-15M1:PS-QDP2",
        "SI-15M2:PS-QDP2",
        "SI-15C1:PS-Q1",
        "SI-15C1:PS-QS",
        "SI-15C2:PS-QS",
        "SI-15C3:PS-Q4",
        "SI-15C3:PS-QS",
        "SI-15C4:PS-Q1",
        "SI-16M1:PS-QDB2",
        "SI-16M2:PS-QDB2",
        "SI-16C1:PS-Q1",
        "SI-16C1:PS-QS",
        "SI-16C2:PS-QS",
        "SI-16C3:PS-Q4",
        "SI-16C3:PS-QS",
        "SI-16C4:PS-Q1",
        "SI-17M1:PS-QS",
        "SI-17M2:PS-QS",
        "SI-17C1:PS-Q1",
        "SI-17C1:PS-QS",
        "SI-17C2:PS-QS",
        "SI-17C3:PS-Q4",
        "SI-17C3:PS-QS",
        "SI-17C4:PS-Q1",
        "SI-18M1:PS-QDB2",
        "SI-18M2:PS-QDB2",
        "SI-18C1:PS-Q1",
        "SI-18C1:PS-QS",
        "SI-18C2:PS-QS",
        "SI-18C3:PS-Q4",
        "SI-18C3:PS-QS",
        "SI-18C4:PS-Q1",
        "SI-19M1:PS-QDP2",
        "SI-19M2:PS-QDP2",
        "SI-19C1:PS-Q1",
        "SI-19C1:PS-QS",
        "SI-19C2:PS-QS",
        "SI-19C3:PS-Q4",
        "SI-19C3:PS-QS",
        "SI-19C4:PS-Q1",
        "SI-20M1:PS-QDB2",
        "SI-20M2:PS-QDB2",
        "SI-20C1:PS-Q1",
        "SI-20C1:PS-QS",
        "SI-20C2:PS-QS",
        "SI-20C3:PS-Q4",
        "SI-20C3:PS-QS",
        "SI-20C4:PS-Q1",
        "SI-01M1:PS-QS",
    )

    BPMNAMES = tuple([_PVName(bpm) for bpm in BPMNAMES])
    QUADNAMES = tuple([_PVName(quad) for quad in QUADNAMES])

    def __init__(self):
        """."""
        super().__init__()

        self.quad_deltakl = 0.01  # [1/m]

        self.wait_correctors = 0.3  # [s]
        self.wait_quadrupole = 0.3  # [s]
        self.timeout_wait_orbit = 3  # [s]

        self.corr_nr_iters = 6

        self.sofb_nrpoints = 10
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]

    def __str__(self):
        """."""
        return "..."


class DoParallelBBA(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=ParallelBBAParams(), target=self._do_pbba, isonline=isonline
        )
        self._groups2dopbba = list()
        self._activegroup_id = None
        self.clt_confdb = ConfigDBClient(config_type="si_bbadata")
        self.clt_confdb._TIMEOUT_DEFAULT = 20
        self.data["bpmnames"] = list(ParallelBBAParams.BPMNAMES)
        self.data["quadnames"] = list(ParallelBBAParams.QUADNAMES)
        self.data["scancenterx"] = _np.zeros(len(ParallelBBAParams.BPMNAMES))
        self.data["scancentery"] = _np.zeros(len(ParallelBBAParams.BPMNAMES))
        self.data["measure"] = dict()

        if self.isonline:
            self.devices["sofb"] = _SOFB(_SOFB.DEVICES.SI)
            self.devices["currinfosi"] = _CurrInfoSI()
            self.connect_to_quadrupoles()

    def __str__(self):
        """."""
        return "..."

    @property
    def havebeam(self):
        """."""
        haveb = self.devices["currinfosi"]
        return haveb.connected and haveb.storedbeam

    @property
    def measuredbpms(self):
        """."""
        return sorted(self.data["measure"])

    @property
    def groups2dopbba(self):
        return _dcopy(self._groups2dopbba)

    @groups2dopbba.setter
    def groups2dopbba(self, groups):
        self._groups2dopbba = [[_PVName(bpm) for bpm in group if isinstance(bpm, str)] for group in groups if isinstance(group, (list, tuple, _np.ndarray))]

    @property
    def active_group_id(self):
        return self._activegroup_id

    @active_group_id.setter
    def active_group_id(self, group_idx):
        if group_idx < len(self._groups2dopbba):
            self._activegroup_id = group_idx

    def connect_to_quadrupoles(self):
        """."""
        for bpm in self.groups2dopbba[self.active_group_id]:
            idx = self.data["bpmnames"].index(bpm)
            qname = self.data["quadnames"][idx]
            if qname and qname not in self.devices:
                self.devices[qname] = _PowerSupply(qname)

    def get_connected_quadnames(self):
        qnames = []
        for bpm in self.groups2dopbba[self.active_group_id]:
            idx = self.data["bpmnames"].index(bpm)
            qname = self.data["quadnames"][idx]
            if qname:
                qnames.append(qname)
        return qnames

    def get_connected_bpmnames(self):
        return self.groups2dopbba[self.active_group_id]

    def get_orbit(self):
        """."""
        sofb = self.devices["sofb"]
        sofb.cmd_reset()
        sofb.wait_buffer(self.params.timeout_wait_orbit)
        return _np.hstack([sofb.orbx, sofb.orby])

    def correct_orbit(self):
        """."""
        sofb = self.devices["sofb"]
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

    @staticmethod
    def get_default_quads(model, fam_data):
        """."""
        quads_idx = _dcopy(fam_data["QN"]["index"])
        qs_idx = [idx for idx in fam_data["QS"]["index"]]
        quads_idx.extend(qs_idx)
        quads_idx = _np.array([idx[len(idx) // 2] for idx in quads_idx])
        quads_pos = _np.array(_pyacc.lattice.find_spos(model, quads_idx))

        bpms_idx = _np.array([idx[0] for idx in fam_data["BPM"]["index"]])
        bpms_pos = _np.array(_pyacc.lattice.find_spos(model, bpms_idx))

        diff = _np.abs(bpms_pos[:, None] - quads_pos[None, :])
        bba_idx = _np.argmin(diff, axis=1)
        quads_bba_idx = quads_idx[bba_idx]
        bpmnames = list()
        qnames = list()
        for i, qidx in enumerate(quads_bba_idx):
            name = model[qidx].fam_name
            idc = fam_data[name]["index"].index([qidx])
            sub = fam_data[name]["subsection"][idc]
            inst = fam_data[name]["instance"][idc]
            name = "QS" if name.startswith(("S", "F")) else name
            qname = "SI-{0:s}:PS-{1:s}-{2:s}".format(sub, name, inst)
            qnames.append(qname.strip("-"))

            sub = fam_data["BPM"]["subsection"][i]
            inst = fam_data["BPM"]["instance"][i]
            bname = "SI-{0:s}:DI-BPM-{1:s}".format(sub, inst)
            bname = bname.strip("-")
            bpmnames.append(bname.strip("-"))
        return bpmnames, bpms_idx, qnames, quads_bba_idx

    @staticmethod
    def list_bpm_subsections(bpms):
        """."""
        return _DoBBA.list_bpm_subsections(bpms)

    # ##### Make Figures #####
    #! empty

    # #### pbba utils #####
    def get_group_delta_kl(self):
        group = self.groups2dopbba[self.active_group_id]
        delta_kl = _np.ones(len(group))*self.params.quad_deltakl
        delta_kl[1::2] *= -1
        return delta_kl

    def get_group_ordering(self):
        group = self.groups2dopbba[self.active_group_id]
        return _np.arange(len(group))

    def set_strengths(self, strengths, model=None, fam_data=None):
        order = self.get_group_ordering()
        group = self.groups2dopbba[self.active_group_id]
        if len(strengths) != len(group):
            raise ValueError('dim mismatch between the active group and "strengths".')
        if model is not None:
            bnames, _, qnames, qindices = self.get_default_quads(model, fam_data)
            for _o in order:
                _id = bnames.index(group[_o])
                qidx = qindices[_id]
                if "QS" in qnames[_id]:
                    model[qidx].KsL = strengths[_o]
                else:
                    model[qidx].KL = strengths[_o]
        else:
            qnames = self.data["quadnames"]
            bnames = self.data["bpmnames"]
            for _o in order:
                bname = group[_o]
                qname = qnames[bnames.index(bname)]
                quad = self.devices[qname]
                quad.strength = strengths[_o]
                _time.sleep(self.params.wait_quadrupole)

    def get_strengths(self, model=None, fam_data=None):
        order = self.get_group_ordering()
        group = self.groups2dopbba[self.active_group_id]
        strengths = _np.zeros(len(group))
        if model is not None:
            bnames, _, qnames, qindices = self.get_default_quads(model, fam_data)
            for _o in order:
                _id = bnames.index(group[_o])
                qidx = qindices[_id]
                if "QS" in qnames[_id]:
                    strengths[_o] = model[qidx].KsL
                else:
                    strengths[_o] = model[qidx].KL
        else:
            qnames = self.data["quadnames"]
            bnames = self.data["bpmnames"]
            for _o in order:
                bname = group[_o]
                qname = qnames[bnames.index(bname)]
                quad = self.devices[qname]
                strengths[_o] = quad.strength
        return strengths

    def calc_ios(self, model, fam_data):
        if any([model.cavity_on, model.radiation_on > 0]):
            findorbit = _pyacc.tracking.find_orbit6
        else:
            findorbit = _pyacc.tracking.find_orbit4
        bpmidx = _np.array(fam_data['BPM']['index']).ravel()
        delta_stren = self.get_group_delta_kl()
        strens_orig = self.get_strengths(model, fam_data)
        self.set_strengths(strens_orig + delta_stren/2, model, fam_data)
        try:
            orb_pos = findorbit(model, indices=bpmidx)
        except Exception as E:
            self.set_strengths(strens_orig, model, fam_data)
            print(E)
            return
        self.set_strengths(strens_orig - delta_stren/2, model, fam_data)
        try:
            orb_neg = findorbit(model, indices=bpmidx)
        except Exception as E:
            self.set_strengths(strens_orig, model, fam_data)
            print(E)
            return
        self.set_strengths(strens_orig, model, fam_data)
        orb_pos = _np.hstack((orb_pos[0,:], orb_pos[2,:]))
        orb_neg = _np.hstack((orb_neg[0,:], orb_neg[2,:]))
        return orb_pos - orb_neg

    def meas_ios(self):
        def get_orbit():
            orbx = self.devices['soft'].orbx
            orby = self.devices['soft'].orby
            return _np.hstack((orbx, orby))
        delta_stren = self.get_group_delta_kl()
        strens_orig = self.get_strengths()
        self.set_strengths(strens_orig + delta_stren/2)
        orb_pos = get_orbit()
        self.set_strengths(strens_orig + delta_stren/2)
        orb_neg = get_orbit()
        self.set_strengths(strens_orig)
        return orb_pos - orb_neg

    def calc_ios_jacobian(self, model, fam_data):
        use6d = any([model.cavity_on, model.radiation_on>0])
        _orbcorr = _OrbitCorr(model=model, acc='SI', corr_system='SOFB', use6dorb=use6d)
        delta_stren = self.get_group_delta_kl()
        strens_orig = self.get_strengths(model, fam_data)
        self.set_strengths(strens_orig + delta_stren/2, model, fam_data)
        try:
            jac_pos = _orbcorr.get_jacobian_matrix()
        except Exception as E:
            self.set_strengths(strens_orig, model, fam_data)
            print("ggwp")
            return
        self.set_strengths(strens_orig - delta_stren/2, model, fam_data)
        try:
            jac_neg = _orbcorr.get_jacobian_matrix()
        except Exception as E:
            self.set_strengths(strens_orig, model, fam_data)
            print("ggez")
            return
        self.set_strengths(strens_orig, model, fam_data)
        return jac_pos - jac_neg

    def meas_ios_jacobian(self):
        raise NotImplementedError('Not available yet. Use \'calc_ios_jacobian\' for a model-based jacobian.')

    @staticmethod
    def inverse_matrix(matrix):
        return _np.dot(_np.linalg.pinv(_np.dot(matrix.T, matrix), rcond=1e-8), matrix.T)

    def correct_ios(self, jacobian=None, model=None, fam_data=None):
        if jacobian is None:
            jacobian = self.calc_ios_jacobian(model, fam_data)
        inverse_jacobian = self.inverse_matrix(jacobian)
        ios_evo = []
        nr_iters = self.params.corr_nr_iters
        for i in range(nr_iters+1):
            if model is not None:
                ios = self.calc_ios(model, fam_data)
            else:
                ios = self.meas_ios()
            ios_evo.append(ios)
            if i >= nr_iters:
                break
            dkicks = list(-1 * _np.dot(inverse_jacobian, ios))
            if model is not None:
                use6d = any([model.cavity_on, model.radiation_on>0])
                _orbcorr = _OrbitCorr(model=model, acc='SI', corr_system='SOFB', use6dorb=use6d)
                _orbcorr.set_delta_kicks(dkicks)
            else:
                sofb = self.devices['sofb']
                sofb.APPLY_DELTA_KICKS(dkicks) #!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.data[self.active_group_id] = {'ios': ios_evo}
        return ios_evo


    # #### private methods ####
    def _do_pbba(self):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        print(
            "Starting measurement at {:s}".format(
                tini.strftime("%Y-%m-%d %Hh%Mm%Ss")
            )
        )

        sofb = self.devices["sofb"]
        sofb.nr_points = self.params.sofb_nrpoints
        loop_on = False
        if sofb.autocorrsts:
            loop_on = True
            print("SOFB feedback is enable, disabling it...")
            sofb.cmd_turn_off_autocorr()

        for i in range(len(self.groups2dopbba)):
            if self._stopevt.is_set():
                print("stopped!")
                break
            if not self.havebeam:
                print("Beam was Lost")
                break
            print("\nCorrecting Orbit...", end="")
            self.correct_orbit()
            print("Ok!")
            self._dopbba_single_group(i)

        if loop_on:
            print("SOFB feedback was enable, restoring original state...")
            sofb.cmd_turn_on_autocorr()

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split(".")[0]
        print("finished! Elapsed time {:s}".format(dtime))

    def _dopbba_single_group(self, group_id):
        """."""
        self.activegroup_id = group_id
        bpmnames = self.get_connected_bpmnames()
        self.connect_to_quadrupoles()
        quadnames = self.get_connected_quadnames()

        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime("%Hh%Mm%Ss")
        print("{:s} --> Doing PBBA for Group {:d}".format(strtini))

        sofb = self.devices["sofb"]

        korig = []
        for quadname in quadnames:
            _quad = self.devices[quadname]
            if not _quad.pwrstate:
                print("\n    error: quadrupole " + quadname + " is Off.")
                self._stopevt.set()
                print("    exiting...")
                return
            korig.append(_quad.strength)

        deltakl = self.get_group_delta_kl(self.active_group_id)
        ios_jac = self.get_ios_jacobian(
            self._model, self.groups2dopbba[self.active_group_id]
        )

        nrsteps = self.params.corr_nr_iters
        refx0, refy0 = sofb.refx.copy(), sofb.refy.copy()

        for i in range(nrsteps):
            if self._stopevt.is_set() or not self.havebeam:
                print("   exiting...")
                break
            print("    {0:02d}/{1:02d} --> ".format(i + 1, nrsteps), end="")

            res = self.correct_ios_single_iter(
                jacobian=ios_jac
            )

        print("restoring initial conditions.")

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split(".")[0]
        print("Done! Elapsed time: {:s}\n".format(dtime))
