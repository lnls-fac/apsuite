"""Main module."""

import datetime as _datetime
import time as _time
from copy import deepcopy as _dcopy

from siriuspy.devices import (
    CurrInfoSI as _CurrInfoSI,
    PowerSupply as _PowerSupply,
    SOFB as _SOFB,
)

from siriuspy.namesys import SiriusPVName as _PVName

from ..utils import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)

from .measure_bba import BBAParams as _BBAParams


class ACBBAParams(_ParamsBaseClass):
    """."""

    BPMNAMES = _BBAParams.BPMNAMES
    QUADNAMES = _BBAParams.QUADNAMES

    def __init__(self):
        """."""
        super().__init__()

    def __str__(self):
        """."""
        stg = ""
        return stg


class DoACBBA(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        self.params = ACBBAParams()
        super().__init__(
            params=self.params, target=self._do_acbba, isonline=isonline
        )

    def _do_acbba(self):
        pass
