"""Pynel package."""

from . import std_si_data
from . import misc_functions as functions
from .buttons import Button
from .base import Base
from . import fitting

#from . import std_si_data
#from . import last_complete_pynel as completePynel

import os as _os
with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()

__all__ = ['Base', 'Button', 'functions', 'fitting', 'std_si_data']
