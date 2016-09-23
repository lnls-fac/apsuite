
#from . import lattice_errors
#from . import matching
from . import trackcpp_utils
from . import lifetime_measurement

import os as _os
with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
