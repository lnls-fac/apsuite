
__all__ = (
    'commissioning_scripts', 'loco', 'optics_analysis', 'optimization',
    'trackcpp_utils')

import os as _os
with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
