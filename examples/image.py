#!/usr/bin/env python-sirius

import matplotlib.pyplot as _plt

from siriuspy.devices import DVF
from apsuite.image import Image


dvf = DVF(DVF.DEVICES.CAX_DVF2)
img = Image(data=dvf.image)

fig, axis = img.imshow()

_plt.show()


