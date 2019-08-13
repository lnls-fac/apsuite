#!/usr/bin/env python-sirius
"""."""

import matplotlib.pyplot as plt
import numpy as np


def read_file(fname):
    """."""
    with open(fname, 'r') as fp:
        text = fp.readlines()
    data = []
    for line in text:
        words = line.split()
        data.append(float(words[-1]))
    w1 = text[0].split()
    w2 = text[-1].split()
    print(fname, w1[-2], w2[-2])
    return np.array(data)


sd = read_file('sd.txt')
sf = read_file('sf.txt')
qd = read_file('qd.txt')
qf = read_file('qf.txt')
tsd = 0.1*np.array(range(0, len(sd)))
tsf = 0.1*np.array(range(0, len(sf)))
tqd = 0.1*np.array(range(0, len(qd)))
tqf = 0.1*np.array(range(0, len(qf)))

total_time = 42  # [minutes]
tsd *= total_time*60/tsd[-1]
tsf *= total_time*60/tsf[-1]
tqd *= total_time*60/tqd[-1]
tqf *= total_time*60/tqf[-1]

plt.plot(tsd/60, 1e6*(sd - np.mean(sd))/150, label='SD')
plt.plot(tsf/60, 1e6*(sf - np.mean(sf))/150, label='SF')
plt.plot(tqd/60, 1e6*(qd - np.mean(qd))/150, label='QD')
plt.plot(tqf/60, 1e6*(qf - np.mean(qf))/150, label='QF')
plt.legend()
plt.xlabel('Time [minutes]')
plt.ylabel('Current Variation [ppm of 150A]')
plt.grid()
plt.show()
