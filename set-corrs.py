#!/usr/bin/env python-sirius
"""."""

import epics
import time

chs = [
 'BO-01U:MA-CH',
 'BO-03U:MA-CH',
 'BO-05U:MA-CH',
 'BO-07U:MA-CH',
 'BO-09U:MA-CH',
 'BO-11U:MA-CH',
 'BO-13U:MA-CH',
 'BO-15U:MA-CH',
 'BO-17U:MA-CH',
 'BO-19U:MA-CH',
 'BO-21U:MA-CH',
 'BO-23U:MA-CH',
 'BO-25U:MA-CH',
 'BO-27U:MA-CH',
 'BO-29U:MA-CH',
 'BO-31U:MA-CH',
 'BO-33U:MA-CH',
 'BO-35U:MA-CH',
 'BO-37U:MA-CH',
 'BO-39U:MA-CH',
 'BO-41U:MA-CH',
 'BO-43U:MA-CH',
 'BO-45U:MA-CH',
 'BO-47U:MA-CH',
 'BO-49D:MA-CH']

pvs_sp = {c: epics.PV(c + ':Kick-SP') for c in chs}
pvs_rb = {c: epics.PV(c + ':Kick-RB') for c in chs}
pvs_delta = epics.PV('BO-Glob:AP-SOFB:DeltaKicksCH-Mon')


delta = pvs_delta.value
print(delta)

for i in range(len(chs)):
    ov = pvs_sp[chs[i]].value
    nv = pvs_sp[chs[i]].value + delta[i]
    print(chs[i], ov, nv)
    r = input('write?')
    pvs_sp[chs[i]].value = pvs_sp[chs[i]].value + delta[i]
