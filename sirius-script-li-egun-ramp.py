#!/usr/bin/env python-sirius
"""."""

import time as _time
from epics import PV as _PV

egun_hvv = _PV('egun:hvps:voltoutsoft')
egun_hvv_rb = _PV('egun:hvps:voltinsoft')
egun_hvc = _PV('egun:hvps:currentoutsoft')
egun_hve = _PV('egun:hvps:enable')
egun_hves = _PV('egun:hvps:enstatus')
egun_filaps = _PV('egun:filaps:currentoutsoft')
egun_biasps = _PV('egun:biasps:voltoutsoft')
egun_permit = _PV('LA-CN:H1MPS-1:GunPermit')
ccg1_va = _PV('LA-VA:H1VGC-01:RdPrs-1')
li_alarm = _PV('LA-CN:H1MPS-1:LAAlarm')

names = ['LA-CN:H1MPS-1:CCG1', 'LA-CN:H1MPS-1:CCG2']
resets = []
sts_ou = []
sts_in = []
for name in names:
    resets.append(_PV(name + 'Warn_R'))
    sts_ou.append(_PV(name + 'Warn_L'))
    sts_in.append(_PV(name + 'Warn_I'))


goal_volt = 80  # [kV]
goal_pressure = 8.0e-9


def main_loop():
    """."""
    for i in range(100):
        print()
        print('ATTEMPT {0:04d}'.format(i))
        reset_interlocks()
        if not check_ok():
            print('Error, could not reset all interlocks.')
            t0 = _time.time()
            while True:
                print('\a')  # make a sound
                if _time.time()-t0 > 20:
                    break
            return
        turnon_egun()
        max_hv = 0
        while check_ok():
            max_hv = max(max_hv, egun_hvv_rb.value)
            _time.sleep(0.1)
        strtime = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime())
        print('{0:s} Vacuum Interlock!'.format(strtime))
        print('Max HV was {0:.4f}'.format(max_hv))


def check_ok():
    """."""
    isok = [inn.value == 0 for inn in sts_in]
    return all(isok) and egun_permit.value == 1


def turnon_egun():
    """."""
    print('Preparing Egun')
    egun_biasps.value = -38  # 3nC
    egun_filaps.value = 1.45
    if egun_hves.value == 0:
        egun_hve.value = 1
    egun_hvc.value = 0.008

    print('Waiting for vacuum recovery...')
    while ccg1_va.value > goal_pressure:
        print('    Over pressure: {0:.3f}'.format(ccg1_va.value/goal_pressure))
        _time.sleep(2)

    print('Setting HV from {0:.2f} to {1:.2f} kV'.format(
                                        egun_hvv_rb.value, goal_volt))
    egun_hvv.value = goal_volt


def reset_interlocks():
    """."""
    print('Reseting interlocks...')
    for rst, out, inn in zip(resets, sts_ou, sts_in):
        while inn.value != 0:
            _time.sleep(0.5)
        _time.sleep(0.5)
        rst.value = 1
        _time.sleep(1.0)
        rst.value = 0
    _time.sleep(2)


if __name__ == '__main__':
    main_loop()
