#!/usr/local/env python-sirius
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
ccg1_in = _PV('LA-CN:H1MPS-1:CCG1Warn_I')
ccg2_in = _PV('LA-CN:H1MPS-1:CCG2Warn_I')
ccg1_rst = _PV('LA-CN:H1MPS-1:CCG1Warn_R')
ccg2_rst = _PV('LA-CN:H1MPS-1:CCG2Warn_R')


def main_loop():
    """."""
    for i in range(100):
        print()
        print('ATTEMPT {0:04d}'.format(i))
        reset_interlocks()
        if not check_ok():
            print('Error, could not reset all interlocks.')
            return
        volt = 80
        turnon_egun(volt)
        while check_ok():
            _time.sleep(0.002)
        print('Vacuum Interlock! HV was in {0:.4f}'.format(egun_hvv_rb.value))


def check_ok():
    """."""
    return egun_permit.value == 1


def turnon_egun(volt):
    """."""
    print('Preparing Egun')
    egun_biasps.value = -40
    egun_filaps.value = 1.45
    if egun_hves.value == 0:
        egun_hve.value = 1
    egun_hvc.value = 0.008

    print('Waiting for vacuum recovery...')
    goal = 8.0e-9
    while ccg1_va.value > goal:
        print('    Over pressure: {0:.4f}'.format(ccg1_va.value/goal))
        _time.sleep(2)

    print('Setting HV from {0:.2f} to {1:.2f} kV'.format(egun_hvv.value, volt))
    egun_hvv.value = volt


def reset_interlocks():
    """."""
    print('Reseting interlocks...')
    while ccg1_in.value != 0 or ccg2_in.value != 0:
        _time.sleep(0.5)

    ccg1_rst.value = 1
    ccg2_rst.value = 1
    _time.sleep(0.5)
    ccg1_rst.value = 0
    ccg2_rst.value = 0
    _time.sleep(2)


if __name__ == '__main__':
    main_loop()
