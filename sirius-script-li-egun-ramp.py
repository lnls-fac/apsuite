#!/usr/bin/env python-sirius
"""."""

import time as _time
from datetime import timedelta as _timedelta
from epics import PV as _PV

egun_hvv = _PV('egun:hvps:voltoutsoft')
egun_hvv_rb = _PV('egun:hvps:voltinsoft')
egun_hvc = _PV('egun:hvps:currentoutsoft')
egun_hve = _PV('egun:hvps:enable')
egun_hves = _PV('egun:hvps:enstatus')
egun_filaps = _PV('egun:filaps:currentoutsoft')
egun_biasps = _PV('egun:biasps:voltoutsoft')
egun_trigger = _PV('egun:triggerps:enable')
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


# def main_loop():
#     if keep_egun_on():
#         exit_()
#         return


def main_loop():
    """."""
    # time_off = [1, 5, 10, 20, 30, 60, 120, 180, 240, 300, 360]  # in minutes
    time_off = [16*60, ]  # in minutes
    time_on = len(time_off) * [20, ]  # in minutes
    filament_hot = True

    for ton, toff in zip(time_on, time_off):
        print('Turning Egun OFF!')
        control_egun(turn_on=False, filahot=filament_hot)
        print('Waiting {0:.1f} minutes'.format(toff))
        for i in range((toff*60)):  # convert to seconds
            dur = _timedelta(minutes=toff-i/60)
            print('Remaining Time {0:s}'.format(str(dur)), end='\r')
            _time.sleep(1)
        print(90*' ')
        if not check_ok():
            print('Error, some problem happened.')
            exit_()
            return
        print('Turning Egun ON!')
        if keep_egun_on(period=ton):
            exit_()
            return
        print('\n')


def keep_egun_on(period=-1):
    """."""
    for i in range(100):
        print('ATTEMPT {0:04d}'.format(i))
        reset_interlocks()
        if not check_ok():
            print('Error, could not reset all interlocks.')
            return 1

        control_egun()

        max_hv = 0
        tini = _time.time()
        while check_ok():
            max_hv = max(max_hv, egun_hvv_rb.value)
            _time.sleep(0.1)
            tfin = _time.time()
            dur = (tfin-tini)/60
            if 0 < period < dur:
                print(
                    'Success! No vacuum breakdown in the last ' +
                    '{0:.1f} minutes'.format(period))
                return 0
            else:
                if period >= 0:
                    dur = _timedelta(minutes=period - dur)
                    print('Remaining Time: {0:s}'.format(str(dur)), end='\r')
                else:
                    dur = _timedelta(minutes=dur)
                    print('Elapsed Time: {0:s}'.format(str(dur)), end='\r')

        strtime = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime())
        print('{0:s} --> Vacuum Interlock!'.format(strtime))
        print('Max HV was {0:.4f}'.format(max_hv))
    print('Problem! Too many vacuum breakdowns.')
    return 1


def check_ok():
    """."""
    isok = [inn.value == 0 for inn in sts_in]
    return all(isok) and egun_permit.value == 1


def control_egun(turn_on=True, filahot=True):
    """."""
    print('Preparing Egun')
    filahot |= turn_on
    egun_biasps.value = -38  # 3nC
    egun_filaps.value = 1.45 if filahot else 0.0
    egun_trigger.value = turn_on
    if egun_hves.value == 0:
        egun_hve.value = 1
    egun_hvc.value = 0.008

    cnt = 0
    std = '    '
    while ccg1_va.value > goal_pressure:
        if not cnt:
            print('Waiting for vacuum recovery...')
            print('Over pressure: ')
        cnt += 1
        std = '{0:.3f} '.format(ccg1_va.value/goal_pressure)
        print(std, end='\r')
        if not cnt % 10:
            std = '    '
            print()
        _time.sleep(2)

    volt = goal_volt if turn_on else 0.0
    print('Setting HV from {0:.2f} to {1:.2f} kV'.format(
        egun_hvv_rb.value, volt))
    egun_hvv.value = volt


def reset_interlocks():
    """."""
    print('Reseting interlocks...')
    for rst, inn in zip(resets, sts_in):
        while inn.value != 0:
            _time.sleep(0.5)
        _time.sleep(0.5)
        rst.value = 1
        _time.sleep(1.0)
        rst.value = 0
    _time.sleep(2)


def exit_():
    """."""
    t0 = _time.time()
    while True:
        print('\a')  # make a sound
        if _time.time()-t0 > 10:
            break
    print('I quit!')


if __name__ == '__main__':
    main_loop()
