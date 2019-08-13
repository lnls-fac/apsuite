#!/usr/bin/env python-sirius
"""."""

import epics
import numpy as np
import time

step_ch = 0.1
step_cv = 0.1
step_qf = 0.1
step_qd = 0.1
noise_level = 5000

codx = epics.PV('BO-Glob:AP-SOFB:OrbitSmoothSinglePassX-Mon')
cody = epics.PV('BO-Glob:AP-SOFB:OrbitSmoothSinglePassY-Mon')
cods = epics.PV('BO-Glob:AP-SOFB:OrbitSmoothSinglePassSum-Mon')

ch = [
    'BO-01U:PS-CH', 'BO-03U:PS-CH', 'BO-05U:PS-CH', 'BO-07U:PS-CH',
    'BO-09U:PS-CH', 'BO-11U:PS-CH', 'BO-13U:PS-CH', 'BO-15U:PS-CH',
    'BO-17U:PS-CH', 'BO-19U:PS-CH', 'BO-21U:PS-CH', 'BO-23U:PS-CH',
    'BO-25U:PS-CH', 'BO-27U:PS-CH', 'BO-29U:PS-CH', 'BO-31U:PS-CH',
    'BO-33U:PS-CH', 'BO-35U:PS-CH', 'BO-37U:PS-CH', 'BO-39U:PS-CH',
    'BO-41U:PS-CH', 'BO-43U:PS-CH', 'BO-45U:PS-CH', 'BO-47U:PS-CH',
    'BO-49D:PS-CH',
]
cv = [
    'BO-01U:PS-CV', 'BO-03U:PS-CV', 'BO-05U:PS-CV', 'BO-07U:PS-CV',
    'BO-09U:PS-CV', 'BO-11U:PS-CV', 'BO-13U:PS-CV', 'BO-15U:PS-CV',
    'BO-17U:PS-CV', 'BO-19U:PS-CV', 'BO-21U:PS-CV', 'BO-23U:PS-CV',
    'BO-25U:PS-CV', 'BO-27U:PS-CV', 'BO-29U:PS-CV', 'BO-31U:PS-CV',
    'BO-33U:PS-CV', 'BO-35U:PS-CV', 'BO-37U:PS-CV', 'BO-39U:PS-CV',
    'BO-41U:PS-CV', 'BO-43U:PS-CV', 'BO-45U:PS-CV', 'BO-47U:PS-CV',
    'BO-49U:PS-CV',
]

pvs_ch = [epics.PV(v + ':Current-SP') for v in ch]
pvs_cv = [epics.PV(v + ':Current-SP') for v in cv]
pvs_qf = epics.PV('BO-Fam:PS-QF:Current-SP')
pvs_qd = epics.PV('BO-Fam:PS-QD:Current-SP')


def get_corr_sp():
    """."""
    values_ch = [pvs_ch[i].value for i in range(len(ch))]
    values_cv = [pvs_cv[i].value for i in range(len(cv))]
    return values_ch, values_cv


def get_quad_sp():
    """."""
    values_qf = pvs_qf.value
    values_qd = pvs_qd.value
    return values_qf, values_qd


def set_corr_sp(v_ch, v_cv):
    """."""
    # return
    for i in range(len(ch)):
        pvs_ch[i].value = v_ch[i]
    for i in range(len(cv)):
        pvs_cv[i].value = v_cv[i]


def set_quad_sp(v_qf, v_qd):
    """."""
    pvs_qd.value = v_qf
    pvs_qd.value = v_qd


def new_corr_sp(v1_ch, v1_cv):
    """."""
    delta_ch = step_ch * (np.random.rand(len(v1_ch)) - 0.5)
    delta_cv = step_cv * (np.random.rand(len(v1_cv)) - 0.5)
    v2_ch = v1_ch + delta_ch
    v2_cv = v1_cv + delta_cv
    return v2_ch, v2_cv


def new_quad_sp(v1_qf, v1_qd):
    """."""
    delta_qf = step_qf * (np.random.rand(len(v1_qf)) - 0.5)
    delta_qd = step_qd * (np.random.rand(len(v1_qd)) - 0.5)
    v2_qf = v1_qf + delta_qf
    v2_qd = v1_qd + delta_qd
    return v2_qf, v2_qd


def calc_merit1():
    """."""
    s = np.array(cods.value)
    m = sum(s)/25
    return m


def calc_merit2():
    """."""
    x = np.array(codx.value)
    y = np.array(cody.value)
    s = np.array(cods.value)
    n, m = 0, 0.0
    for i in range(len(s)):
        if v > noise_level:
            n += 1
            m += 1*x[i]*x[i] + 0*y[i]*y[i]
    m /= n**2
    return 1/m


# a, b = get_corr_sp()
# print(a)


def run(nrpts):
    """."""
    iter = 0
    calc_merit = calc_merit1
    cur_m = calc_merit()
    cur_ch, cur_cv = get_corr_sp()
    cur_qf, cur_qd = get_quad_sp()
    print('* {}: {}'.format(iter, cur_m))
    while iter < nrpts:
        iter += 1
        new_ch, new_cv = new_corr_sp(cur_ch, cur_cv)
        new_qf, new_qd = new_quad_sp(cur_qf, cur_qd)
        set_corr_sp(new_ch, new_cv)
        set_quad_sp(new_qf, new_qd)
        time.sleep(0.51)
        new_m = calc_merit()
        if new_m > cur_m:
            print('* {}: {}'.format(iter, new_m))
            cur_m = new_m
            cur_ch, cur_cv = new_ch, new_cv
            cur_qf, cur_qd = new_qf, new_qd
        else:
            set_corr_sp(cur_ch, cur_cv)
            set_quad_sp(cur_qf, cur_qd)
            print('  {}'.format(iter))
        # nb = input('enter for next...')
