#!/usr/bin/env python-sirius
"""."""

# import epics

tb_b_iocnames = ['TB-Fam:PS-B']
tb_b_properties = (
    'Abort-Cmd BSMPComm-Sel BSMPComm-Sts CtrlLoop-Sel CtrlLoop-Sts '
    'CtrlMode-Mon Current-Mon Current-RB Current-SP Current1-Mon '
    'Current2-Mon CurrentRef-Mon CycleAmpl-RB CycleAmpl-SP '
    'CycleAuxParam-RB CycleAuxParam-SP CycleEnbl-Mon CycleFreq-RB '
    'CycleFreq-SP CycleIndex-Mon CycleNrCycles-RB CycleNrCycles-SP '
    'CycleOffset-RB CycleOffset-SP CycleType-Sel CycleType-Sts '
    'IntlkHard-Mon IntlkHardLabels-Cte IntlkSoft-Mon IntlkSoftLabels-Cte '
    'OpMode-Sel OpMode-Sts PRUBlockIndex-Mon PRUCtrlQueueSize-Mon '
    'PRUSyncMode-Mon PRUSyncPulseCount-Mon Properties-Cte PwrState-Sel '
    'PwrState-Sts Reset-Cmd RmpIncNrCycles-Mon RmpIncNrCycles-RB '
    'RmpIncNrCycles-SP RmpReady-Mon Version-Cte WfmData-RB '
    'WfmData-SP WfmIndex-Mon')

tb_quad_iocnames = [
    'TB-01:PS-QD1',
    'TB-01:PS-QF1',
    'TB-02:PS-QD2A',
    'TB-02:PS-QF2A',
    'TB-02:PS-QD2B',
    'TB-02:PS-QF2B',
    'TB-03:PS-QD3A',
    'TB-03:PS-QF3A',
    'TB-04:PS-QD4B',
    'TB-04:PS-QF4B']
tb_quad_properties = (
    'Abort-Cmd BSMPComm-Sel BSMPComm-Sts CtrlLoop-Sel CtrlLoop-Sts '
    'CtrlMode-Mon Current-Mon Current-RB Current-SP CurrentRef-Mon '
    'CycleAmpl-RB CycleAmpl-SP CycleAuxParam-RB CycleAuxParam-SP '
    'CycleEnbl-Mon CycleFreq-RB CycleFreq-SP CycleIndex-Mon '
    'CycleNrCycles-RB CycleNrCycles-SP CycleOffset-RB CycleOffset-SP '
    'CycleType-Sel CycleType-Sts IntlkHard-Mon IntlkHardLabels-Cte '
    'IntlkSoft-Mon IntlkSoftLabels-Cte OpMode-Sel OpMode-Sts '
    'PRUBlockIndex-Mon PRUCtrlQueueSize-Mon PRUSyncMode-Mon '
    'PRUSyncPulseCount-Mon Properties-Cte PwrState-Sel PwrState-Sts '
    'Reset-Cmd RmpIncNrCycles-Mon RmpIncNrCycles-RB RmpIncNrCycles-SP '
    'RmpReady-Mon Version-Cte WfmData-RB WfmData-SP WfmIndex-Mon')

tb_corr_iocnames = [
    'TB-01:PS-CH-1',
    'TB-01:PS-CV-1',
    'TB-01:PS-CH-2',
    'TB-01:PS-CV-2',
    'TB-02:PS-CH-1',
    'TB-02:PS-CV-1',
    'TB-02:PS-CH-2',
    'TB-02:PS-CV-2',
    'TB-03:PS-CH',
    'TB-04:PS-CV-1',
    'TB-04:PS-CV-2']
tb_corr_properties = (
    'Abort-Cmd BSMPComm-Sel BSMPComm-Sts CtrlLoop-Sel CtrlLoop-Sts '
    'CtrlMode-Mon Current-Mon Current-RB Current-SP CurrentRef-Mon '
    'CycleAmpl-RB CycleAmpl-SP CycleAuxParam-RB CycleAuxParam-SP '
    'CycleEnbl-Mon CycleFreq-RB CycleFreq-SP CycleIndex-Mon '
    'CycleNrCycles-RB CycleNrCycles-SP CycleOffset-RB CycleOffset-SP '
    'CycleType-Sel CycleType-Sts IntlkHard-Mon IntlkHardLabels-Cte '
    'IntlkSoft-Mon IntlkSoftLabels-Cte OpMode-Sel OpMode-Sts '
    'PRUBlockIndex-Mon PRUCtrlQueueSize-Mon PRUSyncMode-Mon '
    'PRUSyncPulseCount-Mon Properties-Cte PwrState-Sel PwrState-Sts '
    'Reset-Cmd RmpIncNrCycles-Mon RmpIncNrCycles-RB RmpIncNrCycles-SP '
    'RmpReady-Mon Version-Cte WfmData-RB WfmData-SP WfmIndex-Mon')

bo_b_iocnames = [
    'BO-Fam:PS-B-1',
    'BO-Fam:PS-B-2']
bo_b_properties = (
    'Abort-Cmd Arm1Current-Mon Arm2Current-Mon BSMPComm-Sel BSMPComm-Sts '
    'CapacitorBank1Voltage-Mon CapacitorBank2Voltage-Mon '
    'CapacitorBank3Voltage-Mon CapacitorBank4Voltage-Mon '
    'CapacitorBank5Voltage-Mon CapacitorBank6Voltage-Mon '
    'CapacitorBank7Voltage-Mon CapacitorBank8Voltage-Mon '
    'CtrlLoop-Sel CtrlLoop-Sts CtrlMode-Mon Current-Mon '
    'Current-RB Current-SP Current1-Mon Current2-Mon CurrentRef-Mon '
    'CycleAmpl-RB CycleAmpl-SP CycleAuxParam-RB CycleAuxParam-SP '
    'CycleEnbl-Mon CycleFreq-RB CycleFreq-SP CycleIndex-Mon CycleNrCycles-RB '
    'CycleNrCycles-SP CycleOffset-RB CycleOffset-SP CycleType-Sel '
    'CycleType-Sts IntlkHard-Mon IntlkHardLabels-Cte IntlkSoft-Mon '
    'IntlkSoftLabels-Cte LoadVoltage-Mon Module1Voltage-Mon '
    'Module2Voltage-Mon Module3Voltage-Mon Module4Voltage-Mon '
    'Module5Voltage-Mon Module6Voltage-Mon Module7Voltage-Mon '
    'Module8Voltage-Mon OpMode-Sel OpMode-Sts PRUBlockIndex-Mon '
    'PRUCtrlQueueSize-Mon PRUSyncMode-Mon PRUSyncPulseCount-Mon '
    'PWMDutyCycle1-Mon PWMDutyCycle2-Mon PWMDutyCycle3-Mon PWMDutyCycle4-Mon '
    'PWMDutyCycle5-Mon PWMDutyCycle6-Mon PWMDutyCycle7-Mon PWMDutyCycle8-Mon '
    'Properties-Cte PwrState-Sel PwrState-Sts Reset-Cmd RmpIncNrCycles-Mon '
    'RmpIncNrCycles-RB RmpIncNrCycles-SP RmpReady-Mon Version-Cte WfmData-RB '
    'WfmData-SP WfmIndex-Mon')

bo_qd_sx_sd_iocnames = [
    'BO-Fam:PS-SD',
    'BO-Fam:PS-SF',
    'BO-Fam:PS-QD']
bo_qd_sx_sd_properties = (
    'Abort-Cmd BSMPComm-Sel BSMPComm-Sts CtrlLoop-Sel CtrlLoop-Sts '
    'CtrlMode-Mon Current-Mon Current-RB Current-SP Current1-Mon Current2-Mon '
    'CurrentRef-Mon CycleAmpl-RB CycleAmpl-SP CycleAuxParam-RB '
    'CycleAuxParam-SP CycleEnbl-Mon CycleFreq-RB CycleFreq-SP CycleIndex-Mon '
    'CycleNrCycles-RB CycleNrCycles-SP CycleOffset-RB CycleOffset-SP '
    'CycleType-Sel CycleType-Sts IntlkHard-Mon IntlkHardLabels-Cte '
    'IntlkSoft-Mon IntlkSoftLabels-Cte OpMode-Sel OpMode-Sts '
    'PRUBlockIndex-Mon PRUCtrlQueueSize-Mon PRUSyncMode-Mon '
    'PRUSyncPulseCount-Mon Properties-Cte PwrState-Sel PwrState-Sts '
    'Reset-Cmd RmpIncNrCycles-Mon RmpIncNrCycles-RB RmpIncNrCycles-SP '
    'RmpReady-Mon Version-Cte WfmData-RB WfmData-SP WfmIndex-Mon')

bo_qf_iocnames = ['BO-Fam:PS-QF']
bo_qf_properties = (
    'Abort-Cmd BSMPComm-Sel BSMPComm-Sts CtrlLoop-Sel CtrlLoop-Sts '
    'CtrlMode-Mon Current-Mon Current-RB Current-SP Current1-Mon Current2-Mon '
    'CurrentRef-Mon CycleAmpl-RB CycleAmpl-SP CycleAuxParam-RB '
    'CycleAuxParam-SP CycleEnbl-Mon CycleFreq-RB CycleFreq-SP CycleIndex-Mon '
    'CycleNrCycles-RB CycleNrCycles-SP CycleOffset-RB CycleOffset-SP '
    'CycleType-Sel CycleType-Sts IntlkHard-Mon IntlkHardLabels-Cte '
    'IntlkSoft-Mon IntlkSoftLabels-Cte OpMode-Sel OpMode-Sts '
    'PRUBlockIndex-Mon PRUCtrlQueueSize-Mon PRUSyncMode-Mon '
    'PRUSyncPulseCount-Mon Properties-Cte PwrState-Sel PwrState-Sts '
    'Reset-Cmd RmpIncNrCycles-Mon RmpIncNrCycles-RB RmpIncNrCycles-SP '
    'RmpReady-Mon Version-Cte WfmData-RB WfmData-SP WfmIndex-Mon')


def print_pvs(iocnames, properties):
    """."""
    for ioc in iocnames:
        for p in properties.split():
            print(ioc + ":" + p)


print_pvs(tb_b_iocnames, tb_b_properties)
print_pvs(tb_quad_iocnames, tb_quad_properties)
print_pvs(tb_corr_iocnames, tb_corr_properties)
print_pvs(bo_b_iocnames, bo_b_properties)
print_pvs(bo_b_iocnames, bo_b_properties)
print_pvs(bo_qd_sx_sd_iocnames, bo_qd_sx_sd_properties)
print_pvs(bo_qf_iocnames, bo_qf_properties)
