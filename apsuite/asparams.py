"""Nominal Accelerator Structure Parameters."""

# These are nominal accelerator structure parameters conveniently stored
# in this module. Some parameters are optics-related and should be updated
# if any lattice changes.

RF_FREQ = 499666000  # [Hz]

BO_RING_LEN = 496.78745  # [m]
BO_HARM_NR = 828
BO_REV_FREQ = RF_FREQ / BO_HARM_NR
BO_REV_TIME = BO_HARM_NR / RF_FREQ
BO_NR_BPMS = 50
BO_TUNEX = 19.20433
BO_TUNEY = 7.31417
BO_MOM_COMPACT = 7.166e-04

SI_RING_LEN = 518.3899  # [m]
SI_HARM_NR = 864
SI_REV_FREQ = RF_FREQ / SI_HARM_NR
SI_REV_TIME = SI_HARM_NR / RF_FREQ
SI_NR_BPMS = 160
SI_TUNEX = 49.09619
SI_TUNEY = 14.15194
SI_MOM_COMPACT = 1.636e-04
SI_ENERGY_SPREAD = 0.085  # [%]

BBBL_DAC_NBITS = 14
BBBL_SAT_THRES = 2**(BBBL_DAC_NBITS-1) - 1
BBBL_CALIBRATION_FACTOR = 1000  # [Counts/mA/degree]
BBBL_DAMPING_RATE = 1/13.0  # [Hz]

BPM_SWITCHING_FREQ = 12.5e3  # Hz
FOFB_DOWNSAMPLING = 23
MONIT1_DOWNSAMPLING = 25*FOFB_DOWNSAMPLING
