"""Nominal Accelerator Structure Parameters."""

# These are nominal accelerator structure parameters conveniently stored
# in this module. Some parameters are optics-related and should be updated
# with lattice changes.

import mathphys as _mp


# --- BEAM ---

_BEAM_ENERGY_HIGH = 3.0  # [GeV]
_BEAM_ENERGY_LOW = 0.150  # [GeV]
_BEAMPARAMS_HIGH = _mp.beam_optics.beam_rigidity(energy=_BEAM_ENERGY_HIGH)
_BEAMPARAMS_LOW = _mp.beam_optics.beam_rigidity(energy=_BEAM_ENERGY_LOW)

SI_BEAM_ENERGY = _BEAM_ENERGY_HIGH  # [GeV]
SI_BEAM_BRHO = _BEAMPARAMS_HIGH[0]  # [T.m]
SI_BEAM_SPEED = _BEAMPARAMS_HIGH[1]  # [m/s]
SI_BEAM_BETA = _BEAMPARAMS_HIGH[2]
SI_BEAM_GAMMA = _BEAMPARAMS_HIGH[3]

BO_BEAM_ENERGY_HIGH = _BEAM_ENERGY_HIGH  # [GeV]
BO_BEAM_BRHO_HIGH = _BEAMPARAMS_HIGH[0]  # [T.m]
BO_BEAM_SPEED_HIGH = _BEAMPARAMS_HIGH[1]  # [m/s]
BO_BEAM_BETA_HIGH = _BEAMPARAMS_HIGH[2]
BO_BEAM_GAMMA_HIGH = _BEAMPARAMS_HIGH[3]

BO_BEAM_ENERGY_LOW = _BEAM_ENERGY_LOW  # [GeV]
BO_BEAM_BRHO_LOW = _BEAMPARAMS_LOW[0]  # [T.m]
BO_BEAM_SPEED_LOW = _BEAMPARAMS_LOW[1]  # [m/s]
BO_BEAM_BETA_LOW = _BEAMPARAMS_LOW[2]
BO_BEAM_GAMMA_LOW = _BEAMPARAMS_LOW[3]

del(_BEAM_ENERGY_HIGH, _BEAM_ENERGY_LOW)
del(_BEAMPARAMS_HIGH, _BEAMPARAMS_LOW)

# --- RF GENERAL ---

RF_FREQ = 499663824.380981 # [Hz] -- compatible with SI nominal model length
# RF_FREQ = 499666862 - 150  # [Hz] -- compatible with BO nominal model @ 3 Gev
# RF_FREQ = 499666862        # [Hz] -- in use 2022-04-10

# --- BPMS ---

BPM_SWITCHING_FREQ = 12.5e3  # [Hz] NOTE: what is the exact expression ?
BPM_FOFB_DOWNSAMPLING = 23
BPM_MONIT1_DOWNSAMPLING = 25*BPM_FOFB_DOWNSAMPLING

# --- SI LATTICE ---

SI_HARM_NR = 864
SI_REV_FREQ = RF_FREQ / SI_HARM_NR
SI_REV_TIME = SI_HARM_NR / RF_FREQ
# SI_RING_LEN = 518.3899  # [m]  model value
SI_RING_LEN = SI_BEAM_SPEED * SI_REV_TIME
SI_NR_BPMS = 160
SI_TUNEX = 49.09619  # NOTE: this is the nominal model value!
SI_TUNEY = 14.15194  # NOTE: this is the nominal model value!
SI_MOM_COMPACT = 1.636e-04
SI_ENERGY_SPREAD = 0.085  # [%]

# --- BO LATTICE ---

BO_HARM_NR = 828
BO_REV_FREQ = RF_FREQ / BO_HARM_NR
BO_REV_TIME = BO_HARM_NR / RF_FREQ
# BO_RING_LEN = 496.78745  # [m] model value
BO_RING_LEN = BO_BEAM_SPEED_HIGH * BO_REV_TIME  # NOTE: high energy match!
BO_NR_BPMS = 50
BO_TUNEX = 19.20433  # NOTE: this is the nominal model value!
BO_TUNEY = 7.31417  # NOTE: this is the nominal model value!
BO_MOM_COMPACT = 7.166e-04

# --- BBBs ---

BBBL_DAC_NR_BITS = 14
BBBL_SAT_THRES = 2**(BBBL_DAC_NR_BITS-1) - 1
BBBL_CALIBRATION_FACTOR = 1000  # [Counts/mA/degree]
BBBL_DAMPING_RATE = 1/13.0  # [Hz]

BBBH_CALIBRATION_FACTOR = 1000  # [Counts/mA/um]
BBBH_DAMPING_RATE = 1/16.9e-3  # [Hz]

BBBV_CALIBRATION_FACTOR = 1000  # [Counts/mA/um]
BBBV_DAMPING_RATE = 1/22.0e-3  # [Hz]
