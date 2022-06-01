"""."""

import numpy as np

import pyaccel
from pymodels import si


def create_model():
    """Return SI model and family data."""
    # create model and set global parms for 6D tracking
    model = si.create_accelerator()
    model.cavity_on = True
    model.radiation_on = True
    model.vchamber_on = True
    # get family data
    famdata = si.get_family_data(model)
    fch_names = famdata['FCH']['devnames']
    fcv_names = famdata['FCV']['devnames']
    bpm_names = famdata['BPM']['devnames']
    fch_idx = famdata['FCH']['index']
    fcv_idx = famdata['FCV']['index']
    bpm_idx = famdata['BPM']['index']
    # bpm_idx = pa.lattice.find_indices(si, 'fam_name', 'BPM')
    return model, bpm_names, fch_names, fcv_names, fch_idx, fcv_idx, bpm_idx



def calc_orbit_respm(model, bpm_idx, corr_idx):
    """."""
    nrows = len(bpm_idx)
    ncolumns = len(corr_idx)
    respm = np.zeros([2*nrows, ncolumns])
    kick = 10e-6
    for j, fidx in enumerate(corr_idx):
        model[fidx[0]].hkick_polynom = +kick/2
        orbit1 = pyaccel.tracking.find_orbit6(accelerator=si, indices=bpm_idx)
        model[fidx[0]].hkick_polynom = -kick/2
        orbit2 = pyaccel.tracking.find_orbit6(accelerator=si, indices=bpm_idx)
        dorbitx = orbit1[0] - orbit2[0]
        respm[:nrows, j] = dorbitx
        dorbity = orbit1[2] - orbit2[2]
        respm[nrows:, j] = dorbity
        model[fidx[0]].hkick_polynom = 0

if __name__ == "__main__":
    print('ok')
