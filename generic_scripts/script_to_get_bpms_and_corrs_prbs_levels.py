import numpy as np
import matplotlib.pyplot as mplt

from siriuspy.clientconfigdb import ConfigDBClient


def calc_svd(mat, idcs_bpm=None, idcs_corr=None):
    if idcs_bpm is None:
        idcs_bpm = np.ones(mat.shape[0], dtype=bool)
    if idcs_corr is None:
        idcs_corr = np.ones(mat.shape[1], dtype=bool)
    return np.linalg.svd(mat[idcs_bpm][:, idcs_corr], full_matrices=True)


def get_levels_corrs(mat, lvl0=-9000, lvl1=9000, singval=0, idcs_bpm=None, idcs_corr=None):
    if idcs_bpm is None:
        idcs_bpm = np.ones(mat.shape[0], dtype=bool)
    if idcs_corr is None:
        idcs_corr = np.ones(mat.shape[1], dtype=bool)

    u, s, v = calc_svd()
    vs = v[singval]
    vs /= np.abs(vs).max()
    amp = (lvl1-lvl0)/2
    off = (lvl1+lvl0)/2

    lvl0 = np.zeros(mat.shape[1])
    lvl1 = np.zeros(mat.shape[1])
    lvl0[idcs_corr] = off - amp * vs
    lvl1[idcs_corr] = off + amp * vs
    return lvl0, lvl1


def get_levels_bpms(mat, lvl0=-9000, lvl1=9000, singval=0, idcs_bpm=None, idcs_corr=None):
    if idcs_bpm is None:
        idcs_bpm = np.ones(mat.shape[0], dtype=bool)
    if idcs_corr is None:
        idcs_corr = np.ones(mat.shape[1], dtype=bool)

    u, s, v = calc_svd()
    us = u[:, singval]
    us /= np.abs(us).max()
    amp = (lvl1-lvl0)/2
    off = (lvl1+lvl0)/2

    lvl0 = np.zeros(mat.shape[0])
    lvl1 = np.zeros(mat.shape[0])
    lvl0[idcs_bpm] = off - amp * us
    lvl1[idcs_bpm] = off + amp * us
    return lvl0, lvl1


clt = ConfigDBClient(config_type='si_fastorbcorr_respm')
mat = np.array(clt.get_config_value('ref_respmat'))

idcs_corr = np.ones(mat.shape[1], dtype=bool)
idcs_corr[-1] = False
idcs_corr[0] = False
idcs_corr[79] = False
idcs_corr[0+80] = False
idcs_corr[79+80] = False

idcs_bpm = np.ones(mat.shape[0], dtype=bool)
idcs_bpm[1:160:8] = False
idcs_bpm[2:160:8] = False
idcs_bpm[5:160:8] = False
idcs_bpm[6:160:8] = False

idcs_bpm[160+1::8] = False
idcs_bpm[160+2::8] = False
idcs_bpm[160+5::8] = False
idcs_bpm[160+6::8] = False

# u, s, v = np.linalg.svd(mat, full_matrices=True)
# fig, (ax, ay) = mplt.subplots(2, 1, figsize=(12, 10)); ax.plot(v[0]); ay.plot(u[:, 0]); fig.show()
lvl0, lvl1 = get_levels_corrs(mat, lvl0=-250, lvl1=250, singval=0, idcs_bpm=idcs_bpm, idcs_corr=idcs_corr)
# fig, ax = mplt.subplots(1, 1); ax.plot(lvl0); ax.plot(lvl1); fig.show()
np.savetxt('levels_corrs_sing_val0.txt', np.array([lvl0, lvl1]).T)

lvl0, lvl1 = get_levels_bpms(mat, lvl0=-5000, lvl1=5000, singval=0, idcs_bpm=idcs_bpm, idcs_corr=idcs_corr)
# fig, ax = mplt.subplots(1, 1); ax.plot(lvl0); ax.plot(lvl1); fig.show()
np.savetxt('levels_bpms_sing_val0.txt', np.array([lvl0, lvl1]).T)
