#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt

from mathphys.functions import load_pickle as _load_pickle

from lib import create_newtbt as _create_newtbt
from lib import load_pickle as _load_pickle

import scipy.signal as _scysig


def calc_beta_pca(traj, beta=1, nturns=100, full=False, fit=False):
    umat, smat, vtmat = _np.linalg.svd(traj[:, :nturns])

    beta_pca = (smat[1]*umat[:, 1])**2 + (smat[0]*umat[:,0])**2
    if fit:
        beta_pca *= _np.sum(beta*beta_pca)/_np.sum(beta_pca*beta_pca)
    else:
        beta_pca *= (_np.mean(beta)/_np.mean(beta_pca))
    if full:
        return beta_pca, umat, smat, vtmat
    else:
        return beta_pca


def calc_bpm_noise(folder, fnames, nturn_ini=0, nturn_end=None, filter_window=None):

    beta = []
    traj = []
    tbt = []
    optics = []
    for i, fname in enumerate(fnames):

        # get data and analysis
        fname_ = 'results/' + (folder + fname).replace('/','-').replace('.pickle','-optics.pickle')
        optics_ = _load_pickle(fname_)
        kicktype = 'CHROMX' if 'orizontal' in fname else 'CHROMY'
        tbt_ = _create_newtbt(folder + fname, kicktype, print_flag=True)
        nturn_end = tbt_.data_nr_turns if nturn_end is None else nturn_end
        traj_ = tbt_.data_traj[0].copy()
        tbt.append(tbt_)
        optics.append(optics_)
        # filter data ?
        if filter_window is not None:
            for bpm in range(traj_.shape[1]):
                data_anal = _np.array(_scysig.hilbert(traj_[:,bpm]))
                # calculate DFT:
                data_dft = _np.fft.fft(data_anal)
                freq = _np.fft.fftfreq(data_anal.shape[0])
                center_freq = optics_.tune
                sigma_freq =filter_window
                # Apply Gaussian filter to get only the synchrotron frequency
                H = _np.exp(-(freq - center_freq)**2/2/sigma_freq**2)
                H += _np.exp(-(freq + center_freq)**2/2/sigma_freq**2)
                H /= H.max()
                data_dft *= H
                # get the processed data by inverse DFT
                data_anal = _np.fft.ifft(data_dft)
                # phase_hil = _np.unwrap(_np.angle(data_anal))
                # instant_freq = _np.gradient(phase_hil)/(2*_np.pi)
                amp_filtered = _np.abs(data_anal)
                traj_[:, bpm] = amp_filtered
                traj.append(traj_)
        traj_filtered = traj
        beta.append(calc_beta_pca(traj_[nturn_ini:nturn_end,:].T, nturns=nturn_end-nturn_ini))
    return beta, traj_filtered, optics, tbt
    

def plot_noise(folder, fnames):

    nturn_ini = 1600
    nturn_end = 2000

    # unfiltered
    beta, traj, optics, tbt = calc_bpm_noise(folder, fnames, nturn_ini=nturn_ini, nturn_end=nturn_end, filter_window=None)
    if tbt[0].select_plane_x:
        plane = 'Horizontal'
    else:
        plane = 'Vertical'
    title = (
        'PCA analysis for {} TbT Data - Last {} turns\n' + \
        r'(Unfiltered, J units: $\mu m.rad$)').format(plane, nturn_end-nturn_ini)
    n, sx, sy, mb = len(beta)-1, 5, 1, min(beta[0]) - 0.01 * (max(beta[0]) - min(beta[0]))
    _plt.plot([0+n*sx,160+n*sx], [mb+n*sy,mb+n*sy], 'k--', alpha=0.5)
    _plt.plot([160-160,160+n*sx-160], [mb,mb+n*sy], 'k--')
    for i in range(len(beta)):
        x = _np.linspace(0, 1, 160) * 159 + sx*i
        y = beta[i] + sy*i
        J = optics[i].beta.J
        _plt.plot(x, y, label='J = {:.03f} '.format(J))
    _plt.plot([0,160], [mb,mb], 'k--')
    _plt.plot([160,160+n*sx], [mb,mb+n*sy], 'k--')
    _plt.legend()
    _plt.xlabel('BPM index')
    _plt.ylabel('a.u.')
    _plt.title(title)
    _plt.savefig(plane+'-tbt-noise-unfiltered.svg')
    _plt.show()

    # filtered
    beta, traj, optics, tbt = calc_bpm_noise(folder, fnames, nturn_ini=nturn_ini, nturn_end=nturn_end, filter_window=0.01)
    if tbt[0].select_plane_x:
        plane = 'Hozizontal'
    else:
        plane = 'Vertical'
    title = (
        'PCA analysis for {} TbT Data - Last {} turns\n' + \
        r'(w=0.01 filter around tune, J units: $\mu m.rad$)').format(plane, nturn_end-nturn_ini)
    n, sx, sy, mb = len(beta)-1, 15, 2, min(beta[0]) - 0.05 * (max(beta[0]) - min(beta[0]))
    _plt.plot([0+n*sx,160+n*sx], [mb+n*sy,mb+n*sy], 'k--', alpha=0.5)
    _plt.plot([160-160,160+n*sx-160], [mb,mb+n*sy], 'k--')
    for i in range(len(beta)):
        x = _np.linspace(0, 1, 160) * 159 + sx*i
        y = beta[i] + sy*i
        J = optics[i].beta.J
        _plt.plot(x, y, label='J = {:.03f} '.format(J))
    _plt.plot([0,160], [mb,mb], 'k--')
    _plt.plot([160,160+n*sx], [mb,mb+n*sy], 'k--')
    _plt.legend()
    _plt.xlabel('BPM index')
    _plt.ylabel('a.u.')
    _plt.title(title)
    _plt.savefig(plane+'-tbt-noise-filtered.svg')
    _plt.show()

    
def plot_pca_beta_noise():

    # horizontal
    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fnames = [
    #     'tbt_data_horizontal_m100urad_after_cycle.pickle',
    #     'tbt_data_horizontal_m150urad_after_cycle.pickle',
    #     'tbt_data_horizontal_m200urad_after_cycle.pickle',
    #     'tbt_data_horizontal_m250urad_after_cycle.pickle',
    #     ]
    # plot_noise(folder, fnames)

    # vertical
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fnames = [
        'tbt_data_vertical_100volts_after_cycle.pickle',
        'tbt_data_vertical_150volts_after_cycle.pickle',
        'tbt_data_vertical_200volts_after_cycle.pickle',
        'tbt_data_vertical_250volts_after_cycle.pickle',
        'tbt_data_vertical_300volts_after_cycle.pickle',
        'tbt_data_vertical_350volts_after_cycle.pickle',
        'tbt_data_vertical_400volts_after_cycle.pickle',
        'tbt_data_vertical_450volts_after_cycle.pickle',
        'tbt_data_vertical_500volts_after_cycle.pickle',
        'tbt_data_vertical_550volts_after_cycle.pickle',
        'tbt_data_vertical_600volts_after_cycle.pickle',
        'tbt_data_vertical_650volts_after_cycle.pickle',
        'tbt_data_vertical_700volts_after_cycle.pickle',
        ]
    plot_noise(folder, fnames)   

    

plot_pca_beta_noise()
