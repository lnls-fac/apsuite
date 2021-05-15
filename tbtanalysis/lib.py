import numpy as _np
import scipy.optimize as _opt

from mathphys.functions import load_pickle
from apsuite.tbt_analysis import TbTAnalysis
from apsuite.tbt_analysis import TbT

def multibunch_kick_spread(kick):
    """."""
    rev = 1.7e-6
    kickx_width = 2 * rev
    kicky_width = 3 * rev
    bunch_half_duration = 50e-9 / 2
    time = bunch_half_duration
    percentx = 100*(_np.cos((_np.pi/2)*time/(kickx_width/2)) - 1)
    percenty = 100*(_np.cos((_np.pi/2)*time/(kicky_width/2)) - 1)
    print('kickx spread: {} %'.format(percentx))
    print('kicky spread: {} %'.format(percenty))
    print('kickx spread: {} urad'.format(kick * percentx / 100))
    print('kicky spread: {} urad'.format(kick * percenty / 100))

def convert_data(fname, correct_chrom=True):
    data = load_pickle(fname)
    if 'trajx' in data:
        if isinstance(data['chromx_err'], (list, tuple)):
            data['chromx_err'] = max(data['chromx_err'])
        if isinstance(data['chromy_err'], (list, tuple)):
            data['chromy_err'] = max(data['chromy_err'])
        data['kicktype'] = 'CHROMX' if data['kicktype'] == 'X' else data['kicktype']
        data['kicktype'] = 'CHROMY' if data['kicktype'] == 'Y' else data['kicktype']
        ndata = data
    else:
        ndata = dict()
        ndata['trajx'] = data['sofb_tbt']['x'].reshape((1, -1, 160))
        ndata['trajy'] = data['sofb_tbt']['y'].reshape((1, -1, 160))
        ndata['trajsum'] = data['sofb_tbt']['sum'].reshape((1, -1, 160))
        ndata['kicks'] = [data['kick']]
        ndata['tunex'] = data['tune']['x']
        ndata['tuney'] = data['tune']['y']
    if correct_chrom:
        # discutir!
        tbt = TbTAnalysis
        if 'chromx' in ndata:
            ndata['chromx'] = ndata['chromx'] - tbt.NOM_HARMONIC_NR * tbt.NOM_ALPHA
            ndata['chromy'] = ndata['chromy'] - tbt.NOM_HARMONIC_NR * tbt.NOM_ALPHA
    return ndata


def calc_stats(data, cutoff=3):
    data_avg = _np.mean(data)
    data_std = _np.std(data)
    insiders = _np.abs(data - data_avg) <= 3 * data_std
    data_avg = _np.mean(data[insiders])
    data_std = _np.std(data[insiders])
    inds = _np.arange(len(data))
    outliers = set(inds) - set(inds[insiders])
    return data_avg, data_std, list(outliers), list(insiders)


def calc_param_stats(param, cutoff):

    param = _np.array(param)
    stdval = _np.std(param)
    meanval = _np.median(param)
    filtered = (abs(param - meanval) <= cutoff*stdval)
    filtered_out = (abs(param - meanval) > cutoff*stdval)
    param_mean = _np.mean(param[filtered])
    param_std = _np.std(param[filtered])

    return filtered, filtered_out, param_mean, param_std


def create_tbt(fname, kicktype=None, correct_chrom=True):
    newdata = convert_data(fname, correct_chrom)
    tbt = TbTAnalysis(data_fname=fname, data=newdata, kicktype=kicktype)
    print(fname)
    print('meas. chromx : {:+.4f} ± {:.4f}'.format(tbt.chromx, tbt.chromx_err))
    print('meas. chromy : {:+.4f} ± {:.4f}'.format(tbt.chromy, tbt.chromy_err))
    if 'tunex' in tbt.data:
        print('meas. tunex  : {:+.6f} ± {:.6f}'.format(tbt.data['tunex'], 0.0))
    if 'tuney' in tbt.data:
        print('meas. tuney  : {:+.6f} ± {:.6f}'.format(tbt.data['tuney'], 0.0))
    if 'tunex_excitation_sts' in tbt.data:
        print('exc. tunex   : {}'.format(tbt.data['tunex_excitation_sts']))
    if 'tuney_excitation_sts' in tbt.data:
        print('exc. tuney   : {}'.format(tbt.data['tuney_excitation_sts']))

    print()
    return tbt
    

def create_newtbt(fname, kicktype=None, correct_chrom=True):
    newdata = convert_data(fname, correct_chrom)
    tbt = TbT(data_fname=fname, data=newdata, kicktype=kicktype)
    print(fname)
    print('meas. chrom  : {:+.4f} ± {:.4f}'.format(tbt.chrom, tbt.chrom_err))
    if 'tunex' in tbt.data:
        print('meas. tunex  : {:+.6f} ± {:.6f}'.format(tbt.data['tunex'], 0.0))
    if 'tuney' in tbt.data:
        print('meas. tuney  : {:+.6f} ± {:.6f}'.format(tbt.data['tuney'], 0.0))
    if 'tunex_excitation_sts' in tbt.data:
        print('exc. tunex   : {}'.format(tbt.data['tunex_excitation_sts']))
    if 'tuney_excitation_sts' in tbt.data:
        print('exc. tuney   : {}'.format(tbt.data['tuney_excitation_sts']))

    print()
    return tbt


def fit_leastsqr(tbt, args, params, calc_residue_vector):

    fit_data = _opt.least_squares(
        fun=calc_residue_vector,
        x0=params,
        args=args,
        method='lm')
    params_fit = fit_data['x']
    params_fit_err = fit_leastsqr_error(fit_data)
    return params_fit, params_fit_err


def fit_leastsqr_error(fit_data):
        """."""
        # based on fitting error calculation of scipy.optimization.curve_fit
        # do Moore-Penrose inverse discarding zero singular values.
        _, smat, vhmat = _np.linalg.svd(
            fit_data['jac'], full_matrices=False)
        thre = _np.finfo(float).eps * max(fit_data['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[:smat.size]
        pcov = _np.dot(vhmat.T / (smat*smat), vhmat)

        # multiply covariance matrix by residue 2-norm
        ysize = len(fit_data['fun'])
        cost = 2 * fit_data['cost']  # res.cost is half sum of squares!
        popt = fit_data['x']
        if ysize > popt.size:
            # normalized by degrees of freedom
            s_sq = cost / (ysize - popt.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(_np.nan)
            print('# of fitting parameters larger than # of data points!')
        return _np.sqrt(_np.diag(pcov))



class Analysis:
    pass
