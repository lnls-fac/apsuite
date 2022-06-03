#!/usr/bin/env python-sirius

"""."""

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import load_pickle


def get_respm_model(fname):
    data = load_pickle(fname)
    respmx = data['respmx']
    respmy = data['respmy']
    return respmx, respmy


def get_respm_meas(fname):
    data = load_pickle(fname)
    fch_names = data['fch_names']
    fcv_names = data['fcv_names']
    data.pop('fch_names', None)
    data.pop('fcv_names', None)
    return fch_names, fcv_names, data


def make_comparison(ps_names, data, respmx, respmy, plot_flag=True, filter=None):

    fmt = '{} {:+4.0f} {:+.3f} {:+06.1f} {:+.3f} {:+06.1f}'
    for idx, fc in enumerate(ps_names):
        if filter and filter not in fc:
            continue
        datum = data[fc]
        curr0, curr1 = datum['curr0'], datum['curr1']
        orbx0, orbx1 = datum['orbx0'], datum['orbx1']
        orby0, orby1 = datum['orby0'], datum['orby1']
        dcurr = curr1 - curr0
        codx_measu = orbx1 - orbx0
        cody_measu = orby1 - orby0
        codx_model = 1e6 * respmx[:160, idx]
        cody_model = 1e6 * respmy[160:, idx]

        crossx = np.dot(codx_model, codx_measu)
        alphax = crossx / np.dot(codx_model, codx_model)
        thetax = crossx / np.sqrt(np.dot(codx_model, codx_model) * np.dot(codx_measu, codx_measu))

        crossy = np.dot(cody_model, cody_measu)
        alphay = crossy / np.dot(cody_model, cody_model)
        thetay = crossy / np.sqrt(np.dot(cody_model, cody_model) * np.dot(cody_measu, cody_measu))

        # print(codx_measu.shape, codx_model.shape)
        print(fmt.format(fc, 1000*dcurr, alphax*10, 100*thetax, alphay*10, 100*thetay))

        if plot_flag:
            plt.plot(codx_measu, label='measu')
            plt.plot(alphax*codx_model, label=f'model {alphax*10:+.3f} urad')
            plt.xlabel('BPM index')
            plt.ylabel('codx [um]')
            plt.legend()
            plt.title(fc)
            plt.show()

            plt.plot(cody_measu, label='measu')
            plt.plot(alphay*cody_model, label=f'model {alphay*10:+.3f} urad')
            plt.xlabel('BPM index')
            plt.ylabel('cody [um]')
            plt.legend()
            plt.title(fc)
            plt.show()


def run():
    fname = 'fast-corr-cod-signature-model.pickle'
    respmx, respmy = get_respm_model(fname)

    fname = 'fast-corr-cod-signature-meas.pickle'
    fch_names, fcv_names, data = get_respm_meas(fname)

    filters = ['M1', 'M2', 'C2', 'C3']
    make_comparison(fch_names, data, respmx, respmy, plot_flag=False, filter=None)
    make_comparison(fcv_names, data, respmx, respmy, plot_flag=False, filter=None)


if __name__ == '__main__':
    run()