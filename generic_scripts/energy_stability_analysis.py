"""."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from apsuite.commisslib.measure_orbit_stability import OrbitAnalysis

plt.rcParams.update({
    'font.size': 18, 'lines.linewidth': 2,
    'axes.grid': True, 'grid.alpha': 0.5, 'grid.linestyle': '--',
    'text.usetex': True})


def create_data(prefix):
    """."""
    folder = ''
    # folder = prefix + '2021-10-25-SI_orbit_stability_characterization/'
    files = os.listdir(folder)
    files = [name for name in files if '.pickle' in name and 'fofb' in name]
    orb_objs = []
    for fil in files:
        obj = OrbitAnalysis(filename=folder+fil)
        orb_objs.append(obj)
        print(fil)

    # folder = prefix + '2021-11-22-orbit-stability-fancoils/'
    files = os.listdir(folder)
    files = [
        name for name in files if '.pickle' in name and
        'fofb' in name and 'fancoils' in name]
    for fil in files:
        obj = OrbitAnalysis(filename=folder+fil)
        orb_objs.append(obj)
        print(fil)
    return orb_objs


if __name__ == '__main__':
    prefix = ''
    orb_objs = create_data(prefix)

    etas, denergies = [], []
    scurrs = []
    for obj in orb_objs:
        obj.orbit_stability_analysis(central_freq=60, window=10)
        obj.energy_stability_analysis(central_freq=12*64, window=10)
        scurr = obj.data['stored_current']
        scurrs.append(scurr)

    sort_scurr = np.argsort(scurrs)
    scurrs = sorted(scurrs)
    orb_objs = [orb_objs[idx] for idx in sort_scurr]
    colors = cm.rainbow(np.linspace(0, 1, len(orb_objs)))

    title = r'Sqrt of Integrated Spectrum for energy deviation $\delta$'
    title += '\n'
    title += r'Beam response analyzed around 12 $\times$ 64 Hz = 768Hz'
    lab = f"I = {scurrs[0]:.1f}mA"
    fig, axs = obj.plot_energy_integrated_psd(label=lab, title=title)
    for idx, obj in enumerate(orb_objs[1:]):
        lab = f"I = {scurrs[idx+1]:.1f}mA"
        obj.plot_energy_integrated_psd(
            label=lab, fig=fig, axs=axs, color=colors[idx+1])
    axs.axhline(
        OrbitAnalysis.ENERGY_SPREAD*0.1,
        ls='--', label=r'10$\%$ of $\sigma_{\delta}$', color='k')
    axs.legend(
        loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 12})
    axs.set_xlim([0.1, 3])
    fig.savefig('psd_energy.png', dpi=300)
    plt.show()

    title = 'SI-01M2:DI-BPM - Orbit Integrated PSD'
    bpmidx = 0
    lab = f"I = {scurrs[0]:.1f}mA"
    fig, axs = obj.plot_orbit_integrated_psd(
        bpmidx=bpmidx, label=lab, title=title)
    for idx, obj in enumerate(orb_objs[1:]):
        lab = f"I = {scurrs[idx+1]:.1f}mA"
        obj.plot_orbit_integrated_psd(
            bpmidx=bpmidx, label=lab, fig=fig, axs=axs, color=colors[idx+1])
    axs[0].legend(
        loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})
    axs[0].set_xlim([55, 65])
    axs[1].set_xlim([55, 65])
    fig.savefig('psd_orbit.png', dpi=300)
    plt.show()

    title = 'SI-01M2:DI-BPM - Orbit Spectrum'
    bpmidx = 0
    lab = f"I = {scurrs[0]:.1f}mA"
    fig, axs = obj.plot_orbit_spectrum(
        bpmidx=bpmidx, label=lab, title=title)
    for idx, obj in enumerate(orb_objs[1:]):
        lab = f"I = {scurrs[idx+1]:.1f}mA"
        obj.plot_orbit_spectrum(
            bpmidx=bpmidx, label=lab, fig=fig, axs=axs, color=colors[idx+1])
    axs[0].legend(
        loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})
    axs[0].set_xlim([55, 65])
    axs[1].set_xlim([55, 65])
    fig.savefig('spectrum_orbit.png', dpi=300)
    plt.show()

    title = 'SI-01M2:DI-BPM - Space Modes'
    lab = f"I = {scurrs[0]:.1f}mA"
    fig, axs = obj.plot_orbit_spacial_modes(
        modes=[0, ], label=lab, title=title)
    for idx, obj in enumerate(orb_objs[1:]):
        lab = f"I = {scurrs[idx+1]:.1f}mA"
        obj.plot_orbit_spacial_modes(
            modes=[0, ], label=lab,
            fig=fig, axs=axs, color=colors[idx+1])
    axs[0].legend(
        loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 10})
    fig.savefig('space_modes_orbit.png', dpi=300)
    plt.show()
