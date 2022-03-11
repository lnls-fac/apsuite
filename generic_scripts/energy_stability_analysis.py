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
    # folder = prefix + '2021-10-25-SI_orbit_stability_characterization/'
    folder = ''
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
        eta, den = obj.energy_stability_analysis(central_freq=12*64, window=10)
        scurr = obj.data['stored_current']
        etas.append(etas)
        denergies.append(den)
        scurrs.append(scurr)

    sort_scurr = np.argsort(scurrs)
    scurrs = sorted(scurrs)
    etas = [etas[idx] for idx in sort_scurr]
    denergies = [denergies[idx] for idx in sort_scurr]

    plt.figure(figsize=(12, 6))
    color = cm.rainbow(np.linspace(0, 1, len(denergies)))
    for idx, den in enumerate(denergies):
        obj = orb_objs[idx]
        den_spec, freq = obj.calc_spectrum(den, fs=obj.sampling_freq)
        intpsd = obj.calc_integrated_spectrum(den_spec, inverse=True)
        freq = freq/1e3
        label = f"I = {scurrs[idx]:.1f}mA"
        plt.plot(freq, intpsd*100, label=label, color=color[idx])

    espread = 0.085*0.1
    plt.axhline(
        espread, color='k', ls='--', label=r'10$\%$ of $\sigma_\delta$')

    plt.xlabel('Frequency [kHz]')
    plt.ylabel(r'Sqrt of Int. Spec. [\%]')
    plt.xlim([0.1, 3])
    plt.legend(
        loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})

    title = r'Sqrt of Integrated Spectrum for energy deviation $\delta$'
    title += '\n'
    title += r'Beam response analyzed around 12 $\times$ 64 Hz = 768Hz'
    plt.title(title)
    plt.grid(False)
    plt.tight_layout(True)
    plt.show()
