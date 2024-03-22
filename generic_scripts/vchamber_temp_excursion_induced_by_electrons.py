#!/usr/bin/env python3
"""Temperature excursion induced by electron beam.

This script calculates the temperature excursions of SIRIUS vacuum chambers
induced by the collision with the eletron beam.

The dose generated is calculated based on equations from [1] and the
temperature excursion is calculated based on [2].

The materials properties are taken from [3].

The beam equilibrium parameters for the storage ring model with vertical
dispersion and coupling fitted to measurements are used.

[1] Jeffrey Dooling, private communication.
[2] equation 2.4 of APS-U Final Design Report. Available at
    https://publications.anl.gov/anlpubs/2019/07/153666.pdf
[3] table 2.26 of APS-U Final Design Report. Available at
    https://publications.anl.gov/anlpubs/2019/07/153666.pdf

"""

import matplotlib.pyplot as mplt
import numpy as np
import pyaccel as pa
from pymodels import si


class CalcTempExcursion:
    """."""

    _DOSES = {  # doses from APS-U report
        'Graphite': 2.115,
        'Aluminum': 2.001,
        'Titaniun': 1.881,
        'Copper': 1.901,
        'Tungsten': 2.338,
    }
    # SPC in  [MeV*cm^2/g]
    _spc_al = 2.10
    _dose_al = 2.001
    SPC = {}
    for k, v in _DOSES.items():
        SPC[k] = v / _dose_al * _spc_al

    SPEC_HEAT = {  # [J/mole/K]
        'Graphite': 8.52,
        'Aluminum': 24.2,
        'Titaniun': 25.06,
        'Copper': 24.44,
        'Tungsten': 24.27,
    }
    ATOMIC_WEIGHT = {  # [g/mole]
        'Graphite': 12.01,
        'Aluminum': 26.98,
        'Titaniun': 47.87,  # for Ti Only
        'Copper': 63.55,
        'Tungsten': 183.84,
    }
    FUSION_TEMP = {  # [K]
        'Graphite': 3915,  # sublimation
        'Aluminum': 933,
        'Titaniun': 1900,  # 1950
        'Copper': 1358,
        'Tungsten': 3695,
    }

    MATERIALS = ('Aluminum', 'Titaniun', 'Copper', 'Graphite', 'Tungsten')

    def __init__(self, model, material='Al', current=0.350):
        """Create object.

        Args:
            model (pyaccel.accelerator.Accelerator): model of the ring.
            material (str, optional): material of the vacuum chamber.
                Defaults to 'Al'.
            current (float, optional): Current of the electron beam in [A].
                Defaults to 0.350.

        """
        self.current = current
        self.model = model
        self._material = self.MATERIALS[0]
        self.material = material

        self.eqpar = None
        self.energy_density = None
        self.peak_dose = None
        self.temp_excursion = None
        self.update()

    @property
    def material(self):
        """Material of the vacuum chamber.

        Returns:
            str: material.

        """
        return self._material

    @material.setter
    def material(self, val):
        if isinstance(val, (int, np.int_)) and 0 <= val < len(self.MATERIALS):
            self._material = self.MATERIALS[val]
        elif isinstance(val, str):
            idc = [m for m in self.MATERIALS if val.lower() in m.lower()]
            if not idc:
                raise ValueError('Wrong value for material')
            self._material = idc[0]
        else:
            raise ValueError('Wrong value for material')

    @property
    def melting_temp(self):
        """Melting temperature of the vacuum chamber material in [K].

        Returns:
            float: melting temperature.
        """
        return self.FUSION_TEMP[self._material]

    def update(self):
        """Update relevant quantities."""
        mod = self.model
        harm = mod.harmonic_number
        energy = mod.energy  # [eV]
        f_rf = pa.optics.get_rf_frequency(mod)
        f_rev = f_rf / harm  # [Hz]
        charge = self.current / f_rev  # [C]

        self.eqpar = pa.optics.EqParamsFromBeamEnvelope(mod)
        self.spos = np.array(pa.lattice.find_spos(mod, indices='closed'))
        sigx = self.eqpar.sigma_rx
        sigy = self.eqpar.sigma_ry

        # energy density in [J/m^2]
        self.energy_density = charge * energy / 2 / np.pi / sigx / sigy

        # from [MeV*cm^2/g] to [eV*m^2/kg]
        spc = self.SPC[self._material] * 1e6 * 1e-4 / 1e-3

        # Dose in [Gy] or [J/kg]
        self.peak_dose = self.energy_density / energy * spc

        # atomic weight in [kg/mole]
        at_weig = self.ATOMIC_WEIGHT[self._material] * 1e-3
        spc_heat = self.SPEC_HEAT[self._material]

        # temperature excursion in [K]
        self.temp_excursion = self.peak_dose * at_weig / spc_heat


if __name__ == "__main__":
    mod = si.create_accelerator()
    mod = si.fitted_models.vertical_dispersion_and_coupling(mod)

    tmpexc = CalcTempExcursion(mod, material='Copper', current=1e-3)

    temp_env = 273 + 24
    temp = tmpexc.temp_excursion + temp_env
    fig, ax = mplt.subplots(1, 1, figsize=(6, 2.6))
    ax.plot(tmpexc.spos, temp, label='Max. Temp.')
    ax.axhline(tmpexc.melting_temp, ls='--', color='k', label='Melting Temp.')
    pa.graphics.draw_lattice(
        mod, symmetry=5, offset=0, height=temp.max()/10, gca=ax)
    ax.set_xlim(0, tmpexc.spos[-1]/5)
    ax.set_ylim(-temp.max()/9, None)
    ax.set_title(f'{tmpexc.material:s} with {tmpexc.current*1e3:.2f} mA')
    ax.grid(True, alpha=0.5, ls='--', lw=1)
    ax.set_ylabel('Temperature [K]')
    ax.set_xlabel('Position [m]')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.show()

    melt_temp = tmpexc.melting_temp
    min_curr = (melt_temp + temp_env) / tmpexc.temp_excursion
    min_curr *= 1e3*tmpexc.current
    fig, ax = mplt.subplots(1, 1, figsize=(6, 2.6))
    ax.plot(tmpexc.spos, min_curr, label='Current')
    pa.graphics.draw_lattice(
        mod, symmetry=5, offset=0, height=min_curr.max()/10, gca=ax)
    ax.set_xlim(0, tmpexc.spos[-1]/5)
    ax.set_ylim(-min_curr.max()/9, None)
    ax.set_title(
        f'Min. Current to Reach Melting Point for {tmpexc.material:s}')
    ax.grid(True, alpha=0.5, ls='--', lw=1)
    ax.set_ylabel('Current [mA]')
    ax.set_xlabel('Position [m]')
    fig.tight_layout()
    fig.show()
