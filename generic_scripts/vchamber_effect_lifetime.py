"""."""

import numpy as np
import matplotlib.pyplot as plt
import pyaccel
from pymodels import si

plt.rcParams.update({
    'grid.alpha': 0.5, 'grid.linestyle': '--', 'axes.grid': True,
    'lines.linewidth': 2, 'font.size': 14})


class VchamberLifetime:
    """."""

    def __init__(self):
        """."""
        self.chamb_len = 2.6  # [m]
        self.bscx = 3.78e-3  # [m]
        self.bscy = 2.49e-3  # [m]
        self.dlt22_chamb_side = 6.5e-3
        self.dlt22_chamb = self.dlt22_chamb_side/np.sqrt(2)  # [m]
        self.coupling = 1.00/100
        self.avg_pressure = 3e-10  # [mbar]
        self.total_curr = 100  # [mA]
        self.rf_voltage = 3.0e6  # [V]
        self.model = None

    def create_model(self):
        """."""
        mod = si.create_accelerator(ids_vchamber=False)
        mod.cavity_on = True
        mod.radiation_on = True
        cavidx = pyaccel.lattice.find_indices(mod, 'fam_name', 'SRFCav')[0]
        mod[cavidx].voltage = self.rf_voltage

        mib = pyaccel.lattice.find_indices(mod, 'fam_name', 'mib')
        spos = pyaccel.lattice.find_spos(mod)

        mod = pyaccel.lattice.insert_marker_at_position(
            mod, 'end_delta22', spos[mib[0]] + self.chamb_len/2)
        mod = pyaccel.lattice.insert_marker_at_position(
            mod, 'end_delta22', spos[mib[0]] - self.chamb_len/2)
        self.model = mod

    def calc_bsc(self):
        """."""
        dlt22 = pyaccel.lattice.find_indices(
            self.model, 'fam_name', 'end_delta22')
        accepx, accepy, twiss = pyaccel.optics.calc_transverse_acceptance(
            self.model)
        accepx = np.min(accepx)
        accepy = np.min(accepy)
        bscx = np.sqrt(twiss.betax*accepx)
        bscy = np.sqrt(twiss.betay*accepy)
        return bscx[dlt22[0]], bscy[dlt22[0]]

    def calc_lifetime(self, mod=None):
        """."""
        modc = mod or self.model
        lifetime = pyaccel.lifetime.Lifetime(modc)
        lifetime.avg_pressure = self.avg_pressure
        emit0 = lifetime.emit1
        lifetime.curr_per_bunch = self.total_curr/864
        kappa = self.coupling
        lifetime.emit1 = emit0/(1+kappa)
        lifetime.emit2 = emit0*kappa/(1+kappa)
        lifetime.touschek_model = lifetime.TOUSCHEKMODEL.Piwinski
        rf_accep = lifetime.equi_params.rf_acceptance

        accep = pyaccel.optics.calc_touschek_energy_acceptance(modc)
        accep_neg, accep_pos = accep[0], accep[1]
        accep_neg = np.maximum(accep_neg, -rf_accep)
        accep_pos = np.minimum(accep_pos, +rf_accep)
        accep = np.array([accep_neg, accep_pos])
        lifetime.accelerator = modc
        lifetime.accepen = accep
        touschek = lifetime.lifetime_touschek/3600
        total = lifetime.lifetime_total/3600
        return touschek, total

    def change_horizontal_limitation(self, limitations):
        """."""
        dlt22idx = pyaccel.lattice.find_indices(
            self.model, 'fam_name', 'end_delta22')
        lifetimes_touschek = []
        lifetimes_total = []
        for side in limitations:
            print(f'H. limitation: {side*1e3:.2f} mm')
            for idx in dlt22idx:
                elem = self.model[idx]
                elem.hmin, elem.hmax = [-side, side]
                elem.vmin, elem.vmax = [-self.bscy, self.bscy]
            touschek, total = self.calc_lifetime()
            lifetimes_touschek.append(touschek)
            lifetimes_total.append(total)
        return lifetimes_touschek, lifetimes_total

    def plot_lifetime(self, hlimitation, lifetime, fname=None):
        """."""
        plt.figure()
        plt.plot(hlimitation*1e3, lifetime, 'o-', color='gray')

        plt.xlabel(r'horizontal limitation @ SB [mm]')
        plt.ylabel('total lifetime [h]')

        plt.vlines(
            x=self.bscx*1e3, ymin=min(lifetime), ymax=max(lifetime)*1.05,
            ls='--', lw=2, label='BSCx', color='tab:red')
        stg = f'Delta22 Chamber: ({self.dlt22_chamb_side*1e3:.1f}/'
        stg += r'$\sqrt{2}$)mm'
        plt.vlines(
            x=self.dlt22_chamb*1e3,
            ymin=min(lifetime), ymax=max(lifetime)*1.05,
            ls='--', lw=2, label=stg, color='tab:blue')

        plt.grid(True, alpha=0.5, ls='--')
        plt.legend(fontsize=9, loc='upper left')

        title = f'{self.total_curr}mA, {self.coupling*100}%, '
        title += f'{self.avg_pressure*1e10}'
        title += r'$\times10^{-10}$mbar, '
        title += r'$V_{RF}=$'
        title += f'{self.rf_voltage*1e-6:.1f}MV'
        plt.title(title)
        plt.tight_layout()
        if fname:
            plt.savefig(fname, dpi=300)
        plt.show()


if __name__ == '__main__':
    vchamb = VchamberLifetime()
    vchamb.create_model()
    vchamb.chamb_len = 2.6  # [m]
    vchamb.coupling = 1.00/100
    vchamb.avg_pressure = 3e-10  # [mbar]
    vchamb.total_curr = 100  # [mA]
    vchamb.rf_voltage = 3.0e6  # [V]
    _, bscy = vchamb.calc_bsc()
    vchamb.bscy = bscy
    # Reduced horizontal aperture to not compromise vertical BSC:
    vchamb.bscx = np.sqrt(vchamb.dlt22_chamb**2 - bscy**2)

    hlim = np.linspace(2e-3, vchamb.dlt22_chamb, 21)
    _, ltimes_total = vchamb.change_horizontal_limitation(limitations=hlim)
    vchamb.plot_lifetime(hlim, ltimes_total)
