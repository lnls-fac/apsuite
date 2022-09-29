"""Model of the vacuum chamber effect on oscillating field.

Script that calculates the effect of the vacuum chamber on the performance of
the feedforward system.

The modelling of the transfer function of the vacuum chamber is based on the
paper:

Podobedov, B., Ecker, L., Harder D., Rakowsky G.,
Eddy Current Shielding By Electrically Thick Vacuum Chambers
Proceedings of PAC09, Vancouver, BC, Canada
code: TH5PFP083, pp 3398-3400. Available at:
https://accelconf.web.cern.ch/pac2009/papers/th5pfp083.pdf

"""

import numpy as np
import matplotlib.pyplot as mplt

mplt.rcParams.update({
    'grid.alpha': 0.5, 'grid.linestyle': '--', 'axes.grid': True,
    'lines.linewidth': 2, 'font.size': 14})


class CalcFeedForwardPerf:
    """."""

    # Copper Conductivity
    CONDUCTIVITY = 59e6  # [S/m] == [1/Ohm/m]

    def __init__(
            self, thick=3, radius=7.9, velocity=10, amp_phase=26):
        """."""
        # chamber properties
        self.chamb_thick = thick  # [mm]
        self.chamb_radius = radius  # [mm]

        # transfer function calculation parameters:
        self.transf_error_tolerance = 1e-5

        # Phase ramp parameters
        self.velocity = velocity  # [mm/s]
        self.amp_phase = amp_phase  # [mm]

        self.ff_amp = 100  # [G.cm]

    def feedforward_table(self, phase):
        """."""
        norm_phase = phase / self.amp_phase
        # consider the field applied by the corrector must go
        # from 0 to 100 G.cm quadractically when the phase goes
        # from 0 to the maximum value:
        # return norm_phase**2 * amp  # [G.cm]

        # Consider a Gaussian curve for continuity:
        # vary sigma to get larger derivatives:
        sig = 0.1
        mean = 0.5
        return self.ff_amp * np.exp(-(norm_phase - mean)**2/2/sig**2)

    def calc_vchamb_transfer_function(
            self, freqs=None, delay=0, maxpoles=100, plot=False):
        """."""
        if plot:
            fig, (ax1, ax2) = mplt.subplots(2, 1, sharex=True, figsize=(9, 6))
            ax1.set_ylabel('Magnitude')
            ax2.set_ylabel('Phase [°]')
            ax2.set_xlabel('Frequency [Hz]')
            ax1.set_xscale('log')
        mu0 = 4e-7*np.pi  # [H/m]

        if freqs is None:
            freqs = np.linspace(0, 500, 1000)

        s = 2j * np.pi * freqs
        tau = 0.5*mu0*self.CONDUCTIVITY
        tau *= self.chamb_radius*self.chamb_thick / 1e6  # [mm] -> [m]
        pole0 = 1/tau

        transf = np.ones(freqs.shape, dtype=complex)
        for n in range(maxpoles+1):
            if n == 0:
                polen = pole0
            else:
                polen = 0.5 * pole0 * n*n * np.pi*np.pi
                polen *= self.chamb_radius/self.chamb_thick
            tr_old = transf.copy()
            transf *= polen / (polen + s)
            if plot:
                phase = np.unwrap(np.angle(transf))/np.pi*180
                ax1.plot(freqs, np.abs(transf))
                ax2.plot(freqs, phase)
            res = np.abs(transf - tr_old)
            if res.max() < self.transf_error_tolerance:
                print(res.max())
                break

        if plot:
            ax1.set_title(
                f'thickness = {self.chamb_thick:.1f} mm,  '
                f'radius = {self.chamb_radius:.1f} mm,  '
                f'n° poles = {n:d}')
            fig.tight_layout()

        transf *= np.exp(-s*delay)
        return transf, n

    # ######################################################################
    def test_ramp(self, delay=0, fullres=False):
        """."""
        phase_ramp = np.linspace(-2, 2, 10000)
        df = phase_ramp[1]-phase_ramp[0]
        df *= self.amp_phase

        phase_ramp = 1.5 - np.abs(phase_ramp)
        phase_ramp[phase_ramp < 0] = 0
        phase_ramp[phase_ramp > 1] = 1
        phase_ramp *= self.amp_phase

        time = np.arange(0, phase_ramp.size) * df / self.velocity
        dt = time[1] - time[0]

        # Model for the Feedforward table:
        ff_stren = self.feedforward_table(phase_ramp)

        dft_ff_stren = np.fft.rfft(ff_stren)
        freqs = np.fft.rfftfreq(ff_stren.size, d=dt)
        chamb_transf, npoles = self.calc_vchamb_transfer_function(
            freqs, delay=delay)

        dft_resp = chamb_transf*dft_ff_stren
        resp = np.fft.irfft(dft_resp)

        # Feedfoward performance:
        fig, ax1 = mplt.subplots(1, 1, figsize=(9, 6))
        ax2 = ax1.twinx()
        ax2.spines['right'].set_color('tab:red')
        ax2.tick_params(axis='y', colors='tab:red')
        ax2.yaxis.label.set_color('tab:red')
        ax1.set_ylabel('Integrated Field [G.cm]')
        ax2.set_ylabel('Diff [G.cm]')
        ax1.set_xlabel('Time [s]')
        ax1.set_title(
            f'Phase Ramp: 0 --> {self.amp_phase:.1f} mm '
            f'@ {self.velocity:.1f}mm/s \n'
            f'Chamber: thickness = {self.chamb_thick:.1f} mm,  '
            f'radius = {self.chamb_radius:.1f} mm,  '
            f'n° poles = {npoles:d}\n'
            f'Control System: Delay = {delay*1e6:.0f} us')

        ax1.plot(time, ff_stren, label="'Ideal' field")
        ax1.plot(time, resp, label='Filtered field')
        ax2.plot(time, resp - ff_stren, color='tab:red')
        ax1.legend(loc='best')
        fig.tight_layout()

        if not fullres:
            return

        # Phase Ramp:
        fig, ax1 = mplt.subplots(1, 1, figsize=(9, 6))
        ax1.set_ylabel('Phase [mm]')
        ax1.set_xlabel('Time [s]')
        ax1.set_title(
            f'Phase Ramp: 0 --> {self.amp_phase:.1f} mm '
            f'@ {self.velocity:.1f}mm/s')
        ax1.plot(time, phase_ramp)
        fig.tight_layout()

        # Feedforward table:
        fig, ax1 = mplt.subplots(1, 1, figsize=(9, 6))
        ax1.set_ylabel('Corr Strength [G.cm]')
        ax1.set_xlabel('Phase [mm]')
        ax1.set_title(f'BL[G.cm] = 100 * exp(-(phase[mm]/26-0.5)**2/2/0.1**2')
        ax1.plot(phase_ramp, ff_stren)
        fig.tight_layout()

        # fig, ax = mplt.subplots(1, 1, figsize=(5, 9))
        # ax.plot(freqs, np.abs(dft_ff_stren))
        # ax.plot(freqs, np.abs(dft_resp))
        # ax.set_yscale('log')
        # fig.tight_layout()


if __name__ == '__main__':
    calc = CalcFeedForwardPerf(velocity=10, amp_phase=52/2)
    calc.transf_error_tolerance = 1e-4

    # Thickness of the chamber:
    # calc.chamb_thick = 3  # [mm]
    calc.chamb_thick = 3  # [mm]

    # Assume the inner radius of the chamber at the corrector position
    # is the average radius of the transition:
    und_half_hgap = 13.6 / 2
    und_half_vgap = 7.6 / 2
    nom_chamb_radius = 12
    calc.chamb_radius = (nom_chamb_radius + und_half_vgap) / 2  # [mm]
    # calc.chamb_radius = und_half_vgap  # [mm]

    # calc.calc_vchamb_transfer_function(plot=True, tol=1e-3)

    calc.test_ramp(delay=0, fullres=False)
    # calc.test_ramp(delay=150e-6, fullres=True)
    calc.test_ramp(delay=280e-6, fullres=False)
    # calc.test_ramp(delay=-1.1e-3)
    mplt.show()
