"""Orbit bumps at the storage ring (SI) sextupoles."""

import numpy as _np
import matplotlib.pyplot as _plt

import pyaccel as _pa
from pymodels import si as _si
from apsuite.orbcorr import OrbRespmat as _OrbRespmat
from apsuite.optics_analysis import TuneCorr


class SextupoleBumps:
    """Compute CH/CV kicks that produce orbit bumps at sextupoles.

    The orbit response matrices are computed numerically (tracking) with
    _OrbRespmat, using the sextupoles as observation points (bpm_idx).
    The full matrices (x and y rows, CH and CV columns) are kept, so x-y
    coupling is taken into account. The coupled problem is solved with a
    truncated least squares:

        [ M_sext          ]       [ target_bump ]
        [ w_bpm * M_bpm   ] k  =  [      0      ]

    where rows are ordered as [x; y] and k = [CH kicks; CV kicks].

    Each index returned by find_indices is treated as one magnet.
    Units: bumps in [um], kicks in [urad] (M in m/rad == um/urad).
    """

    SEXT_FAMS = (
        'SFA0',
        'SFB0',
        'SFP0',
        'SFA1',
        'SFB1',
        'SFP1',
        'SFA2',
        'SFB2',
        'SFP2',
        'SDA0',
        'SDB0',
        'SDP0',
        'SDA1',
        'SDB1',
        'SDP1',
        'SDA2',
        'SDB2',
        'SDP2',
        'SDA3',
        'SDB3',
        'SDP3',
    )
    CORR_FAMS = {'x': 'CH', 'y': 'CV', 'delta': 'SRFCav'}

    def __init__(self, model=None, w_bpm=1.0, rcond=1e-3):
        """.

        Args:
            model: pyaccel model. If None, a fitted SI model with
                corrected tunes is created.
            w_bpm (float): weight of the zero-orbit constraint at the BPMs
                (0 disables it).
            rcond (float): relative cutoff of the singular values
                (regularization), passed to numpy.linalg.lstsq.
        """
        self.w_bpm = w_bpm
        self.rcond = rcond
        self.model = model
        if self.model is None:
            model = _si.fitted_models.vertical_dispersion_and_coupling(
                _si.create_accelerator()
            )
            tune = TuneCorr(model, 'SI')
            tune.correct_parameters((49.16, 14.22))
            self.model = tune.model
        self.famdata = _si.get_family_data(self.model)
        self.model.cavity_on = True
        self.model.radiation_on = 1

        # {family: [sextupole indices]}
        self.fams_idcs = {
            f: [
                int(i)
                for i in _pa.lattice.find_indices(self.model, 'fam_name', f)
            ]
            for f in self.SEXT_FAMS
        }
        self.sext_idcs = _pa.lattice.flatten(self.fams_idcs.values())
        self.bpms_idcs = _pa.lattice.flatten(self.famdata['BPM']['index'])
        self.nr_corr = {
            p: len(_pa.lattice.flatten(self.famdata[f]['index']))
            for p, f in self.CORR_FAMS.items()
        }

        # rows of each family within one plane block of the sextupole matrix
        self._fam_rows, n = {}, 0
        for fam, idcs in self.fams_idcs.items():
            self._fam_rows[fam] = _np.arange(n, n + len(idcs))
            n += len(idcs)
        self.nr_sexts_total = n

        self.orm_sextupoles = None  # shape (2*nsext, nch+ncv+1)
        self.orm_bpms = None  # shape (2*nbpm, nch+ncv+1)
        self.spos_sexts = _pa.lattice.find_spos(self.model, self.sext_idcs)
        self.spos_bpms = _pa.lattice.find_spos(self.model, self.bpms_idcs)

    @property
    def families(self):
        """Sextupole family names."""
        return self.SEXT_FAMS

    def nr_sexts(self, fam):
        """Number of sextupoles in family `fam`."""
        return len(self.fams_idcs[fam])

    # ------------------------- response matrices --------------------------
    def calc_matrices(self):
        """Compute orbit response matrices numerically (tracking)."""

        # sextupoles as observation points
        respmat_sexts = _OrbRespmat(self.model, acc='SI')
        respmat_sexts.bpm_idx = self.sext_idcs
        mat_sexts = respmat_sexts.get_respm()

        respmat_bpms = _OrbRespmat(self.model, acc='SI')
        mat_bpms = respmat_bpms.get_respm()

        # rows: [x; y]; columns: [CH, CV, RF]
        self.orm_sextupoles = mat_sexts
        self.orm_bpms = mat_bpms

    # ------------------------------ bumps ---------------------------------
    def calc_kicks(self, bumps, constrain_others=False):
        """Compute the kicks needed to impose the requested bumps.

        Both planes are solved together in a single (coupled) problem.

        Args:
            bumps (dict): {family: {'bump_x': val, 'bump_y': val}} with
                values in [um], applied to every sextupole of the family.
                Each of the keys 'bump_x' and 'bump_y' is optional, but at
                least one must be present. A plane without its key is not
                constrained in that family.
                Example: {'SDA0': {'bump_x': 100, 'bump_y': -50},
                          'SFB1': {'bump_x': 30}}
            constrain_others (bool): if True, orbit at sextupoles not
                requested (family or plane) is forced to zero; if False,
                it is left free.

        Returns:
            dict with:
                'kick_ch', 'kick_cv': kicks [urad]
                'kicks': CH + CV kicks concatenated (no RF)
                'orb_sext_x', 'orb_sext_y': resulting orbit at all
                    sextupoles [um]
                'orb_bpm_x', 'orb_bpm_y': resulting orbit at BPMs [um]
                'resid_x', 'resid_y': max |target - obtained| over the
                    requested sextupoles in each plane [um] (None if the
                    plane was not requested)
                'bumps': the requested bumps
        """
        self._check_bumps(bumps)
        if self.orm_sextupoles is None or self.orm_bpms is None:
            self.calc_matrices()

        nsext, nch = self.nr_sexts_total, self.nr_corr['x']
        offsets = {'x': 0, 'y': nsext}

        target = _np.zeros(2 * nsext)
        requested = _np.zeros(2 * nsext, dtype=bool)
        for fam, bump in bumps.items():
            for plane, off in offsets.items():
                key = f'bump_{plane}'
                if key in bump:
                    rows = off + self._fam_rows[fam]
                    target[rows] = bump[key]
                    requested[rows] = True
        mask = requested | constrain_others

        amat = _np.vstack([
            self.orm_sextupoles[mask],
            self.w_bpm * self.orm_bpms,
        ])
        bvec = _np.r_[target[mask], _np.zeros(self.orm_bpms.shape[0])]
        kicks = _np.linalg.lstsq(amat, bvec, rcond=self.rcond)[0]

        orb_sext = self.orm_sextupoles @ kicks
        orb_bpm = self.orm_bpms @ kicks
        nbpm = orb_bpm.size // 2

        out = {
            'kick_ch': kicks[:nch],
            'kick_cv': kicks[nch:],
            'kicks': kicks,
            'orb_sext_x': orb_sext[:nsext],
            'orb_sext_y': orb_sext[nsext:],
            'orb_bpm_x': orb_bpm[:nbpm],
            'orb_bpm_y': orb_bpm[nbpm:],
            'bumps': bumps,
        }
        for plane, off in offsets.items():
            sel = requested[off : off + nsext]
            if sel.any():
                err = orb_sext[off : off + nsext] - target[off : off + nsext]
                out[f'resid_{plane}'] = _np.max(_np.abs(err[sel]))
            else:
                out[f'resid_{plane}'] = None
        return out

    # ------------------------ jacobians -----------------------------------

    def calc_jacobians(self):
        mod = self.model[:]
        orbrespmat = _OrbRespmat(mod, 'SI')
        Jk = _np.zeros((320 * 281, self.nr_sexts_total))
        Jks = _np.zeros((320 * 281, self.nr_sexts_total))

        jac_idcs = 0
        for fam in self.families:
            print(f'Measuring familiy {fam}')
            delta_kl = 0.001
            idcs = _pa.lattice.flatten(self.famdata[fam]['index'])
            for i, idc in enumerate(idcs):
                print(f'\tMeasuring magnet {i}')
                KL0 = _pa.lattice.get_attribute(orbrespmat.model, 'KL', idc)
                KsL0 = _pa.lattice.get_attribute(orbrespmat.model, 'KsL', idc)

                # normal grad
                _pa.lattice.set_attribute(
                    orbrespmat.model, 'KL', idc, delta_kl
                )
                orm_pos = orbrespmat.get_respm()

                _pa.lattice.set_attribute(
                    orbrespmat.model, 'KL', idc, -delta_kl
                )
                orm_neg = orbrespmat.get_respm()

                _pa.lattice.set_attribute(orbrespmat.model, 'KL', idc, KL0)

                Jk_mag = (orm_pos - orm_neg) / (2 * delta_kl)
                Jk[:, jac_idcs + i] = Jk_mag.ravel()

                # skew grad
                _pa.lattice.set_attribute(
                    orbrespmat.model, 'KsL', idc, delta_kl
                )
                orm_pos = orbrespmat.get_respm()

                _pa.lattice.set_attribute(
                    orbrespmat.model, 'KsL', idc, -delta_kl
                )
                orm_neg = orbrespmat.get_respm()

                _pa.lattice.set_attribute(orbrespmat.model, 'KsL', idc, KsL0)

                Jks_mag = (orm_pos - orm_neg) / (2 * delta_kl)
                Jks[:, jac_idcs + i] = Jks_mag.ravel()

            jac_idcs += len(idcs)

        SL = _pa.lattice.get_attribute(orbrespmat.model, 'SL', self.sext_idcs)
        Jx = Jk * SL[None, :]  # Jacobian per magnet
        Jy = -Jks * SL[None, :]  # Jacobian per magnet

        Jr = _np.hstack((Jx, Jy))

        # Jx_F = _np.zeros((320 * 281, 21))  # Jacobian per family

        # for i, (fam, mask) in enumerate(self._fam_rows.items()):
        #     Jx_F[:, i] = Jx[:, mask].sum(axis=1)

        return Jk, Jks, Jr

    # ------------------------- analysis and plots -------------------------
    def _targets(self, bumps, plane):
        """Target bump per sextupole row (NaN where not requested)."""
        target = _np.full(self.nr_sexts_total, _np.nan)
        for fam, bump in bumps.items():
            if f'bump_{plane}' in bump:
                target[self._fam_rows[fam]] = bump[f'bump_{plane}']
        return target

    def analyze_results(self, res, verbose=True):
        """Summarize how well the requested bumps were implemented.

        Args:
            res (dict): output of calc_kicks.
            verbose (bool): print a summary table.

        Returns:
            dict {plane: stats} with, for each plane ('x', 'y'):
                'kick_max', 'kick_rms', 'kick_argmax': CM kick statistics
                    [urad] (argmax is the CM index within CH/CV)
                'bpm_max', 'bpm_rms': orbit at BPMs [um] (leakage of the
                    bump out of the sextupoles)
                'others_max': max |orbit| at sextupoles not requested
                    in this plane [um] (collateral bumps)
                'resid_max': max |target - obtained| at requested
                    sextupoles [um]
                'bpm_rms_over_target': BPM rms / mean |target|
                'fams': {family: {'target', 'mean', 'min', 'max',
                    'rel_err'}} obtained orbit at the family sextupoles
        """
        stats = {}
        for plane in ('x', 'y'):
            target = self._targets(res['bumps'], plane)
            req = ~_np.isnan(target)
            orb_s = res[f'orb_sext_{plane}']
            orb_b = res[f'orb_bpm_{plane}']
            kick = res['kick_ch' if plane == 'x' else 'kick_cv']
            st = {
                'kick_max': _np.max(_np.abs(kick)),
                'kick_rms': _np.sqrt(_np.mean(kick**2)),
                'kick_argmax': int(_np.argmax(_np.abs(kick))),
                'bpm_max': _np.max(_np.abs(orb_b)),
                'bpm_rms': _np.sqrt(_np.mean(orb_b**2)),
                'others_max': (
                    _np.max(_np.abs(orb_s[~req])) if (~req).any() else None
                ),
                'resid_max': None,
                'bpm_rms_over_target': None,
                'fams': {},
            }
            if req.any():
                st['resid_max'] = _np.max(_np.abs(orb_s[req] - target[req]))
                st['bpm_rms_over_target'] = st['bpm_rms'] / _np.mean(
                    _np.abs(target[req])
                )
                for fam, bump in res['bumps'].items():
                    if f'bump_{plane}' not in bump:
                        continue
                    rows = self._fam_rows[fam]
                    tgt, obt = bump[f'bump_{plane}'], orb_s[rows]
                    st['fams'][fam] = {
                        'target': tgt,
                        'mean': _np.mean(obt),
                        'min': _np.min(obt),
                        'max': _np.max(obt),
                        'rel_err': (
                            _np.max(_np.abs(obt - tgt)) / abs(tgt)
                            if tgt != 0
                            else None
                        ),
                    }
            stats[plane] = st

        if verbose:
            self._print_stats(stats)
        return stats

    @staticmethod
    def _print_stats(stats):
        """Print the statistics returned by analyze_results."""

        def fmt(val, spec='.2f'):
            return 'n/a' if val is None else format(val, spec)

        for plane, st in stats.items():
            print(f'--- plane {plane} ---')
            print(
                f'  kicks [urad]: max {st["kick_max"]:.2f} '
                f'(CM #{st["kick_argmax"]}), rms {st["kick_rms"]:.2f}'
            )
            print(
                f'  BPM orbit [um]: max {st["bpm_max"]:.2f}, '
                f'rms {st["bpm_rms"]:.2f}, '
                f'rms/target {fmt(st["bpm_rms_over_target"], ".3f")}'
            )
            print(
                f'  max |orbit| @ non-requested sextupoles [um]: '
                f'{fmt(st["others_max"])}'
            )
            print(
                f'  max residual @ requested sextupoles [um]: '
                f'{fmt(st["resid_max"])}'
            )
            for fam, fs in st['fams'].items():
                print(
                    f'  {fam}: target {fs["target"]:.1f}, obtained '
                    f'mean {fs["mean"]:.1f} [{fs["min"]:.1f}, '
                    f'{fs["max"]:.1f}], max rel. err '
                    f'{fmt(fs["rel_err"], ".1%")}'
                )

    def plot_results(self, res, show=True):
        """Plot kicks, orbit at sextupoles and orbit at BPMs.

        Orbit figures have three rows: x orbit, lattice (middle) and y
        orbit. Requested sextupoles are shown in red (obtained orbit) with
        black dashes marking the targets. A summary is printed
        (analyze_results).

        Args:
            res (dict): output of calc_kicks.
            show (bool): call matplotlib show at the end.

        Returns:
            (dict, list): statistics from analyze_results and the figures.
        """
        stats = self.analyze_results(res, verbose=True)
        fams = list(res['bumps'].keys())
        figs = []
        length = self.model.length

        def make_orbit_fig(title):
            """Figure with [x orbit, lattice, y orbit] sharing s axis."""
            fig, axs = _plt.subplots(
                3,
                1,
                sharex=True,
                figsize=(10, 8),
                gridspec_kw={'height_ratios': [4, 1, 4], 'hspace': 0.05},
            )
            fig.suptitle(title)
            return fig, axs[0], axs[1], axs[2]

        def finish_orbit_fig(ax_x, ax_lat, ax_y):
            """Draw the lattice after the data and fix the axes limits."""
            # fix limits from the data, before drawing the lattice
            # lims = []
            # for ax in (ax_x, ax_y):
            #     ax.relim()
            #     ax.autoscale_view(scalex=False)
            #     lims.append(ax.get_ylim())
            # ax_x.set_xlim(0, length)

            # draw_lattice may act on the current axes: make it the lattice
            # _plt.sca(ax_lat)
            _pa.graphics.draw_lattice(
                self.model,
                offset=0.5,
                height=0.5,
                gca=ax_lat,
                selection=('bpm', 'sextupole', 'dipole'),
                show_label=True,
            )
            # # restore limits in case draw_lattice changed them
            # ax_x.set_xlim(0, length)
            # ax_x.set_ylim(*lims[0])
            # ax_y.set_ylim(*lims[1])

            # ax_lat.patch.set_visible(False)
            for spine in ax_lat.spines.values():
                spine.set_visible(False)
            ax_lat.axis('off')

        # CM kicks
        fig, axs = _plt.subplots(2, 1, sharex=False, figsize=(10, 6))
        fig.suptitle(f'CMs kicks implementing bump @ {fams}')
        axs[0].plot(res['kick_ch'], 'o-', color='blue', mfc='none')
        axs[0].set_ylabel('hor kicks [urad]')
        axs[0].set_xlabel('CH index')
        axs[1].plot(res['kick_cv'], 'o-', color='red', mfc='none')
        axs[1].set_ylabel('ver kicks [urad]')
        axs[1].set_xlabel('CV index')
        fig.tight_layout()
        figs.append(fig)

        # orbit at sextupoles
        fig, ax_x, ax_lat, ax_y = make_orbit_fig(
            f'Orbit @ Sextupoles (bump @ {fams})'
        )
        for ax, plane in ((ax_x, 'x'), (ax_y, 'y')):
            target = self._targets(res['bumps'], plane)
            req = ~_np.isnan(target)
            orb = res[f'orb_sext_{plane}']
            ax.plot(
                self.spos_sexts[~req],
                orb[~req],
                'o',
                color='blue',
                label='other sextupoles',
            )
            if req.any():
                ax.plot(
                    self.spos_sexts[req],
                    orb[req],
                    'o',
                    color='green',
                    label='requested (obtained)',
                )
                ax.plot(
                    self.spos_sexts[req],
                    target[req],
                    '_',
                    color='k',
                    ms=12,
                    label='target',
                )
                for spos in self.spos_sexts[req]:
                    ax.axvline(spos, ls='--', alpha=0.4, color='k')
            ax.set_ylabel(f'orbit {plane} [um]')
            ax.legend(loc='upper right')
        ax_y.set_xlabel('s [m]')
        finish_orbit_fig(ax_x, ax_lat, ax_y)
        figs.append(fig)

        # orbit at BPMs
        fig, ax_x, ax_lat, ax_y = make_orbit_fig(
            f'Orbit @ BPMs (bump @ {fams})'
        )
        for ax, plane in ((ax_x, 'x'), (ax_y, 'y')):
            ax.plot(
                self.spos_bpms,
                res[f'orb_bpm_{plane}'],
                'o-',
                color='blue' if plane == 'x' else 'red',
            )
            req = ~_np.isnan(self._targets(res['bumps'], plane))
            for spos in self.spos_sexts[req]:
                ax.axvline(spos, ls='--', alpha=0.4, color='k')
            ax.set_ylabel(f'orbit {plane} [um]')
        ax_y.set_xlabel('s [m]')
        finish_orbit_fig(ax_x, ax_lat, ax_y)
        figs.append(fig)

        if show:
            _plt.show()
        return stats, figs

    def _check_bumps(self, bumps):
        """Validate the structure of the `bumps` dictionary."""
        bad = [f for f in bumps if f not in self.fams_idcs]
        if bad:
            raise ValueError(
                f'Invalid families: {bad}. Valid: {self.SEXT_FAMS}'
            )
        for fam, bump in bumps.items():
            keys = set(bump)
            if not keys or not keys <= {'bump_x', 'bump_y'}:
                raise ValueError(
                    f"Family {fam}: expected keys 'bump_x' and/or 'bump_y', "
                    f'got {sorted(keys)}.'
                )
