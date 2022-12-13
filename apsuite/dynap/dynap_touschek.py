"""Calculate dynamic aperture."""

import numpy as _np

import matplotlib.pyplot as _mpyplot
import matplotlib.gridspec as _mgridspec
import matplotlib.colors as _mcolors
import matplotlib.text as _mtext

import pyaccel.tracking as _pytrack
import pyaccel.lattice as _pylatt
import pyaccel.optics as _pyopt

from .base import BaseClass as _BaseClass


class TouschekAccepParams():
    """."""

    def __init__(self):
        """."""
        self.nrturns = 512
        self.turn_by_turn = True
        self.x_min = -12.0e-3
        self.x_max = 0.0
        self.de_min = 0.0
        self.de_max = 4.0e-3
        self.x_nrpts = 70
        self.de_nrpts = 30
        self.y_offset = 0.1e-3
        self.xl_off = 1e-5
        self.yl_off = 1e-5
        self.intnux = 49.0
        self.intnuy = 14.0

    def __str__(self):
        """."""
        strng = ''
        strng += 'nrturns      : {:d}\n'.format(self.nrturns)
        strng += 'turn_by_turn : {:s}\n'.format(str(self.turn_by_turn))
        strng += 'x_nrpts      : {:d}\n'.format(self.x_nrpts)
        strng += 'de_nrpts     : {:d}\n'.format(self.de_nrpts)
        strng += 'x_min [m]    : {:.2g}\n'.format(self.x_min)
        strng += 'x_max [m]    : {:.2g}\n'.format(self.x_max)
        strng += 'de_min       : {:.2g}\n'.format(self.de_min)
        strng += 'de_max       : {:.2g}\n'.format(self.de_max)
        strng += 'y_offset [m] : {:.3g}\n'.format(self.y_offset)
        strng += 'xl_off [rad] : {:.2g}\n'.format(self.xl_off)
        strng += 'yl_off [rad] : {:.2g}\n'.format(self.yl_off)
        strng += 'intnux       : {:.2f} (for graphs)\n'.format(self.intnux)
        strng += 'intnuy       : {:.2f} (for graphs)\n'.format(self.intnuy)
        return strng


class TouschekAccep(_BaseClass):
    """."""

    def __init__(self, accelerator):
        """."""
        super().__init__()
        self._acc = accelerator
        self.params = TouschekAccepParams()
        self.data['x_in'] = _np.array([], dtype=float)
        self.data['de_in'] = _np.array([], dtype=float)
        self.data['rout'] = _np.array([], dtype=float)
        self.data['lost_turn'] = _np.array([], dtype=int)
        self.data['lost_element'] = _np.array([], dtype=int)
        self.data['lost_plane'] = _np.array([], dtype=int)
        self.x_dynap = _np.array([], dtype=float)
        self.de_dynap = _np.array([], dtype=float)
        self.x_freq_ini = _np.array([], dtype=float)
        self.x_freq_fin = _np.array([], dtype=float)
        self.y_freq_ini = _np.array([], dtype=float)
        self.y_freq_fin = _np.array([], dtype=float)
        self.x_diffusion = _np.array([], dtype=float)
        self.y_diffusion = _np.array([], dtype=float)
        self.diffusion = _np.array([], dtype=float)

    def calc_linear_acceptance(self):
        """."""
        if self.params.de_min < 0 or self.params.de_max < 0:
            raise ValueError('delta must be a positive vector.')

        delta = _np.linspace(
            self.params.de_min, self.params.de_max, self.params.de_nrpts)

        self._acc.cavity_on = False
        self._acc.radiation_on = False
        self._acc.vchamber_on = False

        spos = _np.asarray(_pylatt.find_spos(self._acc))
        vh_pos = _pylatt.get_attribute(self._acc, 'hmax')
        vh_neg = _pylatt.get_attribute(self._acc, 'hmin')

        twi0 = _pyopt.calc_twiss(self._acc, energy_offset=0)

        tune_p = _np.zeros((delta.size, 2))
        tune_n = _np.zeros((delta.size, 2))

        curlyh_pos = _np.full((delta.size, spos.size), _np.inf)
        curlyh_neg = _np.full((delta.size, spos.size), _np.inf)
        accep_pos = _np.zeros(delta.shape)
        accep_neg = _np.zeros(delta.shape)

        twi_pos, twi_neg = [], []
        for i, dlt in enumerate(delta):
            try:
                twi_p, *_ = _pyopt.calc_twiss(self._acc, dlt)
                twi_n, *_ = _pyopt.calc_twiss(self._acc, -dlt)
            except (_pyopt.OpticsException, _pytrack.TrackingException):
                twi_pos.append(None)
                twi_neg.append(None)

            # positive energies
            btx = twi_p.betax
            alx = twi_p.alphax
            crx = twi_p.rx
            cpx = twi_p.px
            drx = crx - twi0.rx
            dpx = cpx - twi0.px
            curlyh_pos[i] = self.calcH(btx, alx, drx, dpx)

            app_pos = (vh_pos - crx)
            app_neg = (vh_neg - crx)
            apper_loc = _np.minimum(app_pos*app_pos, app_neg*app_neg)
            accep_pos[i] = _np.min(apper_loc/btx)

            # negative energies
            btx = twi_n.betax
            alx = twi_n.alphax
            crx = twi_n.rx
            cpx = twi_n.px
            drx = crx - twi0.rx
            dpx = cpx - twi0.px
            curlyh_neg[i] = self.calcH(btx, alx, drx, dpx)

            app_pos = (vh_pos - crx)
            app_neg = (vh_neg - crx)
            apper_loc = _np.minimum(app_pos*app_pos, app_neg*app_neg)
            accep_neg[i] = _np.min(apper_loc/btx)

            twi_pos.append(twi_p)
            twi_neg.append(twi_n)

        self.data['delta'] = delta
        self.data['twi_pos'] = twi_pos
        self.data['twi_neg'] = twi_neg
        self.data['curlyh_pos'] = curlyh_pos
        self.data['curlyh_neg'] = curlyh_neg
        self.data['accep_pos'] = accep_pos
        self.data['accep_neg'] = accep_neg

    @staticmethod
    def calcH(beta, alpha, x, xl):
        """."""
        gamma = (1 + alpha**2) / beta
        return beta*xl**2 + 2*alpha*x*xl + gamma*x**2

    def calculate_nonlinear_acceptance(self):
        """."""

        x_line = _np.linspace(
            self.params.x_min, self.params.x_max, self.params.x_nrpts)
        de_in, x_in = _np.meshgrid(de_line, x_line)

        orb = _np.zeros(6)
        if self._acc.cavity_on:
            orb = _np.squeeze(_pytrack.find_orbit6(self._acc))
        else:
            orb[:4] = _np.squeeze(_pytrack.find_orbit4(self._acc))

        rin = _np.tile(orb, (de_in.size, 1)).T
        rin[0, :] += x_in.ravel()
        rin[1, :] += self.params.xl_off
        rin[2, :] += self.params.y_offset
        rin[3, :] += self.params.yl_off
        rin[4, :] += de_in.ravel()

        out = _pytrack.ring_pass(
            self._acc, rin, nr_turns=self.params.nrturns,
            turn_by_turn=self.params.turn_by_turn)

        self.data['x_in'] = x_in
        self.data['de_in'] = de_in
        self.data['rout'] = out[0]
        self.data['lost_turn'] = out[2]
        self.data['lost_element'] = out[3]
        self.data['lost_plane'] = out[4]

        n_turns = 131
        H = _np.linspace(0, 4e-6, 30)
        ep = _np.linspace(0.02, 0.06, 20)
        en = -ep

        self._acc.cavity_on = True
        self._acc.radiation_on = True
        self._acc.vchamber_on = True

        beta_p = _np.zeros(delta.size)
        beta_n = _np.zeros(delta.size)
        orb4d_p = _np.zeros(delta.size, 4)
        orb4d_n = _np.zeros(delta.size, 4)

        for i, _ in enumerate(twi_pos):
            twi_p = twi_pos[i][0]
            twi_n = twi_neg[i][0]
            beta_p[i] = twi_p.betax
            orb4d_p[i] = [twi_p.rx, twi_p.px, twi_p.ry, twi_p.py]
            beta_n[i] = twi_n.betax
            orb4d_n[i] = [twi_n.rx, twi_n.px, twi_n.ry, twi_n.py]

        beta_p = _np.interp(delta, beta_p, ep)
        orb4d_p = _np.interp(delta, orb4d_p, ep)
        beta_n = _np.interp(-delta, beta_n, en)
        orb4d_n = _np.interp(-delta, orb4d_n, en)

        orb6d = _pytrack.find_orbit6(self._acc)

        # ---negative-energies--- #
        en_ini, ch_ini = _np.meshgrid(en, H)
        px_ini = _np.sqrt(ch_ini / beta_n[None, :])

        rin = _np.zeros((6, ch_ini.size))
        rin[:4, :] = _np.tile(orb4d_n, (1, H.size))
        rin[2] += px_ini.ravel()
        rin[3] += 1e-5
        rin[5] = orb6d[5] + en_ini.ravel()
        rin[6] = orb6d[6]

        rout, _, lost_plane, *_ = _pytrack.ring_pass(
            self._acc, rin, nr_turns=n_turns,)

        nlost = _np.array([l is None for l in lost_plane], dtype=bool)
        nlost.reshape(en_ini.shape)

        [a_max, ind_dyn] = max(lost, 2)
        ind_dyn = max(ind_dyn - a_max + (~a_max)*(length(H)-1), 1);

        Adyn_n = zeros(1, length(en));
        for j = 1:length(en)
            Adyn_n(j) = H_0(j,ind_dyn(j));
        end
        Adyn_n = interp1(en,Adyn_n,-delta)';

        %---positive-energies---%
        [H0,EP] = meshgrid(H,ep);
        H0 = H0(:);
        EP = EP(:);
        Xl = sqrt( H0./repmat(beta_p,1,length(H))' );

        Rin = zeros(6, length(H0));
        Rin(1:4,:) = repmat(orb4d_p,1,length(H));
        Rin(2,:) = Rin(2,:) + Xl';
        Rin(3,:) = Rin(3,:) + 1e-5;
        Rin(5,:) = orb6d(5) + EP';
        Rin(6,:) = orb6d(6);

        Rou_p = [Rin,ringpass(ring_6d,Rin,n_turns)];
        Rou_p = reshape(Rou_p , 6, length(en), length(H), []);

        H_0 = repmat(H, length(ep), []);
        lost = isnan( squeeze(Rou_p(1,:,:,end)) );

        [a_max, ind_dyn] = max(lost, [], 2);
        ind_dyn = max(ind_dyn - a_max + (~a_max)*(length(H)-1), 1);

        Adyn_p = zeros(1, length(ep));
        for j = 1:length(ep)
            Adyn_p(j) = H_0(j,ind_dyn(j));
        end
        Adyn_p = interp1(ep,Adyn_p,delta)';


        %% Calculate Aperture and Acceptance
        A_p(1) = min([Aphys_p(1), Aphys_n(1), Adyn_p(1)]);
        for j = 2:length(delta)
            A_p(j) = min([Aphys_p(j), Aphys_n(j), Adyn_p(j), A_p(j-1)]);
        end

        A_n(1) = min([Aphys_p(1), Aphys_n(1), Adyn_n(1)]);
        for j = 2:length(delta)
            A_n(j) = min([Aphys_p(j), Aphys_n(j), Adyn_n(j), A_n(j-1)]);
        end

        [sel c_p] = max( repmat(A_p,1,length(Accep.s))' < H_p, [], 2);
        c_p = c_p + (~sel)*(length(delta)-1);
        Accep.pos = delta(c_p);

        [sel c_n] = max( repmat(A_n,1,length(Accep.s))' < H_n, [], 2);
        c_n = c_n + (~sel)*(length(delta)-1);
        Accep.neg = -delta(c_n);

        info.delta = delta;
        info.twi_p = twi_p;
        info.twi_n = twi_n;
        info.A_p = A_p;
        info.A_n = A_n;
        info.Aphys_p = Aphys_p;
        info.Aphys_n = Aphys_n;
        info.Adyn_p = Adyn_p;
        info.Adyn_n = Adyn_n;
        info.H_p = H_p;
        info.H_n = H_n;
        info.tune_n = tune_n - floor(tune_n);
        info.tune_p = tune_p - floor(tune_p);

    def process_data(self):
        """."""
        self.calc_dynap()
        self.calc_fmap()

    def calc_dynap(self):
        """."""
        de_in = self.data['de_in']
        x_in = self.data['x_in']
        lost_plane = self.data['lost_plane']

        self.de_dynap, self.x_dynap = self._calc_dynap(de_in, x_in, lost_plane)

    def calc_fmap(self):
        """."""
        rout = self.data['rout']
        lost_plane = self.data['lost_plane']

        fx1, fx2, fy1, fy2, diffx, diffy, diff = super()._calc_fmap(
            rout, lost_plane)

        self.x_freq_ini = fx1
        self.x_freq_fin = fx2
        self.y_freq_ini = fy1
        self.y_freq_fin = fy2
        self.x_diffusion = diffx
        self.y_diffusion = diffy
        self.diffusion = diff

    def map_resons2real_plane(self, resons, maxdist=1e-5, min_diffusion=1e-3):
        """."""
        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini
        diff = self.diffusion
        return super()._map_resons2real_plane(
            freqx, freqy, diff, resons, maxdist=maxdist, mindiff=min_diffusion)

    # Make figures
    def make_figure_diffusion(
            self, contour=True, resons=None, orders=3, symmetry=1,
            maxdist=1e-5, min_diffusion=1e-3):
        """."""
        fig = _mpyplot.figure(figsize=(7, 7))
        grid = _mgridspec.GridSpec(2, 20)
        grid.update(
            left=0.15, right=0.86, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.25)
        axx = _mpyplot.subplot(grid[0, :19])
        ayy = _mpyplot.subplot(grid[1, :19])
        cbaxes = _mpyplot.subplot(grid[:, -1])
        axx.name = 'XY'
        ayy.name = 'Tune'

        diff = self.diffusion
        diff = _np.log10(diff)
        norm = _mcolors.Normalize(vmin=-10, vmax=-2)

        if contour:
            axx.contourf(
                self.data['de_in']*1e2,
                self.data['x_in']*1e3,
                diff.reshape(self.data['x_in'].shape),
                norm=norm, cmap='jet')
        else:
            axx.scatter(
                self.data['de_in'].ravel()*1e2,
                self.data['x_in'].ravel()*1e3,
                c=diff, norm=norm, cmap='jet')
            axx.grid(False)
        axx.set_xlabel(r'$\delta$ [%]')
        axx.set_ylabel('X [mm]')

        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini
        line = ayy.scatter(
            freqx, freqy, c=diff, norm=norm, cmap='jet')
        ayy.set_xlabel(r'$\nu_x$')
        ayy.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = ayy.axis()
            resons = self.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)

        map2xy = self.map_resons2real_plane(
            resons=resons, maxdist=maxdist, min_diffusion=min_diffusion)
        xdata = self.data['x_in'].ravel()*1e3
        dedata = self.data['de_in'].ravel()*1e2
        for (coefx, coefy, _), ind in zip(resons, map2xy):
            order = int(_np.abs(coefx) + _np.abs(coefy))
            idx = order - 1
            axx.scatter(
                dedata[ind], xdata[ind],
                c=self.COLORS[idx % len(self.COLORS)])

        self.add_resonances_to_axis(ayy, resons=resons)

        cbar = fig.colorbar(line, cax=cbaxes)
        cbar.set_label('Diffusion')
        fig.canvas.mpl_connect('button_press_event', self._onclick)

        ann = ayy.annotate(
            '', xy=(0, 0), xycoords='data',
            xytext=(20, 20), textcoords='offset points',
            arrowprops={'arrowstyle': '->'},
            bbox={'boxstyle': 'round', 'fc': 'w'})
        ann.set_visible(False)
        fig.canvas.mpl_connect('motion_notify_event', self._onhover)

        return fig, axx, ayy

            figure; hold on; grid on; box on;
            title('Energy Acceptance')
            xlabel('Pos [m]')
            ylabel('$\delta\ [\%]$','interpreter','latex')
            plot(Accep.s, 100*[Accep.pos; Accep.neg], 'b')

            figure; hold on; grid on; box on;
            title('Energy Aperture')
            xlabel('$\delta\ [\%]$','interpreter','latex')
            ylabel('A [um]')
            plot(100*[-fliplr(delta), delta],1e6*[fliplr(Aphys_n'), Aphys_p'], 'b', 'LineWidth', 2);
            plot(100*[-fliplr(delta), delta],1e6*[fliplr(Adyn_n'),  Adyn_p'],  'm', 'LineWidth', 2);
            plot(100*[-fliplr(delta), delta],1e6*[fliplr(A_n'),  A_p'],  '--r');

            if isstruct(flag_dyn) || flag_dyn

                lost_turn_p = zeros(length(ep),length(H));
                lost_turn_n = zeros(length(ep),length(H));
                for i = 1:length(H)
                    for j = 1:length(ep)
                        turn = find(isnan(Rou_p(1,j,i,:)),1,'first');
                        if isempty(turn), turn = NaN; end
                        lost_turn_p(j,i) = turn;

                        turn = find(isnan(Rou_n(1,j,i,:)),1,'first');
                        if isempty(turn), turn = NaN; end
                        lost_turn_n(j,i) = turn;
                    end
                end
                norm = max(max(max(lost_turn_p,lost_turn_n)));
                r_p = reshape((lost_turn_p-1)/norm,1,[])';
                r_n = reshape((lost_turn_n-1)/norm,1,[])';

                figure; hold on; box on; grid on;
                title(sprintf('max = %d',norm));
                scatter(reshape(EP,1,[])', reshape(H0,1,[])', 12, [r_p, 1-r_p, 4*r_p.*(1-r_p)], 'filled');
                scatter(reshape(EN,1,[])', reshape(H0,1,[])', 12, [r_n, 1-r_n, 4*r_n.*(1-r_n)], 'filled');

                lost_turn_col = lost_turn_p(:);
                cdf = zeros(1,norm);
                for j = 1:norm, cdf(j) = sum(lost_turn_col>=j); end

                lost_turn_col = lost_turn_n(:);
                for j = 1:norm, cdf(j) = cdf(j) + sum(lost_turn_col>=j); end

                figure; box on; grid on; plot(cdf/cdf(1));
                info.cdf = cdf;
            end
        end



    def _onclick(self, event):
        if not event.dblclick or event.inaxes is None:
            return
        fig = event.inaxes.figure
        ax_nu = [ax for ax in fig.axes if ax.name == 'Tune'][0]
        ax_xy = [ax for ax in fig.axes if ax.name == 'XY'][0]

        xdata = self.data['x_in'].ravel()*1e3
        dedata = self.data['de_in'].ravel()*1e2
        xfreq = self.params.intnux + self.x_freq_ini
        yfreq = self.params.intnuy + self.y_freq_ini

        if event.inaxes == ax_nu:
            xdiff = xfreq - event.xdata
            ydiff = yfreq - event.ydata
        if event.inaxes == ax_xy:
            xdiff = dedata - event.xdata
            ydiff = xdata - event.ydata

        ind = _np.nanargmin(_np.sqrt(xdiff*xdiff + ydiff*ydiff))
        ax_xy.scatter(dedata[ind], xdata[ind], c='k')
        ax_nu.scatter(xfreq[ind], yfreq[ind], c='k')
        fig.canvas.draw()

    def _onhover(self, event):
        if event.inaxes is None:
            return
        fig = event.inaxes.figure
        ax_nu = [ax for ax in fig.axes if ax.name == 'Tune'][0]
        chil = ax_nu.get_children()
        ann = [c for c in chil if isinstance(c, _mtext.Annotation)][0]
        if event.inaxes != ax_nu:
            ann.set_visible(False)
            fig.canvas.draw()
            return

        for line in ax_nu.lines:
            if line.contains(event)[0] and line.name.startswith('reson'):
                ann.set_text(line.name.split(':')[1])
                ann.xy = (event.xdata, event.ydata)
                ann.set_visible(True)
                break
        else:
            ann.set_visible(False)
        fig.canvas.draw()

    def make_figure_map_real2tune_planes(
            self, resons=None, orders=3, symmetry=1):
        """."""
        fig = _mpyplot.figure(figsize=(7, 7))
        grid = _mgridspec.GridSpec(2, 20)
        grid.update(
            left=0.15, right=0.86, top=0.97, bottom=0.1,
            hspace=0.25, wspace=0.25)
        axx = _mpyplot.subplot(grid[0, :19])
        ayy = _mpyplot.subplot(grid[1, :19])
        cbx = _mpyplot.subplot(grid[0, -1])
        cby = _mpyplot.subplot(grid[1, -1])

        freqx = self.params.intnux + self.x_freq_ini
        freqy = self.params.intnuy + self.y_freq_ini

        # X
        norm = _mcolors.Normalize(
            vmin=self.params.x_min*1e3,
            vmax=self.params.x_max*1e3)
        line = axx.scatter(
            freqx, freqy, c=self.data['x_in'].ravel()*1e3,
            norm=norm, cmap='jet')
        axx.set_xlabel(r'$\nu_x$')
        axx.set_ylabel(r'$\nu_y$')

        if resons is None:
            bounds = axx.axis()
            resons = self.calc_resonances_for_bounds(
                bounds, orders=orders, symmetry=symmetry)

        self.add_resonances_to_axis(axx, resons=resons)

        cbar = fig.colorbar(line, cax=cbx)
        cbar.set_label('X [mm]')

        # Y
        norm = _mcolors.Normalize(
            vmin=self.params.de_min*1e2,
            vmax=self.params.de_max*1e2)
        line = ayy.scatter(
            freqx, freqy, c=self.data['de_in'].ravel()*1e2,
            norm=norm, cmap='jet')
        ayy.set_xlabel(r'$\nu_x$')
        ayy.set_ylabel(r'$\nu_y$')
        self.add_resonances_to_axis(ayy, resons=resons)

        cbar = fig.colorbar(line, cax=cby)
        cbar.set_label(r'$\delta$ [%]')

        return fig, axx, ayy
