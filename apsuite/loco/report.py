from apsuite.loco.analysis import LOCOAnalysis
from fpdf import FPDF
import datetime


class LOCOReport(FPDF):
    """."""

    def __init__(self, loco_data=None):
        """."""
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.loco_data = loco_data

    def header(self):
        """."""
        self.image('header.jpg', x=10, y=6, w=40, h=15)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 6, 'SIRIUS LOCO Report', 0, 0, 'C')
        self.set_font('Arial', '', 12)
        today = datetime.date.today()
        self.cell(0, 6, '{:s}'.format(today.strftime("%Y-%m-%d")), 0, 0, 'R')
        self.ln(10)

    def footer(self):
        """."""
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_title(self, title, loc_y=None):
        """."""
        self.set_font('Arial', '', 12)
        self.set_fill_color(215)
        if loc_y is not None:
            self.set_y(loc_y)
        self.cell(0, 6, f'{title:s}', 0, 1, 'C', 1)
        self.ln(2)

    def page_body(self, images):
        """."""
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        # if len(images) == 3:
        #     self.image(images[0], 15, 25, self.WIDTH - 30)
        #     self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        #     self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        # elif len(images) == 2:
        #     self.image(images[0], 15, 25, self.WIDTH - 30)
        #     self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        # else:
        self.image(images[0], 15, 50, self.WIDTH - 30)
        self.image(images[1], 15, 150, self.WIDTH - 30)

    def loco_fingerprint(self):
        """."""
        setup = self.loco_data['setup']
        self.set_font('Arial', 'B', 10)
        table_cell_width = self.WIDTH/3
        table_cell_height = 5

        self.ln(table_cell_height)
        self.set_font('Arial', '', 10)
        tstamp = datetime.datetime.fromtimestamp(setup['timestamp'])
        tstamp = tstamp.strftime('%Y-%m-%d %H:%M:%S')
        data = (
            ('Timestamp', tstamp),
            ('Stored current', f"{setup['stored_current']:.2f} mA"),
            ('Orbit response matrix on ServConf', setup['orbmat_name']),
            ('RF Frequency', f"{setup['rf_frequency']/1e6:.6f} MHz"),
            ('Measured tune x', f"{setup['tunex']:.6f}"),
            ('Measured tune y', f"{setup['tuney']:.6f}"),
            ('SOFB buffer average', f"{setup['sofb_nr_points']:d}"),
            )
        for row in data:
            self.set_x(self.WIDTH/6)
            for datum in row:
                self.cell(
                    table_cell_width, table_cell_height,
                    str(datum), align='C', border=0)
            self.ln(table_cell_height)

    def config_table(self):
        """."""
        self.set_font('Arial', 'B', 10)
        table_cell_width = self.WIDTH/3.8
        table_cell_height = 5

        # self.ln(table_cell_height*5)
        columns = ('Config. Property', 'Value')
        self.set_x(self.WIDTH/3.8)
        for col in columns:
            self.cell(
                table_cell_width, table_cell_height,
                col, align='C', border=1)
        self.ln(table_cell_height)

        config = self.loco_data['config']
        data = (
            ('Tracking dimension', config.dim),
            ('Include dispersion', config.use_dispersion),
            ('Include diagonal blocks', config.use_diagonal),
            ('Include off-diagonal blocks', config.use_offdiagonal),
            ('Minimization method', config.min_method_str),
            ('Lambda LM',  f'{config.lambda_lm:.2e}'),
            ('Fixed lambda LM', f'{config.fixed_lambda:.2e}'),
            ('Jacobian manipulation', config.inv_method_str),
            ('Constraint delta KL total', config.constraint_deltak_total),
            ('Constraint delta KL step', config.constraint_deltak_step),
            ('Singular values method', config.svd_method_str),
            ('SV to be used:', config.svd_sel),
            ('SV threshold (s/smax):', config.svd_thre),
            ('Tolerance delta', config.tolerance_delta),
            ('Tolerance overfit', config.tolerance_overfit),
            ('Dipoles normal gradients', config.fit_dipoles),
            ('Quadrupoles normal gradients', config.fit_quadrupoles),
            ('Sextupoles normal gradients', config.fit_sextupoles),
            ('Use dipoles as families', config.use_dip_families),
            ('Use quadrupoles as families', config.use_quad_families),
            ('Dipoles skew gradients', config.fit_dipoles_coupling),
            ('Quadrupoles skew gradients', config.fit_quadrupoles_coupling),
            ('Sextupoles skew gradients', config.fit_sextupoles_coupling),
            ('Skew quadrupoles skew gradients', config.fit_skew_quadrupoles),
            ('Girders longitudinal shifts', config.fit_girder_shift),
            ('Fit BPM gains', config.fit_gain_bpm),
            ('Fit Corrector gains', config.fit_gain_corr),
            ('Fit BPM roll error', config.fit_roll_bpm),
            ('Horizontal delta kicks',
                str(config.delta_kickx_meas*1e6) + ' urad'),
            ('Vertical delta kicks',
                str(config.delta_kicky_meas*1e6) + ' urad'),
            ('RF delta frequency', str(config.delta_frequency_meas) + ' Hz'),
        )

        self.set_font('Arial', '', 10)
        for row in data:
            self.set_x(self.WIDTH/3.8)
            for datum in row:
                self.cell(
                    table_cell_width, table_cell_height,
                    str(datum), align='C', border=1)
            self.ln(table_cell_height)

    def add_fingerprint(self):
        """."""
        self.add_page()
        self.page_title('LOCO fingerprint')
        self.loco_fingerprint()
        self.page_title('LOCO configuration setup', loc_y=70)
        self.set_y(80)
        self.config_table()

    # def add_config(self):
    #     """."""
    #     self.add_page()
    #     self.page_title('LOCO configuration setup')
    #     self.config_table()

    def add_images_to_page(self, images):
        """."""
        self.add_page()
        self.page_body(images)

    def add_quadfit(self, df_stats):
        """."""
        self.add_page()
        self.page_title('Normal quadrupoles variations')
        self.image('quad_by_family.png', 10, 120, self.WIDTH - 30)
        self.image('quad_by_s.png', 10, 200, self.WIDTH - 30)
        self.df_to_table(df_stats)

    def add_skewquadfit(self):
        """."""
        self.add_page()
        self.page_title('Skew quadrupoles variations')
        self.image('skewquad_by_s.png', x=10, y=30, w=self.WIDTH - 30)
        self.page_title('BPMs and correctors gains', loc_y=100)
        self.image('gains.png', x=30, y=120, w=self.WIDTH*0.6)

    def add_beta_tune(self, df_tunes, df_betabeat):
        """."""
        self.add_page()
        self.page_title('Beta-beating')
        self.df_to_table(df_betabeat)
        self.image('beta_beating.png', x=10, y=50, w=self.WIDTH - 30)
        self.page_title('Tunes', loc_y=120)
        self.set_y(160)
        self.df_to_table(df_tunes)

    def add_dispersion_emittance(self, df_disp, df_emits):
        """."""
        self.add_page()
        self.page_title('Dispersion function')
        self.df_to_table(df_disp, tw=22, th=5)
        self.image('dispersion.png', x=20, y=60, w=self.WIDTH*0.8)
        self.page_title('Emittances', loc_y=200)
        self.set_y(220)
        self.df_to_table(df_emits)

    def df_to_table(self, df, tw=30, th=5):
        """."""
        table_cell_width = tw
        table_cell_height = th

        self.ln(table_cell_height)
        self.set_font('Arial', 'B', 10)

        # Loop over to print column names
        cols = df.columns
        self.set_x(self.WIDTH/(len(cols)+1))
        for col in cols:
            self.cell(
                table_cell_width, table_cell_height, col, align='C', border=1)
        cols = [c.replace(' ', '_') for c in cols]
        cols = [c.replace('[%]', '') for c in cols]
        cols = [c.replace('[mm]', '') for c in cols]
        df.columns = cols
        # Line break
        self.ln(table_cell_height)
        # Loop over to print each data in the table
        for row in df.itertuples():
            self.set_font('Arial', '', 10)
            self.set_x(self.WIDTH/(len(cols)+1))
            for col in cols:
                value = str(getattr(row, col))
                self.cell(
                    table_cell_width, table_cell_height, value,
                    align='C', border=1)
            self.ln(table_cell_height)

    def create_report(self, fname_report, fname_setup, fname_fit, folder=None):
        """."""
        if folder is not None:
            fname_setup = folder + fname_setup
            fname_fit = folder + fname_fit
        loco_anly = LOCOAnalysis(fname_setup=fname_setup, fname_fit=fname_fit)
        loco_anly.get_setup()
        mod, disp_nom = loco_anly.get_nominal_model()
        loco_data, orm_fit, disp_fit, disp_meas = loco_anly.get_loco_results()
        loco_data['setup'] = loco_anly.loco_setup
        config = loco_data['config']
        dnomi = config.matrix - config.goalmat
        dloco = orm_fit - config.goalmat

        loco_anly.plot_histogram(dnomi, dloco, save=True, fname='histogram')
        df_stats = loco_anly.plot_quadrupoles_gradients_by_family(
            mod, loco_data['fit_model'], save=True, fname='quad_by_family')
        loco_anly.plot_quadrupoles_gradients_by_s(
            mod, loco_data['fit_model'], save=True, fname='quad_by_s')
        loco_anly.plot_skew_quadrupoles(
            mod, loco_data['fit_model'], save=True, fname='skewquad_by_s')
        loco_anly.plot_gain(save=True, fname='gains')
        loco_anly.calc_twiss()
        df_tunes, df_betabeat = loco_anly.beta_and_tune()
        df_disp = loco_anly.dispersion()
        df_emits = loco_anly.emittance()

        self.loco_data = loco_data

        self.add_fingerprint()
        self.add_quadfit(df_stats)
        self.add_skewquadfit()
        # self.add_gain()
        self.add_beta_tune(df_tunes, df_betabeat)
        self.add_dispersion_emittance(df_disp, df_emits)

        self.output(fname_report + '.pdf', 'F')
