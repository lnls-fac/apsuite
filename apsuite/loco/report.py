"""Create LOCO Report."""

from apsuite.loco.analysis import LOCOAnalysis
from apsuite.loco.utils import LOCOUtils
from fpdf import FPDF
import datetime

TIME_FMT = "%Y-%m-%d %H:%M:%S"


class LOCOReport(FPDF):
    """."""

    def __init__(self, loco_data=None):
        """."""
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.loco_data = loco_data
        self.loco_analysis = None
        self._df_quad_stats = None
        self._df_tunes = None
        self._df_emits = None
        self._df_betabeat = None
        self._df_disp = None

    def header(self):
        """."""
        path = "/".join(__file__.split("/")[:-1])
        self.image(path + "/cnpem_lnls_logo.jpg", x=10, y=6, w=40, h=15)
        self.set_font("Arial", "B", 14)
        self.cell(0, 6, "SIRIUS LOCO Report", 0, 0, "C")
        self.set_font("Arial", "", 11)
        now = datetime.datetime.now()
        stg = "{:s}".format(now.strftime(TIME_FMT))
        self.cell(0, 6, stg, 0, 0, "R")
        self.ln(10)

    def footer(self):
        """."""
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")

    def page_title(self, title, loc_y=None):
        """."""
        self.set_font("Arial", "", 12)
        self.set_fill_color(215)
        if loc_y is not None:
            self.set_y(loc_y)
        self.cell(0, 6, f"{title:s}", 0, 1, "C", 1)
        self.ln(2)

    def loco_fingerprint(self):
        """."""
        setup = self.loco_data["setup"]
        self.set_font("Arial", "B", 10)
        table_cell_width = self.WIDTH / 2.5
        table_cell_height = 5

        self.set_font("Arial", "", 10)
        tstamp = datetime.datetime.fromtimestamp(setup["timestamp"])
        tstamp = tstamp.strftime(TIME_FMT)
        data = (("Measurement timestamp", tstamp),)
        if "method" in setup:
            data += (("Measurement method", setup["method"]),)
        if "sofb_nr_points" in setup:
            data += (("SOFB buffer average", f"{setup['sofb_nr_points']:d}"),)
        if "orbmat_name" in setup:
            data += (
                ("Orbit response matrix on ServConf", setup["orbmat_name"]),
            )
        data += (
            ("Stored current", f"{setup['stored_current']:.2f} mA"),
            ("RF Frequency", f"{setup['rf_frequency'] / 1e6:.6f} MHz"),
            ("Measured frac. tune x", f"{setup['tunex']:.4f}"),
            ("Measured frac. tune y", f"{setup['tuney']:.4f}"),
        )
        _xp = (self.WIDTH - table_cell_width * len(data[0])) / 2
        for idx, row in enumerate(data):
            self.set_x(_xp)
            for datum in row:
                self.cell(
                    table_cell_width,
                    table_cell_height,
                    str(datum),
                    align="C",
                    border=1,
                )
            tab_h = (
                table_cell_height / 3
                if idx == len(data)
                else table_cell_height
            )
            self.ln(tab_h)

    def config_table(self):
        """."""
        self.set_font("Arial", "B", 10)
        table_cell_width = self.WIDTH / 3.8
        table_cell_height = 5

        self.set_line_width(0.05)
        columns = ("Config. Property", "Value")

        _xp = (self.WIDTH - table_cell_width * len(columns)) / 2
        self.set_x(_xp)
        for col in columns:
            self.cell(
                table_cell_width, table_cell_height, col, align="C", border=1
            )
        self.ln(table_cell_height)

        cfg = self.loco_data["config"]
        cfg.update()
        nr_svd = cfg.svd_sel
        if nr_svd is not None:
            if nr_svd < 0:
                nr_svd = cfg.nr_fit_parameters + nr_svd
        latt_ver = self.loco_data["fit_model"].lattice_version
        data = (
            ("Lattice version", latt_ver),
            ("Tracking dimension", cfg.dim),
            ("Include dispersion", cfg.use_dispersion),
            ("Include diagonal blocks", cfg.use_diagonal),
            ("Include off-diagonal blocks", cfg.use_offdiagonal),
            ("Minimization method", cfg.min_method_str),
            ("Lambda LM", f"{cfg.lambda_lm:.2e}"),
            ("Fixed lambda LM", f"{cfg.fixed_lambda:.2e}"),
            ("Jacobian manipulation", cfg.inv_method_str),
            ("Constraint delta KL total", cfg.constraint_deltakl_total),
            ("Constraint delta KL step", cfg.constraint_deltakl_step),
            ("Nr. of BPMs", cfg.nr_bpm),
            ("Nr. of CHs", cfg.nr_ch),
            ("Nr. of CVs", cfg.nr_cv),
            (
                "Nr. of matrix datapoints",
                2 * cfg.nr_bpm * (cfg.nr_ch + cfg.nr_cv + 1),
            ),
            ("Nr. of fit parameters", cfg.nr_fit_parameters),
            ("Singular values (SV) method", cfg.svd_method_str),
            ("SV to be used:", nr_svd),
            ("SV threshold (s/smax):", cfg.svd_thre),
            ("Tolerance delta", cfg.tolerance_delta),
            ("Tolerance overfit", cfg.tolerance_overfit),
            ("Dipoles normal gradients", cfg.fit_dipoles),
            ("Quadrupoles normal gradients", cfg.fit_quadrupoles),
            ("Sextupoles normal gradients", cfg.fit_sextupoles),
            ("Use dipoles as families", cfg.use_dip_families),
            ("Use quadrupoles as families", cfg.use_quad_families),
            ("Dipoles skew gradients", cfg.fit_dipoles_coupling),
            ("Quadrupoles skew gradients", cfg.fit_quadrupoles_coupling),
            ("Sextupoles skew gradients", cfg.fit_sextupoles_coupling),
            ("Skew quadrupoles skew gradients", cfg.fit_skew_quadrupoles),
            ("Girders longitudinal shifts", cfg.fit_girder_shift),
            ("Fit BPM gains", cfg.fit_gain_bpm),
            ("Fit Corrector gains", cfg.fit_gain_corr),
            ("Fit BPM roll error", cfg.fit_roll_bpm),
            (
                "Horizontal delta kicks",
                str(cfg.delta_kickx_meas * 1e6) + " urad",
            ),
            (
                "Vertical delta kicks",
                str(cfg.delta_kicky_meas * 1e6) + " urad",
            ),
            ("RF delta frequency", str(cfg.delta_frequency_meas) + " Hz"),
        )

        self.set_font("Arial", "", 10)
        for idx, row in enumerate(data):
            self.set_x(_xp)
            for datum in row:
                self.cell(
                    table_cell_width,
                    table_cell_height,
                    str(datum),
                    align="C",
                    border=1,
                )
            tab_h = (
                table_cell_height / 3
                if idx == len(data)
                else table_cell_height
            )
            self.ln(tab_h)

    def add_fingerprint_and_config(self):
        """."""
        self.add_page()
        self.page_title("LOCO fingerprint")
        self.loco_fingerprint()
        self.page_title("LOCO configuration setup", loc_y=70)
        self.set_y(80)
        self.config_table()

    def add_images_to_page(self, images):
        """."""
        self.add_page()
        self.page_body(images)

    def add_histogram(self):
        """."""
        self.add_page()
        self.page_title("Fitting indicators")
        img_w = self.WIDTH - 30
        _xp = (self.WIDTH - img_w) / 2
        self.image("3dplot.png", x=_xp, y=40, w=img_w)
        self.image("histogram.png", x=_xp - 5, y=140, w=img_w)

    def add_quadfit(self):
        """."""
        self.add_page()
        self.page_title("Normal quadrupoles variations")
        img_w = self.WIDTH - 30
        _xp = (self.WIDTH - img_w) / 2
        self.image("quad_by_family.png", x=_xp, y=110, w=img_w)
        self.image("quad_by_s.png", x=_xp, y=200, w=img_w)
        self.df_to_table(self._df_quad_stats)

    def add_skewquadfit_ang_gains(self):
        """."""
        self.add_page()
        self.page_title("Skew quadrupoles variations")
        img_w = self.WIDTH - 30
        _xp = (self.WIDTH - img_w) / 2
        self.image("skewquad_by_s.png", x=_xp, y=30, w=img_w)
        self.page_title("Gains: BPMs and correctors", loc_y=90)
        img_w = self.WIDTH - 30
        _xp = (self.WIDTH - img_w) / 2
        self.image("gains.png", x=_xp, y=100, w=img_w)

    def add_tune_emit_and_optics(self):
        """."""
        self.add_page()
        self.page_title("Global parameters: tunes and emittances")
        self.df_to_table(self._df_tunes, nr_tables=2, idx_table=0)
        self.set_y(28)
        self.df_to_table(self._df_emits, nr_tables=2, idx_table=1)
        self.set_y(60)
        self.page_title("Optics: beta-beating")
        self.df_to_table(self._df_betabeat)
        img_w = self.WIDTH - 30
        img_x = (self.WIDTH - img_w) / 2
        img_y = 90
        self.image("beta_beating.png", x=img_x, y=img_y, w=img_w)
        self.set_y(150)
        self.page_title("Optics: dispersion")
        self.df_to_table(self._df_disp, tw=22, th=5)
        img_w = self.WIDTH - 45
        img_h = self.HEIGHT * 0.3
        img_x = (self.WIDTH - img_w) / 2
        img_y = 185
        self.image("dispersion.png", x=img_x, y=img_y, w=img_w, h=img_h)

    def df_to_table(self, df, tw=30, th=5, nr_tables=1, idx_table=0):
        """."""
        table_cell_width = tw
        table_cell_height = th

        self.ln(table_cell_height / 3)
        self.set_font("Arial", "B", 10)

        # Loop over to print column names
        cols = df.columns

        _xp = self.WIDTH / nr_tables
        _xp -= table_cell_width * len(cols)
        _xp /= 2
        if idx_table == 1:
            _xp += self.WIDTH / nr_tables
        _adj = 5 if not idx_table else -5
        _xp += _adj
        self.set_x(_xp)

        for col in cols:
            self.cell(
                table_cell_width, table_cell_height, col, align="C", border=1
            )
        cols = [c.replace(" ", "_") for c in cols]
        cols = [c.replace("[%]", "") for c in cols]
        cols = [c.replace("[mm]", "") for c in cols]
        df.columns = cols
        # Line break
        self.ln(table_cell_height)
        # Loop over to print each data in the table
        for idx, row in enumerate(df.itertuples()):
            self.set_font("Arial", "", 10)
            self.set_x(_xp)
            for col in cols:
                value = str(getattr(row, col))
                self.cell(
                    table_cell_width,
                    table_cell_height,
                    value,
                    align="C",
                    border=1,
                )
            tab_h = (
                table_cell_height / 3
                if idx == df.shape[0]
                else table_cell_height
            )
            self.ln(tab_h)

    def create_report(self, fname_setup, fname_fit, folder=None):
        """."""
        if folder is not None:
            fname_setup = folder + fname_setup
            fname_fit = folder + fname_fit
        loco_anly = LOCOAnalysis(fname_setup=fname_setup, fname_fit=fname_fit)

        loco_anly.get_setup()
        mod, _ = loco_anly.get_nominal_model()
        loco_data, orm_fit, *_ = loco_anly.get_loco_results()
        loco_data["setup"] = loco_anly.loco_setup
        config = loco_data["config"]
        config.matrix = LOCOUtils.apply_all_gain(
            config.matrix, config.gain_bpm, config.roll_bpm, config.gain_corr
        )
        dnomi = config.matrix - config.goalmat
        dloco = orm_fit - config.goalmat

        loco_anly.plot_histogram(
            dnomi, dloco, fname=folder + "histogram"
        )
        loco_anly.plot_3d_fitting(
            dnomi, dloco, fname=folder + "3dplot"
        )

        df_quad_stats = loco_anly.plot_quadrupoles_gradients_by_family(
            nom_model=mod,
            fit_model=loco_data["fit_model"],
            fname=folder + "quad_by_family"
        )
        self._df_quad_stats = df_quad_stats
        loco_anly.plot_quadrupoles_gradients_by_s(
            nom_model=mod,
            fit_model=loco_data["fit_model"],
            fname=folder + "quad_by_s"
        )
        loco_anly.plot_skew_quadrupoles(
            mod, loco_data["fit_model"], fname=folder + "skewquad_by_s"
        )
        loco_anly.plot_gain(fname=folder + "gains")
        self._df_emits = loco_anly.emittance_and_coupling()

        loco_anly.calc_twiss()
        self._df_tunes, self._df_betabeat = loco_anly.beta_and_tune(
            fname=folder + "beta_beating"
        )
        self._df_disp = loco_anly.dispersion(
            fname=folder + "dispersion_function"
        )

        # loco_anly.calc_edteng()
        # self._df_tunes, self._df_betabeat = \
        #   loco_anly.beta_and_tune(twiss=False)
        # self._df_disp = loco_anly.dispersion(twiss=False)

        loco_anly.save_quadrupoles_variations(
            mod,
            loco_data["fit_model"],
            fname_family=folder + "quad_family_average",
            fname_trims=folder + "quad_trims_deltakl_zero_average",
        )
        loco_anly.save_skew_quadrupoles_variations(
            mod, loco_data["fit_model"], fname=folder + "skewquad_deltaksl"
        )

        self.loco_data = loco_data
        self.loco_analysis = loco_anly

        self.add_fingerprint_and_config()
        self.add_histogram()
        self.add_quadfit()
        self.add_skewquadfit_ang_gains()
        self.add_tune_emit_and_optics()

        self.output(folder + "report.pdf", "F")
