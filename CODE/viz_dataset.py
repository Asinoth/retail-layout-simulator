"""Dataset loading orchestrator (UI layer).

The heavy lifting lives in:
  - dataset_adapters.py    file -> normalized DataFrame + ValidationReport
  - dataset_calibration.py normalized df -> empirical simulation params
  - dataset_layout.py      params -> multi-floor-safe shop layout
  - dataset_validation.py  params + sim -> goodness-of-fit tests
  - dataset_provenance.py  source file -> ProvenanceRecord

This module wires those together behind the Layout-tab 'Dataset' button.
"""

from __future__ import annotations

import json
import os
import time
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from dataset_adapters import (
    TRANSACTIONAL_ADAPTERS,
    TRAJECTORY_ADAPTERS,
    best_transactional_adapter,
    best_trajectory_adapter,
    detect_schema_kind,
    read_any,
    list_excel_sheets,
    read_excel_sheets,
    looks_like_omnichannel,
    load_omnichannel_bundle,
    OmnichannelRetailAdapter,
)
from dataset_calibration import (
    calibrate_transactional, calibrate_trajectory, calibrate_omnichannel,
    CalibratedParams, SpatialParams,
)
from dataset_layout import build_layout_from_calibration
from dataset_provenance import stamp


# --- 3-button quick-load registry -----------------------------------------
# The Dataset button presents these three options and auto-resolves each
# path under ``DATASETS/`` next to the simulator's source. The transactional
# adapter's validation report dialog still appears as a confirm step.

def _resolve_datasets_dir():
    """Locate the DATASETS/ folder regardless of repo layout. After the
    CODE/ + DATASETS/ reorg the datasets sit one level ABOVE this module
    (sibling of CODE/); we also accept a co-located DATASETS/ and the
    current working directory as fallbacks."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(here), 'DATASETS'),  # sibling of CODE/ (new layout)
        os.path.join(here, 'DATASETS'),                   # co-located (legacy flat)
        os.path.join(os.getcwd(), 'DATASETS'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


DATASETS_DIR = _resolve_datasets_dir()

DATASET_REGISTRY = {
    'uci': {
        'label': 'UCI Online Retail II',
        'desc':  'Transactional retail — UK e-commerce 2009-2011, two sheets auto-loaded',
        'path':  os.path.join(DATASETS_DIR, 'UCI Online Retail II .xlsx.xlsx'),
        'kind':  'uci_xlsx',
    },
    'omnichannel': {
        'label': 'Omnichannel Retail',
        'desc':  'Aggregated in-store behavioral data — 134 product families, '
                 'demand / dwell / impulse / hour×day arrivals',
        'path':  os.path.join(DATASETS_DIR, 'Omnichannel-Retail-Datasets-main'),
        'kind':  'omnichannel_folder',
    },
    'opentraj': {
        'label': 'OpenTraj ETH (trajectories)',
        'desc':  'Pedestrian trajectories — calibrates walking-speed / dwell / '
                 'heat-map onto the current shop',
        'path':  os.path.join(DATASETS_DIR, 'OpenTraj-master', 'datasets',
                              'ETH', 'seq_eth', 'obsmat.txt'),
        'kind':  'trajectory_file',
    },
}


class DatasetMixin:
    """CSV/Excel dataset loading + calibration + layout generation."""

    # -- 3-button quick chooser (the Dataset button entry point) --------
    def _on_dataset_button(self):
        """Show the 3-option chooser, resolve the chosen path, hard-stop
        any running sim (so the layout rebuild can't dangle live customer
        refs), and dispatch to the appropriate branch -- which still shows
        its validation-report dialog as the confirm step."""
        choice = self._open_dataset_chooser()
        if not choice:
            return
        info = DATASET_REGISTRY.get(choice)
        if not info:
            return
        path = info['path']
        if not os.path.exists(path):
            messagebox.showerror(
                "Dataset not found",
                f"Could not find:\n  {path}\n\n"
                f"Make sure the DATASETS/ folder sits next to the "
                f"simulator's Python files.",
                parent=self.tk_root)
            return

        # Stop any running sim BEFORE the layout clear so customers can't
        # be left holding dangling refs to walls / items.
        try:
            self.customer_simulation.hard_stop()
        except Exception:
            pass

        kind = info['kind']
        if kind == 'uci_xlsx':
            self._auto_load_transactional_xlsx(path)
        elif kind == 'omnichannel_folder':
            self._load_omnichannel_dataset(path, [])
        elif kind == 'trajectory_file':
            try:
                df, read_notes = read_any(path)
            except Exception as e:
                messagebox.showerror(
                    "Read failed",
                    f"Could not read {os.path.basename(path)}:\n\n{e}",
                    parent=self.tk_root)
                return
            self._load_trajectory_dataset(df, path, read_notes)

    def _open_dataset_chooser(self):
        """Modal Toplevel with 3 dataset buttons. Returns the key chosen
        ('uci' / 'omnichannel' / 'opentraj') or None if cancelled."""
        dlg = tk.Toplevel(self.tk_root)
        dlg.title("Load a calibrated dataset")
        dlg.transient(self.tk_root)
        dlg.geometry("560x320")
        result = {'choice': None}

        tk.Label(dlg, text="Pick a dataset to calibrate against:",
                 font=('Arial', 12, 'bold')
                 ).pack(anchor='w', padx=16, pady=(14, 2))
        tk.Label(dlg,
                 text="The path is resolved automatically; the validation "
                      "report still appears as a confirm step.",
                 fg='#555', font=('Arial', 9, 'italic'),
                 wraplength=520, justify='left'
                 ).pack(anchor='w', padx=16, pady=(0, 10))

        for key in ('uci', 'omnichannel', 'opentraj'):
            info = DATASET_REGISTRY[key]
            row = tk.Frame(dlg)
            row.pack(fill=tk.X, padx=16, pady=4)

            def _pick(k=key):
                result['choice'] = k
                dlg.destroy()

            tk.Button(row, text=info['label'],
                      font=('Arial', 11, 'bold'),
                      width=26, anchor='w',
                      bg='#2d7a2d', fg='white',
                      command=_pick).pack(side=tk.LEFT)
            tk.Label(row, text=info['desc'], fg='#333',
                     font=('Arial', 9), wraplength=270, justify='left'
                     ).pack(side=tk.LEFT, padx=10)

        bottom = tk.Frame(dlg)
        bottom.pack(fill=tk.X, padx=16, pady=12)
        tk.Button(bottom, text="Cancel",
                  command=dlg.destroy).pack(side=tk.RIGHT)

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        return result['choice']

    def _auto_load_transactional_xlsx(self, filename):
        """Read all sheets of an .xlsx (no sheet picker), then run the
        standard transactional pipeline. Used by the chooser for UCI."""
        try:
            sheet_pairs = list_excel_sheets(filename)
        except Exception as e:
            messagebox.showerror(
                "Read failed",
                f"Could not open {os.path.basename(filename)}:\n\n{e}",
                parent=self.tk_root)
            return
        sheet_names = [n for n, _ in sheet_pairs]
        if not sheet_names:
            messagebox.showerror(
                "Workbook empty", "No sheets found.", parent=self.tk_root)
            return
        try:
            df, read_notes = read_excel_sheets(filename, sheet_names)
        except Exception as e:
            messagebox.showerror(
                "Read failed",
                f"Could not read sheets:\n\n{e}", parent=self.tk_root)
            return
        self._run_transactional_pipeline(df, filename, read_notes)

    # -- Tool that the existing 'Hand mode' toggle still expects ------
    def _toggle_hand_mode(self):
        self.hand_mode = bool(self.hand_mode_var.get())
        widget = self.canvas.get_tk_widget()
        widget.configure(cursor='hand2' if self.hand_mode else '')
        if self.hand_mode:
            self._remove_resize_handles()

    # -- Public entry point: Layout tab -> Dataset button -------------
    def _on_load_dataset(self):
        """Pick a file, validate, calibrate, lay out the shop, stamp
        provenance into analytics, then refresh the Validation tab."""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Tabular data", "*.csv;*.tsv;*.xlsx;*.xls;*.txt;*.dat"),
                ("CSV files",   "*.csv;*.tsv"),
                ("Excel files", "*.xlsx;*.xls"),
                ("Text files",  "*.txt;*.dat"),
                ("All files",   "*.*"),
            ],
            title="Load dataset",
            parent=self.tk_root,
        )
        if not filename:
            return

        # Excel workbooks may contain multiple sheets (e.g. UCI Online
        # Retail II ships one per calendar year). Show a picker before
        # reading so we don't silently drop data.
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext in ("xlsx", "xls"):
            try:
                sheets = list_excel_sheets(filename)
            except Exception as e:
                messagebox.showerror(
                    "Read failed",
                    f"Could not open {os.path.basename(filename)}:\n\n{e}",
                    parent=self.tk_root)
                return
            if len(sheets) > 1:
                chosen = self._pick_excel_sheets(filename, sheets)
                if not chosen:
                    return
                try:
                    df, read_notes = read_excel_sheets(filename, chosen)
                except Exception as e:
                    messagebox.showerror(
                        "Read failed",
                        f"Could not read selected sheets:\n\n{e}",
                        parent=self.tk_root)
                    return
            else:
                try:
                    df, read_notes = read_any(filename)
                except Exception as e:
                    messagebox.showerror(
                        "Read failed",
                        f"Could not read {os.path.basename(filename)}:\n\n{e}",
                        parent=self.tk_root)
                    return
        else:
            try:
                df, read_notes = read_any(filename)
            except Exception as e:
                messagebox.showerror(
                    "Read failed",
                    f"Could not read {os.path.basename(filename)}:\n\n{e}",
                    parent=self.tk_root)
                return

        # Omnichannel is an aggregated multi-CSV bundle (no transactions);
        # handle it before the transactional/trajectory split.
        if looks_like_omnichannel(df, filename):
            self._load_omnichannel_dataset(filename, read_notes)
            return

        # Detect schema kind and pick the appropriate adapter family.
        kind = detect_schema_kind(df, filename=filename)
        if kind == "trajectory":
            self._load_trajectory_dataset(df, filename, read_notes)
            return

        self._run_transactional_pipeline(df, filename, read_notes)

    def _run_transactional_pipeline(self, df, filename, read_notes):
        """Run the standard transactional flow on an already-read DataFrame.

        Factored out of ``_on_load_dataset`` so both the legacy file-picker
        and the quick chooser (``_on_dataset_button``) share one
        implementation: adapter -> validation-report dialog -> calibrate ->
        ``hard_stop`` -> layout (naive baseline) -> seed -> provenance ->
        success popup."""
        adapter = best_transactional_adapter(df, filename=filename)
        try:
            normalized, report = adapter.adapt(df)
        except Exception as e:
            messagebox.showerror("Adapter failed",
                                 f"Adapter {adapter.name} crashed:\n\n{e}\n\n"
                                 f"{traceback.format_exc()[-600:]}",
                                 parent=self.tk_root)
            return
        for n in read_notes:
            report.info.append(n)

        # Show validation report, let the user pick an adapter / set
        # assumed conversion / cancel.
        choice = self._show_validation_dialog(report, df, filename)
        if choice is None:
            return

        adapter, assumed_conv = choice
        # Re-run the (possibly different) adapter if the user switched.
        try:
            normalized, report = adapter.adapt(df)
        except Exception as e:
            messagebox.showerror("Adapter failed",
                                 f"Adapter {adapter.name} crashed:\n\n{e}",
                                 parent=self.tk_root)
            return
        if report.is_blocking():
            messagebox.showerror(
                "Cannot calibrate",
                "Required columns are still missing — see the validation "
                "report. Add the missing columns to the source file or "
                "pick a different adapter.",
                parent=self.tk_root,
            )
            return

        # Calibrate
        try:
            params = calibrate_transactional(
                normalized,
                assumed_conversion_rate=assumed_conv,
                currency=report.extra.get('currency', 'currency'),
            )
        except Exception as e:
            messagebox.showerror("Calibration failed",
                                 f"{e}\n\n{traceback.format_exc()[-600:]}",
                                 parent=self.tk_root)
            return

        # Stop any running sim BEFORE the layout clear so customers can't
        # be left holding dangling refs to walls / items the rebuild drops.
        try:
            self.customer_simulation.hard_stop()
        except Exception:
            pass

        # Build layout (multi-floor safe). Default naive=True so the GA has
        # room to demonstrate a measurable lift.
        try:
            layout_stats = build_layout_from_calibration(self, params)
        except Exception as e:
            messagebox.showerror("Layout failed",
                                 f"{e}\n\n{traceback.format_exc()[-600:]}",
                                 parent=self.tk_root)
            return

        # Seed simulation analytics with empirical values
        try:
            params.seed_into(self.customer_simulation)
        except Exception as e:
            messagebox.showerror("Seeding failed",
                                 f"Could not seed simulation analytics:\n\n{e}",
                                 parent=self.tk_root)
            return

        # Stamp provenance into analytics so reports can cite the dataset.
        prov = stamp(
            source_path=filename,
            adapter_name=adapter.name,
            adapter_version=adapter.version,
            rows_in=report.rows_in,
            rows_kept=report.rows_kept,
            currency=report.extra.get('currency'),
            seed=getattr(self.customer_simulation, 'seed', None),
            extra={
                'layout_stats': layout_stats,
                'assumed_conversion_rate': assumed_conv,
            },
        )
        self.customer_simulation.analytics['provenance'] = prov.to_dict()

        # Remember the last params on the visualizer so the Validation tab
        # can rerun goodness-of-fit after the user pushes the simulation.
        self._last_calibration_params = params
        self._last_validation_report = report

        # Refresh UI
        try:
            self.redraw()
        except Exception:
            pass
        try:
            # If the validation tab has been built, refresh its metadata
            # pane immediately so the user sees the dataset name + sample
            # counts even before they run the simulation.
            if hasattr(self, '_refresh_validation_tab_metadata'):
                self._refresh_validation_tab_metadata()
        except Exception:
            pass

        # Final confirmation popup
        messagebox.showinfo(
            "Dataset loaded",
            f"Adapter:        {adapter.name} v{adapter.version}\n"
            f"Source:         {os.path.basename(filename)}\n"
            f"Rows in/kept:   {report.rows_in:,} / {report.rows_kept:,}\n"
            f"Invoices:       {params.n_invoices:,}\n"
            f"Products:       {params.n_unique_products:,} "
            f"(top {layout_stats['items']} placed)\n"
            f"Categories:     {params.n_unique_categories} "
            f"(→ {layout_stats['sections']} sections)\n"
            f"Shop size:      {layout_stats['width_m']:.1f} × "
            f"{layout_stats['height_m']:.1f} m\n"
            f"Arrival rate:   {params.arrivals_per_hour:.1f} invoices/hr\n"
            f"Conversion:     {assumed_conv:.0%} (assumed)\n\n"
            f"The Validation tab now shows goodness-of-fit vs. this dataset.",
            parent=self.tk_root,
        )

    # -- Omnichannel branch (aggregated, multi-CSV bundle) ------------
    def _load_omnichannel_dataset(self, filename, read_notes):
        """Aggregate-calibration branch for the Omnichannel Retail bundle.
        The four CSVs are merged on Aisle ID and the measured family-level
        aggregates map directly into CalibratedParams -- no fabricated
        transactions. Runs on the main thread (modal dialog)."""
        try:
            families, arrivals, notes = load_omnichannel_bundle(filename)
        except Exception as e:
            messagebox.showerror(
                "Omnichannel load failed",
                f"Could not assemble the Omnichannel bundle:\n\n{e}\n\n"
                f"Pick the dataset folder (or any CSV inside it).",
                parent=self.tk_root)
            return

        adapter = OmnichannelRetailAdapter()
        report = adapter.report_for_bundle(families, arrivals)
        for n in list(read_notes) + list(notes):
            report.info.append(n)

        if not self._show_aggregate_validation_dialog(report, filename):
            return

        try:
            params = calibrate_omnichannel(families, arrivals)
        except Exception as e:
            messagebox.showerror("Calibration failed",
                                 f"{e}\n\n{traceback.format_exc()[-600:]}",
                                 parent=self.tk_root)
            return

        # Stop any running sim BEFORE the layout clear so customers can't
        # be left holding dangling refs to walls / items the rebuild drops.
        try:
            self.customer_simulation.hard_stop()
        except Exception:
            pass

        try:
            layout_stats = build_layout_from_calibration(self, params)
        except Exception as e:
            messagebox.showerror("Layout failed",
                                 f"{e}\n\n{traceback.format_exc()[-600:]}",
                                 parent=self.tk_root)
            return
        try:
            params.seed_into(self.customer_simulation)
        except Exception as e:
            messagebox.showerror("Seeding failed",
                                 f"Could not seed simulation analytics:\n\n{e}",
                                 parent=self.tk_root)
            return

        prov = stamp(
            source_path=filename,
            adapter_name=adapter.name,
            adapter_version=adapter.version,
            rows_in=report.rows_in,
            rows_kept=report.rows_kept,
            currency='USD',
            seed=getattr(self.customer_simulation, 'seed', None),
            extra={
                'kind': 'aggregate_retail',
                'layout_stats': layout_stats,
                'n_product_families': params.n_unique_products,
                'n_zones': params.n_unique_categories,
                'conversion_rate_source': params.conversion_rate_source,
            },
        )
        self.customer_simulation.analytics['provenance'] = prov.to_dict()
        self._last_calibration_params = params
        self._last_validation_report = report

        try:
            self.redraw()
        except Exception:
            pass
        try:
            if hasattr(self, '_refresh_validation_tab_metadata'):
                self._refresh_validation_tab_metadata()
        except Exception:
            pass

        messagebox.showinfo(
            "Omnichannel dataset loaded",
            f"Adapter:          {adapter.name} v{adapter.version}\n"
            f"Source:           {os.path.basename(filename)}\n"
            f"Product families: {params.n_unique_products:,}\n"
            f"In-store zones:   {params.n_unique_categories} "
            f"(→ {layout_stats['sections']} sections)\n"
            f"Items placed:     {layout_stats['items']}\n"
            f"Shop size:        {layout_stats['width_m']:.1f} × "
            f"{layout_stats['height_m']:.1f} m\n"
            f"Arrival rate:     {params.arrivals_per_hour:.1f} customers/hr\n"
            f"Conversion:       {params.assumed_conversion_rate:.0%} "
            f"(estimated from purchase probabilities)\n\n"
            f"Aggregate source: basket / per-visit-revenue distributions are "
            f"parametric (not directly observed) and flagged as such in the "
            f"Validation tab.",
            parent=self.tk_root,
        )

    def _show_aggregate_validation_dialog(self, report, filename):
        """Modal report dialog for the aggregate Omnichannel bundle.
        Returns True if the user clicks Calibrate, else False."""
        dlg = tk.Toplevel(self.tk_root)
        dlg.title("Omnichannel (aggregate) dataset validation")
        dlg.transient(self.tk_root)
        dlg.geometry("780x580")
        result = {'ok': False}

        tk.Label(dlg, text=os.path.basename(filename),
                 font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=8)
        tk.Label(dlg,
                 text="Aggregated behavioral dataset (no transactions). "
                      "Family-level purchase %, demand, dwell, impulse rate, "
                      "price and in-store zones map directly into the "
                      "simulator's parameters.",
                 wraplength=740, justify='left', fg='#555'
                 ).pack(anchor='w', padx=10)

        text = tk.Text(dlg, wrap=tk.NONE, bg='#001a33', fg='white',
                       font=('Courier New', 9))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        text.insert(tk.END, report.to_text())
        text.config(state=tk.DISABLED)

        bottom = tk.Frame(dlg)
        bottom.pack(fill=tk.X, padx=10, pady=8)

        def on_calibrate():
            result['ok'] = True
            dlg.destroy()

        def on_cancel():
            result['ok'] = False
            dlg.destroy()

        tk.Button(bottom, text="Calibrate & generate layout",
                  command=on_calibrate, bg='#2d7a2d', fg='white',
                  font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        tk.Button(bottom, text="Cancel",
                  command=on_cancel).pack(side=tk.RIGHT, padx=8)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        return result['ok']

    # -- Trajectory branch (ATC + generic) ---------------------------
    def _load_trajectory_dataset(self, df, filename, read_notes):
        """Spatial-calibration branch: trajectory datasets calibrate
        speed / dwell / traffic-density distributions onto the EXISTING
        shop layout (we don't rebuild the floor plan from trajectory
        data -- the user's layout is what we want to validate the
        simulator's spatial behavior against).

        Multi-floor note: trajectory data is mapped onto floor 1 (the
        observed floors in ATC are ground-floor mall corridors).
        """
        adapter = best_trajectory_adapter(df, filename=filename)
        try:
            normalized, report = adapter.adapt(df)
        except Exception as e:
            messagebox.showerror(
                "Trajectory adapter failed",
                f"Adapter {adapter.name} crashed:\n\n{e}\n\n"
                f"{traceback.format_exc()[-600:]}",
                parent=self.tk_root)
            return
        for n in read_notes:
            report.info.append(n)

        # Show validation dialog (adapter picker reuses the trajectory
        # registry by recognizing the report kind).
        choice = self._show_trajectory_validation_dialog(report, df, filename)
        if choice is None:
            return
        adapter = choice
        try:
            normalized, report = adapter.adapt(df)
        except Exception as e:
            messagebox.showerror(
                "Trajectory adapter failed",
                f"Adapter {adapter.name} crashed:\n\n{e}",
                parent=self.tk_root)
            return
        if report.is_blocking():
            messagebox.showerror(
                "Cannot calibrate",
                "Required trajectory columns missing — see the validation "
                "report.", parent=self.tk_root)
            return

        try:
            sparams = calibrate_trajectory(
                normalized,
                target_width_m=float(self.width),
                target_height_m=float(self.height),
                heat_resolution=getattr(self.customer_simulation,
                                        'heat_map_resolution', 20),
            )
        except Exception as e:
            messagebox.showerror(
                "Spatial calibration failed",
                f"{e}\n\n{traceback.format_exc()[-600:]}",
                parent=self.tk_root)
            return

        try:
            sparams.seed_into(self.customer_simulation)
        except Exception as e:
            messagebox.showerror(
                "Seeding failed",
                f"Could not seed spatial parameters:\n\n{e}",
                parent=self.tk_root)
            return

        # Stamp provenance
        prov = stamp(
            source_path=filename,
            adapter_name=adapter.name,
            adapter_version=adapter.version,
            rows_in=report.rows_in,
            rows_kept=report.rows_kept,
            currency=None,
            seed=getattr(self.customer_simulation, 'seed', None),
            extra={
                'kind': 'trajectory',
                'bbox_m': sparams.bbox_m,
                'n_tracks': sparams.n_tracks,
                'span_seconds': sparams.span_seconds,
            },
        )
        # Trajectory provenance is stored under a separate key so a
        # transactional-data provenance record isn't overwritten.
        self.customer_simulation.analytics['spatial_provenance'] = prov.to_dict()

        self._last_spatial_params = sparams
        self._last_trajectory_report = report

        try:
            self.redraw()
        except Exception:
            pass
        try:
            if hasattr(self, '_refresh_validation_tab_metadata'):
                self._refresh_validation_tab_metadata()
        except Exception:
            pass

        speed_summary = (
            f"  mean={sparams.speeds_m_s.mean():.2f} m/s, "
            f"std={sparams.speeds_m_s.std():.2f}"
            if sparams.speeds_m_s.size else "  (no velocity samples)"
        )
        messagebox.showinfo(
            "Trajectory dataset loaded",
            f"Adapter:        {adapter.name} v{adapter.version}\n"
            f"Source:         {os.path.basename(filename)}\n"
            f"Rows in/kept:   {report.rows_in:,} / {report.rows_kept:,}\n"
            f"Tracks:         {sparams.n_tracks:,}\n"
            f"Span:           {sparams.span_seconds/3600.0:.1f} hours\n"
            f"Speed (m/s):\n{speed_summary}\n"
            f"Heat-map cells: {sparams.n_unique_zones:,} populated\n\n"
            f"The simulator's heat-map and calibration block have been "
            f"updated. Validation tab shows empirical-vs-simulated "
            f"speed/dwell overlays.",
            parent=self.tk_root,
        )

    def _show_trajectory_validation_dialog(self, report, df, filename):
        """Modal dialog for trajectory data -- same layout as the
        transactional version, minus the conversion-rate field."""
        dlg = tk.Toplevel(self.tk_root)
        dlg.title("Trajectory dataset validation")
        dlg.transient(self.tk_root)
        dlg.geometry("780x560")
        result = {'choice': None}

        tk.Label(dlg, text=os.path.basename(filename),
                 font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=8)

        row = tk.Frame(dlg)
        row.pack(fill=tk.X, padx=10)
        tk.Label(row, text="Adapter:").pack(side=tk.LEFT)
        names = [a.name for a in TRAJECTORY_ADAPTERS]
        adapter_var = tk.StringVar(value=report.adapter_name)
        cmb = ttk.Combobox(row, textvariable=adapter_var,
                           values=names, state='readonly', width=32)
        cmb.pack(side=tk.LEFT, padx=8)

        report_holder = {'report': report}
        text = tk.Text(dlg, wrap=tk.NONE, bg='#001a33', fg='white',
                       font=('Courier New', 9))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        def refresh_report(*_):
            adapter = next(a for a in TRAJECTORY_ADAPTERS
                           if a.name == adapter_var.get())
            try:
                _, r = adapter.adapt(df)
            except Exception as e:
                text.config(state=tk.NORMAL)
                text.delete('1.0', tk.END)
                text.insert(tk.END, f"Adapter crashed: {e}")
                text.config(state=tk.DISABLED)
                return
            report_holder['report'] = r
            text.config(state=tk.NORMAL)
            text.delete('1.0', tk.END)
            text.insert(tk.END, r.to_text())
            text.config(state=tk.DISABLED)

        cmb.bind('<<ComboboxSelected>>', refresh_report)
        refresh_report()

        bottom = tk.Frame(dlg)
        bottom.pack(fill=tk.X, padx=10, pady=8)

        def on_calibrate():
            adapter = next(a for a in TRAJECTORY_ADAPTERS
                           if a.name == adapter_var.get())
            if report_holder['report'].is_blocking():
                messagebox.showwarning(
                    "Cannot calibrate",
                    "Required columns missing -- see validation report.",
                    parent=dlg)
                return
            result['choice'] = adapter
            dlg.destroy()

        def on_cancel():
            result['choice'] = None
            dlg.destroy()

        tk.Button(bottom, text="Calibrate spatial parameters",
                  command=on_calibrate, bg='#2d7a2d', fg='white',
                  font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        tk.Button(bottom, text="Cancel",
                  command=on_cancel).pack(side=tk.RIGHT, padx=8)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        return result['choice']

    # -- Excel sheet picker ------------------------------------------
    def _pick_excel_sheets(self, filename, sheets):
        """Modal dialog that lists every sheet in the workbook with its
        row count and lets the user check which ones to load. Returns a
        list of sheet names, or None if the user cancels.

        For UCI Online Retail II (two yearly sheets) the default is to
        select both -- concatenating them gives the calibration the full
        2009-2011 window.
        """
        dlg = tk.Toplevel(self.tk_root)
        dlg.title("Pick sheets to load")
        dlg.transient(self.tk_root)
        dlg.geometry("520x420")
        result = {'sheets': None}

        tk.Label(dlg, text=os.path.basename(filename),
                 font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=(10, 2))
        tk.Label(dlg,
                 text=f"Workbook contains {len(sheets)} sheet(s). "
                      "Tick the ones to include; ticked sheets are "
                      "concatenated row-wise before calibration.",
                 wraplength=480, justify='left',
                 fg='#555').pack(anchor='w', padx=10, pady=(0, 8))

        list_frame = tk.Frame(dlg)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        vars_: list = []
        for i, (name, nrows) in enumerate(sheets):
            v = tk.BooleanVar(value=True)   # default: ALL sheets selected
            vars_.append((v, name, nrows))
            row = tk.Frame(list_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Checkbutton(row, variable=v).pack(side=tk.LEFT)
            tk.Label(row, text=name, font=('Arial', 10, 'bold'),
                     width=28, anchor='w').pack(side=tk.LEFT)
            rt = f"{nrows:,} rows" if nrows >= 0 else "(row count unavailable)"
            tk.Label(row, text=rt, fg='#666').pack(side=tk.LEFT)

        # Helpers
        btn_row = tk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=10, pady=4)

        def select_all():
            for v, *_ in vars_: v.set(True)
        def select_none():
            for v, *_ in vars_: v.set(False)

        tk.Button(btn_row, text="Select all", command=select_all).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Select none", command=select_none).pack(side=tk.LEFT, padx=6)

        bottom = tk.Frame(dlg)
        bottom.pack(fill=tk.X, padx=10, pady=10)

        def on_load():
            chosen = [name for v, name, _ in vars_ if v.get()]
            if not chosen:
                messagebox.showwarning("No sheets",
                                       "Pick at least one sheet.",
                                       parent=dlg)
                return
            result['sheets'] = chosen
            dlg.destroy()

        def on_cancel():
            result['sheets'] = None
            dlg.destroy()

        tk.Button(bottom, text="Load selected sheets",
                  command=on_load, bg='#2d7a2d', fg='white',
                  font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        tk.Button(bottom, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=8)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        return result['sheets']

    # -- Validation dialog (pre-calibration) -------------------------
    def _show_validation_dialog(self, report, df, filename):
        """Modal dialog showing the ValidationReport. Returns
        ``(adapter, assumed_conversion_rate)`` if the user clicks
        Calibrate, or ``None`` if they cancel.

        The dialog also lets the user pick a different adapter from the
        registry (e.g. force the generic adapter on a UCI file).
        """
        dlg = tk.Toplevel(self.tk_root)
        dlg.title("Dataset validation")
        dlg.transient(self.tk_root)
        dlg.geometry("780x620")
        result = {'choice': None}

        # -- Top: file name + adapter picker -------------
        top = tk.Frame(dlg)
        top.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(top, text=os.path.basename(filename),
                 font=('Arial', 11, 'bold')).pack(anchor='w')

        ap_row = tk.Frame(dlg)
        ap_row.pack(fill=tk.X, padx=10)
        tk.Label(ap_row, text="Adapter:").pack(side=tk.LEFT)
        names = [a.name for a in TRANSACTIONAL_ADAPTERS]
        adapter_var = tk.StringVar(value=report.adapter_name)
        cmb = ttk.Combobox(ap_row, textvariable=adapter_var,
                           values=names, state='readonly', width=32)
        cmb.pack(side=tk.LEFT, padx=8)

        # Re-validate when adapter changes.
        report_holder = {'report': report}
        text = tk.Text(dlg, wrap=tk.NONE, bg='#001a33', fg='white',
                       font=('Courier New', 9))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        def refresh_report(*_):
            adapter = next(a for a in TRANSACTIONAL_ADAPTERS
                           if a.name == adapter_var.get())
            try:
                _, r = adapter.adapt(df)
            except Exception as e:
                r = report   # keep old on failure
                text.config(state=tk.NORMAL)
                text.delete('1.0', tk.END)
                text.insert(tk.END, f"Adapter crashed: {e}")
                text.config(state=tk.DISABLED)
                return
            report_holder['report'] = r
            text.config(state=tk.NORMAL)
            text.delete('1.0', tk.END)
            text.insert(tk.END, r.to_text())
            text.config(state=tk.DISABLED)

        cmb.bind('<<ComboboxSelected>>', refresh_report)
        refresh_report()

        # -- Bottom: assumed conversion rate + buttons ---
        bottom = tk.Frame(dlg)
        bottom.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(bottom,
                 text="Assumed conversion (visitors → buyers):").pack(side=tk.LEFT)
        conv_var = tk.DoubleVar(value=0.30)
        tk.Entry(bottom, textvariable=conv_var, width=6).pack(side=tk.LEFT, padx=6)
        tk.Label(bottom,
                 text="(transactional data has no non-buyers; "
                      "this asserts a visitor base for conversion-rate "
                      "calibration)",
                 fg='#666', font=('Arial', 8, 'italic')).pack(side=tk.LEFT)

        btn = tk.Frame(dlg)
        btn.pack(fill=tk.X, padx=10, pady=8)

        def on_calibrate():
            adapter = next(a for a in TRANSACTIONAL_ADAPTERS
                           if a.name == adapter_var.get())
            try:
                conv = float(conv_var.get())
            except Exception:
                conv = 0.30
            conv = max(0.01, min(conv, 0.99))
            if report_holder['report'].is_blocking():
                messagebox.showwarning(
                    "Cannot calibrate",
                    "Required columns are missing — see the validation "
                    "report. Fix the source file or pick a different adapter.",
                    parent=dlg)
                return
            result['choice'] = (adapter, conv)
            dlg.destroy()

        def on_cancel():
            result['choice'] = None
            dlg.destroy()

        tk.Button(btn, text="Calibrate & generate layout",
                  command=on_calibrate, bg='#2d7a2d', fg='white',
                  font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        tk.Button(btn, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=8)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        return result['choice']
