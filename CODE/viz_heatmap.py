import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import tkinter as tk
from tkinter import simpledialog, messagebox, colorchooser, ttk, font as tkfont, filedialog
from collections import defaultdict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from drag import DraggableRectangle
from simulation import CustomerFlowSimulation

import copy
from scipy import stats as sp_stats


class HeatmapMixin:
    """Heat-map window, panning, simulation analytics text panels, speed/spawn knobs."""

    def _update_simulation_speed(self, value):
        """Update simulation speed."""
        self.customer_simulation.simulation_speed = float(value)

    def _update_spawn_rate(self, value):
        """Update customer spawn rate.

        Only the rate is written from here. The pending inter-arrival gap
        belongs to the worker thread, which compares it against elapsed time
        every tick and redraws it itself once the rate it was drawn under
        changes; by memorylessness that redraw is exact, so the new rate
        takes effect on the next arrival instead of after the old one."""
        self.customer_simulation.spawn_rate = float(value)

 

    def _get_heatmap_cmap(self):
        """Perceptually uniform colormap suitable for academic papers.

        Returns a custom black -> dark-red -> red -> orange -> yellow -> white
        gradient that mirrors matplotlib's ``inferno`` but with a slightly
        warmer cold end. This keeps hot zones (yellow/white) unambiguously
        distinct from cold zones (black/deep red), unlike the previous
        rainbow palette which compressed mid-traffic into bright greens.
        """
        return LinearSegmentedColormap.from_list(
            'paper_heat_inferno',
            [
                (0.00, '#000004'),  # near-black (cold)
                (0.15, '#1b0c41'),  # deep purple
                (0.30, '#4a0c6b'),  # purple
                (0.45, '#781c6d'),  # magenta-red
                (0.60, '#b73779'),  # red
                (0.72, '#ed6925'),  # orange
                (0.84, '#fcb519'),  # amber / yellow
                (0.94, '#f6e620'),  # bright yellow
                (1.00, '#fcffa4'),  # pale yellow (hot peak)
            ],
            N=1024
        )

    def _prepare_heatmap_display(self, heat, smooth=True):
        """Normalize raw traffic counts to [0, 1] for ``imshow``.

        Pipeline:
          1. Optional light Gaussian smoothing (sigma~1.5) to remove single-cell
             aliasing without erasing structure.
          2. Percentile clipping at the 99th percentile of *positive* cells
             so a handful of extreme peaks do not flatten the rest.
          3. Mild gamma (gamma=0.7) -- much lighter than the previous gamma=0.32 --
             which gives low-traffic cells visible contrast against the
             background without washing out the hot peaks.

        Output is clipped to [0, 1] and matches the perceptually uniform
        colormap returned by ``_get_heatmap_cmap``.
        """
        if heat is None:
            return heat
        arr = np.asarray(heat, dtype=np.float32)
        if arr.size == 0:
            return arr
        positive_max = float(np.max(arr)) if arr.size else 0.0
        if positive_max <= 0:
            return np.zeros_like(arr)

        combined = arr
        if smooth:
            try:
                from scipy import ndimage
                # Light single-pass smoothing -- preserves spatial resolution.
                combined = ndimage.gaussian_filter(arr, sigma=1.5)
            except Exception:
                combined = arr

        # Robust upper bound: 99th percentile of cells that received any
        # traffic. This makes hot zones distinct even when one extreme
        # outlier cell would otherwise dominate the scale.
        positives = combined[combined > 0]
        if positives.size:
            vmax = float(np.percentile(positives, 99))
        else:
            vmax = float(np.max(combined))
        vmax = max(vmax, 1e-6)

        norm = np.clip(combined / vmax, 0.0, 1.0)
        # Mild gamma to lift low-traffic cells off the background without
        # collapsing the dynamic range. gamma=1.0 means linear; gamma<1 brightens.
        norm = np.power(norm, 0.7)
        return norm

    def _show_heat_map(self):
        """
        Open a live, updating heat-map window showing customer traffic.
        The colormap, smoothing, and layout match the original static version.
        """
        # If no data yet, bail
        if not hasattr(self.customer_simulation, 'heat_map_data'):
            messagebox.showinfo("No Data",
                                "No heat map data available yet.",
                                parent=self.tk_root)
            return

        # Close any existing heat-map window cleanly
        if hasattr(self, 'heat_toplevel') and self.heat_toplevel.winfo_exists():
            if hasattr(self, '_heat_after_id'):
                try:
                    self.heat_toplevel.after_cancel(self._heat_after_id)
                except Exception:
                    pass
            self.heat_toplevel.destroy()

        # Create a new Toplevel for the heat map
        top = tk.Toplevel(self.tk_root)
        self.heat_toplevel = top
        top.title("Live Heat Map - Customer Traffic")

        # Build the Figure & Axes
        fig_heat = Figure(figsize=(14, 10), dpi=100)
        fig_heat.patch.set_facecolor('#1a1a1a')
        ax_heat = fig_heat.add_subplot(111)

        cmap_custom = self._get_heatmap_cmap()

        def _current_heat():
            # The heat buffer keeps its allocated shape when the shop shrinks
            # (e.g. importing a smaller layout), so crop it to the current
            # dimensions and place it by the area its cells actually cover
            # instead of stretching the whole buffer over the shop.
            res = self.customer_simulation.heat_map_resolution
            heat = self.customer_simulation.heat_map_data
            heat = heat[:int(self.width * res) + 1, :int(self.height * res) + 1]
            return heat, [0, heat.shape[0] / res, 0, heat.shape[1] / res]

        # Create an empty image to start with. ``aspect='equal'`` is critical
        # for an academic-paper heatmap: it forces 1 m on the x-axis to span
        # the same number of pixels as 1 m on the y-axis, so spatial features
        # (aisles, sections) keep their true geometry.
        heat0, extent0 = _current_heat()
        initial = np.zeros_like(heat0.T)
        im = ax_heat.imshow(
            initial,
            cmap=cmap_custom,
            alpha=1.0,
            extent=extent0,
            origin='lower',
            aspect='equal',
            interpolation='bilinear',
            vmin=0,
            vmax=1
        )
        self._heat_im = im

        # --- Base styling & grid -----------------------------------
        ax_heat.set_xlim(0, self.width)
        ax_heat.set_ylim(0, self.height)
        ax_heat.set_xlabel("Width (m)", color="white")
        ax_heat.set_ylabel("Depth (m)", color="white")
        ax_heat.set_title(
            "Customer Traffic Density (live)",
            color="white", fontsize=12, fontweight='bold', pad=10
        )
        ax_heat.set_facecolor('#000000')
        ax_heat.grid(True, alpha=0.15, color='white', linestyle=':')

        # --- Meter-spaced, high-contrast ticks --------------------
        ax_heat.set_xticks(np.arange(0, self.width  + 1, 1))
        ax_heat.set_yticks(np.arange(0, self.height + 1, 1))
        ax_heat.tick_params(colors="white", labelsize=8)
        for spine in ax_heat.spines.values():
            spine.set_edgecolor("white")
        for lbl in ax_heat.get_xticklabels() + ax_heat.get_yticklabels():
            lbl.set_color("white")
            lbl.set_fontweight("bold")

        # --- Colorbar (replaces the old 5-row legend table) -------
        # Show the *relative* traffic scale (0% = no visits, 100% = peak),
        # which is far more compact and standard for academic figures than
        # the previous categorical table.
        cbar = fig_heat.colorbar(
            self._heat_im, ax=ax_heat, fraction=0.046, pad=0.04
        )
        cbar.set_label(
            "Relative customer traffic density",
            color='white', fontsize=10, fontweight='bold'
        )
        cbar.ax.tick_params(colors='white', labelsize=8)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        cbar.outline.set_edgecolor('white')

        # Tight layout for the academic-paper look.
        try:
            fig_heat.tight_layout()
        except Exception:
            pass

        # Pack into the Tk window
        canvas = FigureCanvasTkAgg(fig_heat, master=top)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # The title-bar close button destroys the window only on the Tcl side,
        # which leaves the pending refresh scheduled; cancel it before
        # destroying so the loop stops with its window.
        def _close_heat():
            try:
                top.after_cancel(self._heat_after_id)
            except Exception:
                pass
            self._heat_after_id = None
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", _close_heat)

        # Define an update function to refresh data & redraw. It is bound to
        # this window and its image, and stops once the window is gone, so a
        # reopened map never ends up driven by two loops.
        def _refresh_heat():
            try:
                if not top.winfo_exists():
                    return
            except Exception:
                return
            heat, extent = _current_heat()
            disp = self._prepare_heatmap_display(heat.T, smooth=True)
            im.set_data(disp)
            im.set_extent(extent)
            im.set_clim(0, 1)
            canvas.draw_idle()
            self._heat_after_id = top.after(750, _refresh_heat)

        _refresh_heat()


    # Helper to refresh the embedded (Customer Flow tab) heatmap image
    def _refresh_embedded_heatmap(self, zero=False):
        """
        Update the embedded heat map image with either zeros or the current sim data.
        Safe to call anytime; no-op if widgets aren't initialized yet.
        """
        if not hasattr(self, 'heat_im') or not hasattr(self, 'heat_canvas') or self.heat_im is None:
            return
        sim = self.customer_simulation
        raw = np.zeros_like(sim.heat_map_data.T) if zero else sim.heat_map_data.T
        data = raw if zero else self._prepare_heatmap_display(raw, smooth=True)
        self.heat_im.set_data(data)
        self.heat_im.set_clim(0, 1)
        try:
            self.heat_canvas.draw_idle()
        except Exception:
            pass


    # Pan handlers: only active when hand_mode is True
    def _pan_press(self, event):
        """Start panning when left mouse button is pressed over the canvas (hand mode only)."""
        if not getattr(self, 'hand_mode', False):
            return
        if event.button != 1 or event.inaxes != self.ax:
            return
        self._panning = True
        self._pan_start_px = (event.x, event.y)          # pixel coords -- stable!
        self._xlim_on_press = self.ax.get_xlim()
        self._ylim_on_press = self.ax.get_ylim()
        self.canvas.get_tk_widget().configure(cursor='fleur')

    def _pan_motion(self, event):
        """Pan the view by dragging the mouse, but never leave the shop bounds."""
        if not getattr(self, 'hand_mode', False):
            return
        if not getattr(self, '_panning', False):
            return

        # Use pixel coords to avoid the data-coord feedback loop
        x0, x1 = self._xlim_on_press
        y0, y1 = self._ylim_on_press
        bbox = self.ax.get_window_extent()

        dx_px = event.x - self._pan_start_px[0]
        dy_px = event.y - self._pan_start_px[1]

        # Convert pixel delta -> data delta (negative = grab-drag)
        dx_data = -dx_px * (x1 - x0) / bbox.width
        dy_data = -dy_px * (y1 - y0) / bbox.height

        # Apply delta to ORIGINAL limits (never the current ones)
        new_x0 = x0 + dx_data
        new_x1 = x1 + dx_data
        new_y0 = y0 + dy_data
        new_y1 = y1 + dy_data

        # Clamp to shop bounds
        view_w = x1 - x0
        view_h = y1 - y0
        if new_x0 < 0:
            new_x0, new_x1 = 0, view_w
        if new_x1 > self.width:
            new_x0, new_x1 = self.width - view_w, self.width
        if new_y0 < 0:
            new_y0, new_y1 = 0, view_h
        if new_y1 > self.height:
            new_y0, new_y1 = self.height - view_h, self.height

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _pan_release(self, event):
        """Stop panning when the mouse button is released."""
        if not getattr(self, 'hand_mode', False):
            return
        self._panning = False
        # Restore hand cursor
        self.canvas.get_tk_widget().configure(cursor='hand2')



    #  (throttle when tab not visible):
    def _update_simulation_analytics(self):
        """
        Refresh both analytics panes and the embedded heat-map every 2 seconds.
        Left pane: SALES, IMPULSE, CUSTOMER BEHAVIOR.
        Right pane: PRODUCT PERFORMANCE and others.
        Skip heavy string building if the Simulation tab isn't visible.

        """



        # Guard against widget destruction  
        try:  
            if not self.tk_root.winfo_exists():  
                return  
        except Exception:  
                return




        # Cancel any pending run first so a repeated call cannot start a
        # second chain whose id the close handler never sees.
        if getattr(self, '_analytics_after_id', None):
            try:
                self.tk_root.after_cancel(self._analytics_after_id)
            except Exception:
                pass

        # Always reschedule -- but remember the id so the root close handler
        # can cancel it. Otherwise the callback fires after tk_root is
        # destroyed and Tk logs "invalid command name ... _update_simulation_analytics".
        if self.tk_root:
            try:
                self._analytics_after_id = self.tk_root.after(
                    2000, self._update_simulation_analytics
                )
            except Exception:
                self._analytics_after_id = None

        # Labels not ready
        if not self.simulation_analytics_label or not self.product_metrics_label:
            return

        # If Simulation tab not visible, skip heavy summary building
        if getattr(self, "current_tab", "") != "Customer Flow":
            return

        # Build full analytics summary
        full_lines = self.customer_simulation.get_analytics_summary().splitlines()

        # Split where PRODUCT PERFORMANCE begins
        split_idx = next(
            (i for i, line in enumerate(full_lines)
            if line.startswith("🎯 PRODUCT PERFORMANCE")),
            len(full_lines)
        )

        left_text  = "\n".join(full_lines[:split_idx])
        right_text = "\n".join(full_lines[split_idx:])

        self.simulation_analytics_label.config(text=left_text)
        self.product_metrics_label.config(text=right_text)

        # Update embedded heat-map image if present
        if hasattr(self, 'heat_im') and hasattr(self, 'heat_canvas'):
            heat_data = self._prepare_heatmap_display(self.customer_simulation.heat_map_data.T, smooth=True)
            self.heat_im.set_data(heat_data)
            self.heat_im.set_clim(0, 1)
            self.heat_canvas.draw_idle()

