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


class LayoutMixin:
    """Layout redraw / save / import / clear and patch movement handlers."""

    def redraw(self):
        """
        Redraw the entire shop layout.

        Drawing order / z-index:
         1. Background + shop border  (zorder 0-2)
         2. Walls & sections          (zorder 3)
         3. Items                     (zorder 4)
         4. Text annotations          (zorder+1 of parent)
        Automatically schedules update_metrics() afterwards.
        """

        # Ensure figure & axes exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()

        # Clear previous drawings
        self._remove_resize_handles()
        self.ax.clear()
        self.annotation_patches.clear()
        self.draggables.clear()
        self.wall_patches.clear()
        self.item_patches.clear()

        # -- Background (no grid) --
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_facecolor('#16213e')

        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Width (m)', color='#7f8c8d', fontsize=9)
        self.ax.set_ylabel('Depth (m)', color='#7f8c8d', fontsize=9)
        self.ax.set_title('')
        self.ax.tick_params(colors='#7f8c8d', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#2a3a5e')
            spine.set_linewidth(0.5)

        border = plt.Rectangle((0, 0), self.width, self.height,
                    fill=False, edgecolor='#e94560', linewidth=1.5,
                    linestyle='-', zorder=2)
        self.ax.add_patch(border)

        # -- Draw walls & sections (zorder=3) --
        for nm, w in self.walls.items():
            x, y   = w['position']
            wm, hm = w['size']
            col    = w.get('color', 'gray')

            patch = plt.Rectangle((x, y), wm, hm,
                    facecolor=col, edgecolor='black',
                    alpha=0.5, zorder=3)
            is_section = nm.startswith("Section_")
            if is_section:
                patch.is_section = True
            patch.set_picker(True)
            self.ax.add_patch(patch)

            # fetch text attributes
            text_color = w.get('textcolor', 'black') if is_section else w.get('textcolor', 'white')
            text_size  = min(w.get('fontsize', 14) * self.zoom, 48)
            text_font  = w.get('font', None)
            if text_font in (None, '', 'System'):
                text_font = None

            if is_section:
                # section title
                sec_label = nm[len("Section_"):]
                loc = w.get('label_loc', 'bottom')
                if loc == 'top':
                    y_text, va = y + hm - 0.02*hm, 'top'
                elif loc == 'bottom':
                    y_text, va = y + 0.02*hm, 'bottom'
                else:
                    y_text, va = y + hm/2, 'center'
                txt = self.ax.text(
                    x + wm/2, y_text, sec_label,
                    ha='center', va=va,
                    color=text_color,
                    fontsize=text_size,
                    fontfamily=text_font,
                    fontweight='bold',
                    zorder=patch.get_zorder() + 2,
                    clip_on=True
                )
            else:
                # non-section wall label
                if hm > wm:
                    txt = self.ax.text(
                        x + wm/2, y + hm/2, nm,
                        ha='center', va='center',
                        color=text_color,
                        fontsize=text_size,
                        fontfamily=text_font,
                        rotation=90,
                        rotation_mode='anchor',
                        clip_on=True
                    )
                else:
                    txt = self.ax.text(
                        x + wm/2, y + hm/2, nm,
                        ha='center', va='center',
                        color=text_color,
                        fontsize=text_size,
                        fontfamily=text_font,
                        clip_on=True
                    )

            self.wall_patches[nm]       = patch
            self.annotation_patches[patch] = txt
            dr = DraggableRectangle(patch,
                on_update=self._on_patch_moved,
                owner=self)
            self.draggables[patch] = dr

        # -- Draw items on top (zorder=4) --
        for nm, it in self.items.items():
            x, y   = it['position']
            wm, hm = it['size']
            col    = it.get('color') or {
                'food':'green','electronics':'blue','clothing':'red'
            }.get(it['category'], 'gray')

            patch = plt.Rectangle((x, y), wm, hm,
                facecolor=col, edgecolor='black',
                alpha=0.8, zorder=4)
            patch.set_picker(True)
            self.ax.add_patch(patch)

            # fetch item-text attributes
            text_color = it.get('textcolor', 'white')
            text_size  = min(it.get('fontsize', 10) * self.zoom, 48)
            text_font  = it.get('font', None)

            if hm > wm:
                txt = self.ax.text(
                    x + wm/2, y + hm/2, nm,
                    ha='center', va='center',
                    color=text_color,
                    fontsize=text_size,
                    fontfamily=text_font,
                    rotation=90,
                    rotation_mode='anchor',
                    zorder=patch.get_zorder()+1,
                    clip_on=True
                )
            else:
                txt = self.ax.text(
                    x + wm/2, y + hm/2, nm,
                    ha='center', va='center',
                    color=text_color,
                    fontsize=text_size,
                    fontfamily=text_font,
                    zorder=patch.get_zorder()+1,
                    clip_on=True
                )

            self.item_patches[nm]       = patch
            self.annotation_patches[patch] = txt
            dr = DraggableRectangle(patch,
                on_update=self._on_patch_moved,
                owner=self)
            self.draggables[patch] = dr

        # -- Refresh canvas & update metrics --
        if self.canvas:
            self.canvas.draw_idle()

        # we cleared axes above; ensure customer markers are rebuilt lazily
        if hasattr(self, 'customer_simulation') and hasattr(self.customer_simulation, 'customer_patches_by_id'):
            self.customer_simulation.customer_patches_by_id.clear()

        # mark metrics dirty; compute only if metrics tab is active
        self._metrics_dirty = True
        if self.current_tab == "Shop Area Metrics":
            self.update_metrics()



    def _on_define_simulation_requirements(self):
        """
        Open a single dialog with buttons to define:
         - Entrance
         - Checkout
         - WC (restroom) area
         - Manual impulse-item placement
        """

        dlg = tk.Toplevel(self.tk_root)
        dlg.transient(self.tk_root)
        dlg.title("Define Simulation Requirements")

        btn_frame = tk.Frame(dlg)
        btn_frame.pack(padx=20, pady=15)

        tk.Button(
            btn_frame, text="Entrance", width=20,
            command=lambda: [self._on_add_entrance_clicked(), dlg.lift()]
        ).pack(pady=4)

        tk.Button(
            btn_frame, text="Checkout", width=20,
            command=lambda: [self._on_add_checkout_clicked(), dlg.lift()]
        ).pack(pady=4)

        tk.Button(
            btn_frame, text="WC Area", width=20,
            command=lambda: [self._on_add_wc_clicked(), dlg.lift()]
        ).pack(pady=4)

        tk.Button(
            btn_frame, text="Impulse Items", width=20,
            command=lambda: [self._on_add_impulse_items_clicked(), dlg.lift()]
        ).pack(pady=4)

        tk.Button(dlg, text="Done", width=10, command=dlg.destroy) \
          .pack(pady=(10,5))

        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x = (dlg.winfo_screenwidth() // 2) - (w // 2)
        y = (dlg.winfo_screenheight() // 2) - (h // 2)
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.grab_set()
        self.tk_root.wait_window(dlg)

    

      
   
    # (added geometry cache invalidation):
    def _on_patch_moved(self, patch):
        """
        Callback when a patch is dragged: update model and reposition its text label.
        Sections always keep their label at the top; other objects stay centered. 
        Sync a moved patch's new xy back into self.items or self.walls, and reposition its label.
        """
        # 1) Sync the moved patch position back into items or walls dict
        is_section = False
        for nm, p in self.item_patches.items():
            if p is patch:
                self.items[nm]["position"] = patch.get_xy()
                break
        for nm, p in self.wall_patches.items():
            if p is patch:
                self.walls[nm]["position"] = patch.get_xy()
                if nm.startswith("Section_"):
                    is_section = True
                break

        # 2) Reposition the annotation based on object type
        ann = self.annotation_patches.get(patch)
        if ann:
            x, y = patch.get_xy()
            w, h = patch.get_width(), patch.get_height()
            if is_section:
                # Always place section-label at top edge
                y_text = y + h - 0.02 * h
                ann.set_position((x + w/2, y_text))
                ann.set_va('top')
            else:
                # Center for items and non-section walls
                ann.set_position((x + w/2, y + h/2))
                ann.set_va('center')

        # Invalidate metrics and simulation geometry caches
        self._metrics_dirty = True
        if hasattr(self, 'customer_simulation'):
            self.customer_simulation._zones_cache = None
            self.customer_simulation.geometry_dirty = True

    def export_layout(self):
        """
        Export the current shop layout--including all floors, connectors,
        entrance & checkout--to JSON.
        """
        data = {
            "shop_dimensions": {"width": self.width, "height": self.height},
            "door": {
                "position": getattr(self, 'door_position', None),
                "side":     getattr(self, 'door_side', None),
            },
            "num_floors":     getattr(self, 'num_floors', 1),
            "current_floor":  getattr(self, 'current_floor', 1),
            # floors keyed as strings for JSON; each floor stores its own items/walls/prices
            "floors": {
                str(fid): {
                    "items":  f.get("items", {}),
                    "walls":  f.get("walls", {}),
                    "prices": f.get("prices", {}),
                    "is_main_floor": bool(f.get("is_main_floor", fid == 1)),
                    "door_position": f.get("door_position"),
                    "door_side": f.get("door_side"),
                }
                for fid, f in getattr(self, 'floors', {1: {
                    "items": self.items, "walls": self.walls, "prices": self.prices
                }}).items()
            },
            # Serialize connector floor-ID keys as strings for JSON
            "connectors": {
                cid: {str(fid): wname for fid, wname in mapping.items()}
                for cid, mapping in getattr(self, 'connectors', {}).items()
            },
        }

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files","*.json"),("All files","*.*")],
            title="Export Shop Layout",
            parent=self.tk_root
        )
        if not filename:
            return

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Export Successful",
                f"Layout exported to {filename}", parent=self.tk_root)
        except Exception as e:
            messagebox.showerror("Export Failed",
                f"Error exporting layout: {e}", parent=self.tk_root)

    def import_layout(self):
        """
        Import shop layout (multi-floor aware) from JSON.
        Falls back gracefully to legacy single-floor files.
        """
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files","*.json"),("All files","*.*")],
            title="Import Shop Layout",
            parent=self.tk_root
        )
        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if "shop_dimensions" not in data:
                raise ValueError("Missing 'shop_dimensions' in layout file")

            dims = data["shop_dimensions"]
            new_w, new_h = dims["width"], dims["height"]
            if (new_w, new_h) != (self.width, self.height):
                if messagebox.askyesno("Different Dimensions",
                    f"Current: {self.width}×{self.height}\nImport: {new_w}×{new_h}\nContinue?",
                    parent=self.tk_root):
                    self.width, self.height = new_w, new_h
                else:
                    return

            # Multi-floor format
            if "floors" in data:
                self.floors = {}
                for fid_str, fdata in data["floors"].items():
                    fid = int(fid_str)
                    self.floors[fid] = {
                        "items":  fdata.get("items",  {}),
                        "walls":  fdata.get("walls",  {}),
                        "prices": fdata.get("prices", {}),
                        "is_main_floor": bool(fdata.get("is_main_floor", fid == 1)),
                        "door_position": tuple(fdata["door_position"]) if fdata.get("door_position") else None,
                        "door_side": fdata.get("door_side"),
                    }
                self.num_floors = len(self.floors)
                # Restore connector floor-ID keys from JSON strings back to ints
                raw_connectors = data.get("connectors", {})
                self.connectors = {}
                for cid, mapping in raw_connectors.items():
                    self.connectors[cid] = {
                        int(k): v for k, v in mapping.items()
                    }
                self.current_floor = int(data.get("current_floor", 1))
            else:
                # Legacy single-floor file
                if "items" not in data or "walls" not in data:
                    raise ValueError("Missing 'items' or 'walls' in layout file")
                self.floors = {1: {
                    "items":  data["items"],
                    "walls":  data["walls"],
                    "prices": {},
                    "is_main_floor": True,
                    "door_position": None,
                    "door_side": None,
                }}
                self.num_floors    = 1
                self.connectors    = {}
                self.current_floor = 1

            self._assign_default_prices()
            for name in self.items:
                if name not in self.prices:
                    self.prices[name] = round(np.random.uniform(5.0, 50.0), 2)

            door = data.get("door", {})
            pos = door.get("position")
            floor1 = self.floors.get(1, {})
            floor_door = floor1.get("door_position")
            self.door_position = tuple(floor_door) if floor_door else (tuple(pos) if pos else None)
            self.door_side     = floor1.get("door_side") or door.get("side", None)
            if 1 in self.floors:
                self.floors[1]['is_main_floor'] = True
                self.floors[1]['door_position'] = self.door_position
                self.floors[1]['door_side'] = self.door_side

            if hasattr(self, 'customer_simulation'):
                self.customer_simulation.door_position = self.door_position
                self.customer_simulation.door_side     = self.door_side

            if hasattr(self, '_floor_label') and self._floor_label is not None:
                self._floor_label.config(text=f"F{self.current_floor}/{self.num_floors}")

            self.redraw()
            messagebox.showinfo("Import Successful",
                "Layout imported successfully.", parent=self.tk_root)

        except Exception as e:
            messagebox.showerror("Import Failed",
                f"Error importing layout: {e}", parent=self.tk_root)




    def _clear_current_data(self):
        """Reset all simulation analytics and heat-map data to zero."""
        sim = self.customer_simulation

        # Reinitialize analytics to fresh defaults, including customers_over_time
        sim.analytics = {
            'total_customers':            0,
            'average_time_in_shop':       0,
            'popular_items':              defaultdict(int),
            'area_visits':                defaultdict(int),
            'exit_traffic_by_minute':     defaultdict(int),
            'customers_over_time':        defaultdict(int),
            'total_revenue':              0.0,
            'completed_purchases':        0,
            'abandoned_carts':            0,
            'basket_sizes':               [],
            'impulse_purchases':          0,
            'impulse_item_sales':         defaultdict(int),
            'revenue_by_area':            defaultdict(float),
            'dwell_times_by_zone':        defaultdict(list),
            'customer_paths':             [],
            'return_customers':           0,
            'customer_lifetime_values':   defaultdict(float),
            'cross_merchandising':        defaultdict(int),
            'item_conversion_rates':      defaultdict(lambda: {'visits':0,'purchases':0}),
            'foot_traffic_density':       defaultdict(float),
            'bottlenecks':                defaultdict(int),
            'queue_wait_times':           [],
            'processing_times':           [],
            'data_points_collected':      0,
            'accuracy_metrics':           [],
            'optimization_history':       [],
            'pre_optimization_revenue':   0.0,
            'post_optimization_revenue':  0.0,
            'optimization_impact':        0.0
        }

        # Reset bottleneck threshold on the simulation (previously set on self only)
        self.bottleneck_threshold = 50       # keep for backward compatibility
        sim.bottleneck_threshold  = 50       # actually used by the simulation

        # Fully reset heat map buffers and timers on the simulation.
        # Operate on ``sim`` (the simulation), not ``self`` (the
        # visualizer) -- the visualizer doesn't own analytics/run_time/
        # prev_state_by_cust, so the previous version silently no-op'd
        # via the bare except.
        try:
            self.reset_heatmap()
            if 'foot_traffic_density' in sim.analytics:
                sim.analytics['foot_traffic_density'].clear()
            if 'bottlenecks' in sim.analytics:
                sim.analytics['bottlenecks'].clear()
            sim.prev_state_by_cust.clear()
        except Exception:
            pass

        # Reset the simulation runtime timer back to 0
        sim.run_time = 0.0
        sim.simulation_start_time = time.time()

        # Close any existing live heat map window and cancel its scheduled updates
        try:
            if hasattr(self, 'heat_toplevel') and self.heat_toplevel.winfo_exists():
                if hasattr(self, '_heat_after_id'):
                    try:
                        self.heat_toplevel.after_cancel(self._heat_after_id)
                    except Exception:
                        pass
                self.heat_toplevel.destroy()
        except Exception:
            pass

        # Immediately refresh the embedded heat map (Customer Flow tab) to show zeros
        if hasattr(self, 'heat_im') and hasattr(self, 'heat_canvas'):
            try:
                zero = np.zeros_like(sim.heat_map_data.T)
                self.heat_im.set_data(zero)
                self.heat_im.set_clim(0, 1)  # normalize to zeroed map
                self.heat_canvas.draw_idle()
            except Exception:
                pass

        messagebox.showinfo(
            "Data Cleared",
            "All simulation metrics and heat‐map data have been reset.",
            parent=self.tk_root
        )


  

