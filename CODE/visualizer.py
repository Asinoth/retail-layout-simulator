import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os  # might need later

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

from viz_projections import ProjectionsMixin
from viz_sensitivity import SensitivityMixin
from viz_markov import MarkovMixin
from viz_ga import GAMixin
from viz_ga_run import GARunMixin
from viz_whatif import WhatIfMixin
from viz_dataset import DatasetMixin
from viz_layout import LayoutMixin
from viz_edit import EditMixin
from viz_events import EventsMixin
from viz_addops import AddOpsMixin
from viz_optimize import OptimizeMixin
from viz_optimize_results import OptimizeResultsMixin
from viz_optimize_helpers import OptimizeHelpersMixin
from viz_metrics import MetricsMixin
from viz_generate import GenerateMixin
from viz_simtab import SimTabMixin
from viz_heatmap import HeatmapMixin
from viz_validation import ValidationMixin


# Helper module to draw analytics in json form
def export_analytics_to_json(shop, path="analytics_latest.json"):
    """
    Write the current analytics dict to a JSON file, sanitizing non-JSON-compatible data.
    Call with the ShopVisualizer instance (e.g., export_analytics_to_json(self)).
    """
    analytics = getattr(shop.customer_simulation, "analytics", None)
    if isinstance(analytics, dict) == False:
        print("No analytics to write.")
        return

    # The module-level sanitize converts numpy scalars via .item(); without
    # that, np.int64 / np.bool_ values would be written as strings.
    safe = sanitize(analytics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    print(f"Wrote {path}")

# Convert tuple keys to strings, numpy types to Python types, sets/tuples to lists, and defaultdicts to plain dicts.
def sanitize(obj):
        # Plain leaves are the bulk of a long run's analytics (hundreds of
        # thousands of floats in the per-frame lists) and the json.dumps probe
        # at the bottom is the expensive way to learn they are fine. type() is
        # an exact match, so numpy scalars still take the np.generic branch.
        if obj is None or type(obj) in (str, bool, int, float):
            return obj

        # Dict-like: sanitize keys and values
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # Fix problematic keys (e.g., tuples -> "a|b")
                if isinstance(k, tuple):
                    key = "|".join(map(str, k))
                elif k is None or isinstance(k, (str, int, float, bool)):
                    key = k
                else:
                    key = str(k)
                out[str(key)] = sanitize(v)
            return out

        # Sequences and sets: convert to lists and sanitize elements
        if isinstance(obj, (list, tuple, set)):
            return [sanitize(x) for x in obj]

        # Numpy scalars/arrays: convert to native/list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()

        # Fallback: ensure json-serializable or stringify
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)



# TODO: split this into multiple files eventually
class ShopVisualizer(
    ProjectionsMixin,
    SensitivityMixin,
    MarkovMixin,
    GAMixin,
    GARunMixin,
    WhatIfMixin,
    DatasetMixin,
    SimTabMixin,
    LayoutMixin,
    EditMixin,
    EventsMixin,
    AddOpsMixin,
    OptimizeMixin,
    OptimizeResultsMixin,
    OptimizeHelpersMixin,
    MetricsMixin,
    GenerateMixin,
    HeatmapMixin,
    ValidationMixin,
):
    """
    Main application for 2D shop layout visualization and editing.
    
    This class provides a complete interface for creating, editing, and analyzing
    shop floor plans. It manages items, walls, user interactions, and metrics
    calculations.

     Responsibilities:
     - Building the Tk notebook (Layout / Metrics / Simulation tabs)
     - Redrawing the background grid, walls, sections, items
     - Handling user commands: add/remove/rename/resize items & walls
     - Defining entrance, checkout and WC areas by picking existing items
     - Generating and optimizing full layouts automatically
     - Showing real-time area metrics and customer flow analytics
    
    Attributes:
        width (float): Shop width in meters
        height (float): Shop height in meters
        items (dict): Dictionary of shop items with properties
        walls (dict): Dictionary of walls and sections with properties
        fig (matplotlib.figure.Figure): Matplotlib figure for drawing
        ax (matplotlib.axes.Axes): Matplotlib axes for the shop layout
        item_patches (dict): Maps item names to their matplotlib patches
        wall_patches (dict): Maps wall names to their matplotlib patches
        annotation_patches (dict): Maps patches to their text annotations
        draggables (dict): Maps patches to their DraggableRectangle instances
        resize_handles (list): List of active resize handle DraggableRectangles
        selected_patch (matplotlib.patches.Rectangle): Currently selected patch
        tk_root (tk.Tk): Tkinter root window
        canvas (FigureCanvasTkAgg): Matplotlib canvas for Tkinter
        zoom (float): Current zoom level
    """
    def __init__(self, shop_dimensions_m):
        self.width,self.height = shop_dimensions_m

        # --- Multi-floor data model (MUST come before anything that uses items/walls/prices) ---
         # --- Multi-floor data model ---
        # --- Multi-floor data model ---
        self.floors = {
            1: {
                'items': {},
                'walls': {},
                'prices': {},
                'is_main_floor': True,
                'door_position': None,
                'door_side': None,
            }
        }

        self.current_floor = 1
        self.num_floors = 1
        self.connectors = {}
        #NOTE: do NOT assign self.items / self.walls / self.prices here --
        # they are properties that proxy into self.floors[self.current_floor].


        self._floor_label = None
        self.fig = None
        self.ax = None

        self.item_patches = {}
        self.wall_patches = {}
        self.annotation_patches = {}
        self.draggables = {}
        self.resize_handles = []
        self.selected_patch = None
        self._dimension_annotations = []

        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        self.canvas = None
        self.zoom_label = None
        self.metrics_label = None
        self.customer_simulation = CustomerFlowSimulation(self)
        self.simulation_controls = None
        self.simulation_analytics_label = None
        self.door_position = None
        self.door_side = None

        self.zoom = 1.0

        self._last_metrics_time = 0.0
        self._metrics_interval = 1.0
        self._metrics_after_id = None
        self._layout_variation = -1

        self.current_tab = "Layout"

        self._analytics_after_id = None

        self._is_fullscreen = False
        self._resize_after_id = None

        self.hand_mode = False
        self.hand_mode_var = None
        self.hand_btn = None

        self.snap_to_grid = True
        self._snap_size = 0.25
        self._snap_var = None

        self._status_bar = None
        self._cursor_pos_label = None
        self._selection_info_label = None



        # (Multi-floor data model already initialized above.)

    # -- Per-floor proxy properties so existing mixin code keeps working --
    @property
    def items(self):
        return self.floors[self.current_floor]['items']

    @items.setter
    def items(self, value):
        self.floors[self.current_floor]['items'] = value

    @property
    def walls(self):
        return self.floors[self.current_floor]['walls']

    @walls.setter
    def walls(self, value):
        self.floors[self.current_floor]['walls'] = value

    @property
    def prices(self):
        return self.floors[self.current_floor]['prices']

    @prices.setter
    def prices(self, value):
        self.floors[self.current_floor]['prices'] = value

    # -- Floor management helpers ------------------------------------
    def add_floor(self):
        new_id = max(self.floors.keys()) + 1
        self.floors[new_id] = {
            'items': {},
            'walls': {},
            'prices': {},
            'is_main_floor': False,
            'door_position': None,
            'door_side': None,
        }
        self.num_floors = len(self.floors)
        return new_id

    def remove_floor(self, floor_id):
        if floor_id == 1:
            messagebox.showerror("Cannot remove", "Floor 1 (ground) is required.")
            return
        if floor_id in self.floors:
            # detach any connectors referencing this floor
            orphaned_walls = []  # (floor_id, wall_name) tuples to remove
            for cid, mapping in list(self.connectors.items()):
                # Collect wall names on deleted floor for removal
                if floor_id in mapping:
                    orphaned_walls.append((floor_id, mapping[floor_id]))
                mapping.pop(floor_id, None)
                if len(mapping) < 2:
                    # This connector is now broken; remove wall from remaining floor
                    for fid, wname in mapping.items():
                        orphaned_walls.append((fid, wname))
                    self.connectors.pop(cid, None)
            # Remove orphaned connector walls from floor data
            for fid, wname in orphaned_walls:
                if fid in self.floors and wname in self.floors[fid].get('walls', {}):
                    del self.floors[fid]['walls'][wname]
            del self.floors[floor_id]
            self.num_floors = len(self.floors)
            if self.current_floor == floor_id:
                self.current_floor = 1
                # The heat buffers still hold the deleted floor's traffic and
                # only the running loop refreshes them, so a paused or stopped
                # simulation would keep showing it as floor 1.
                heat_sim = getattr(self, 'customer_simulation', None)
                if heat_sim is not None:
                    try: heat_sim.switch_floor_heatmap(1)
                    except Exception: pass
                    try: self._refresh_embedded_heatmap()
                    except Exception: pass
            # Move any customers on the deleted floor back to floor 1
            sim = getattr(self, 'customer_simulation', None)
            if sim:
                for cust in getattr(sim, 'customers', []):
                    if getattr(cust, 'floor', 1) == floor_id:
                        cust.floor = 1
                        cust.target_floor = 1
                        cust._pending_connector = None
                        # Reposition near entrance
                        entrance = sim._find_entrance_position()
                        cust.position = list(entrance)
                # Rebuild reachable floors for all customers
                try:
                    reach = sim._reachable_floors(start_floor=1)
                    for cust in sim.customers:
                        cust._reachable_floors = set(reach)
                except Exception:
                    pass
                sim.geometry_dirty = True
                # The deleted floor's walls, sections and orphaned
                # connectors leave with it, and the zone table may have
                # been built from them.
                try: sim.invalidate_zones_cache()
                except Exception: pass
            # Clear per-floor heatmap data for deleted floor
            if sim and hasattr(sim, '_floor_heat_raw'):
                sim._floor_heat_raw.pop(floor_id, None)
            # Update floor label
            if getattr(self, '_floor_label', None) is not None:
                try:
                    self._floor_label.config(text=f"F{self.current_floor}/{self.num_floors}")
                except Exception:
                    pass
            self.redraw()

    def switch_floor(self, floor_id):
        """Switch the visible floor. Cleans up customer patches, rebuilds
        the zone cache for analytics, and swaps the heatmap to that floor."""
        if floor_id not in self.floors:
            return
        if floor_id == self.current_floor:
            return

        self.current_floor = floor_id

        sim = getattr(self, 'customer_simulation', None)
        if sim is not None:
            if hasattr(sim, 'customer_patches_by_id'):
                for pid, patch in list(sim.customer_patches_by_id.items()):
                    try: patch.remove()
                    except Exception: pass
                sim.customer_patches_by_id.clear()
            sim.geometry_dirty = True
            # Floor-aware zone cache must be rebuilt for the new floor.
            try: sim.invalidate_zones_cache()
            except Exception: pass
            # Show this floor's heatmap immediately.
            try: sim.switch_floor_heatmap(floor_id)
            except Exception: pass

        if getattr(self, '_floor_label', None) is not None:
            try:
                self._floor_label.config(text=f"F{self.current_floor}/{self.num_floors}")
            except Exception:
                pass

        try: self.redraw()
        except Exception: pass



    def all_items_across_floors(self):
        """Flatten items from every floor with duplicate-safe keys.

        Floors are separate dictionaries, so two floors can legitimately have an
        item with the same display name.  Customer routing and optimization need
        a unique key, therefore later duplicates are exposed as ``F<floor>:name``
        while preserving the original display name in ``source_name``.
        """
        out = {}
        for fid, f in sorted(self.floors.items()):
            for name, data in f.get('items', {}).items():
                d = dict(data)
                d['floor'] = fid
                d.setdefault('source_name', name)
                key = name
                if key in out:
                    key = f"F{fid}:{name}"
                    suffix = 2
                    while key in out:
                        key = f"F{fid}:{name}_{suffix}"
                        suffix += 1
                out[key] = d
        return out



    def _assign_default_prices(self):
        """
        Walk the items on every floor and give each one a default price if it
        has none. This covers manual-add, import, and generate paths.
        self.items / self.prices only reach the floor being viewed, so the
        floors are iterated directly; otherwise unpriced items elsewhere earn
        0.0 in live revenue.
        """
        for fdata in self.floors.values():
            prices = fdata.setdefault('prices', {})
            for name in fdata.get('items', {}):
                if name not in prices:
                    # you can tweak this range or even base it on category
                    prices[name] = round(np.random.uniform(5.0, 50.0), 2)

 
    def _on_tab_changed(self, event):
        """Handle Notebook tab switch; store current_tab name for conditional redraw."""
        nb = event.widget
        sel = nb.select()
        self.current_tab = nb.tab(sel, "text")
        # Recompute metrics only when the metrics tab is visible
        if self.current_tab == "Shop Area Metrics":
            self.update_metrics()



    def show(self):
        """
        Build all Tk controls, the matplotlib canvas, then enter mainloop().
        Layout Tab, Metrics Tab and Customer Flow Tab are all constructed here.
        Construct the Notebook tabs (Layout, Metrics, Simulation), pack controls and canvas, then mainloop().
        """
        # reveal and title main window
        # grid or pack Notebook with expand=True so it fills max window space
        # Layout tab: button row (Add Item/Wall/.../Optimize) + zoom + import/export
        #    then embed FigureCanvasTkAgg and bind pan/pick/motion events
        # Metrics tab: full-frame Label to hold area metrics
        # call redraw() once at the end before entering mainloop()




        # 1) Reveal the main window and set title
        self.tk_root.deiconify()
        self.tk_root.title("2D Visualizer")
        
        

        # 2) Create the Notebook with three tabs: Layout, Metrics, Simulation
        notebook = ttk.Notebook(self.tk_root)
        tab_layout  = ttk.Frame(notebook)                   # Layout editor
        tab_metrics = tk.Frame(notebook, bg='#002747')      # Area metrics
        notebook.add(tab_layout,  text='Layout')
        notebook.add(tab_metrics, text="Shop Area Metrics")
        self._create_simulation_tab(notebook)                # Customer Flow tab
        self._create_projections_tab(notebook)
        self._create_sensitivity_tab(notebook)
        self._create_markov_tab(notebook)
        self._create_ga_tab(notebook)
        self._create_whatif_tab(notebook)
        self._create_validation_tab(notebook)
        notebook.pack(fill=tk.BOTH, expand=True)
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

                # --- Layout Tab Controls ---------------------------------------------
        ctrl = tk.Frame(tab_layout, bg='#1a1a2e')
        ctrl.pack(side=tk.TOP, fill=tk.X)

        btn_style = dict(
            bg='#16213e', fg='#e0e0e0', activebackground='#0f3460',
            activeforeground='white', relief=tk.FLAT, bd=0,
            font=('Segoe UI', 9), padx=10, pady=4, cursor='hand2'
        )
        sep_style = dict(bg='#0f3460', width=1)

        tk.Button(ctrl, text="+ Item", command=self._on_add_item_clicked, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(ctrl, text="+ Wall", command=self._on_add_wall_clicked, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(ctrl, text="+ Section", command=self._on_add_section_clicked, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)

        tk.Frame(ctrl, **sep_style).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=4)

        tk.Button(ctrl, text="Sim Requirements", command=self._on_define_simulation_requirements, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(ctrl, text="Generate", command=self._on_generate_layout, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(ctrl, text="Optimize",
                  command=self._optimize_layout,
                  bg='#e94560', fg='white', activebackground='#c81e45',
                  activeforeground='white', relief=tk.FLAT, bd=0,
                  font=('Segoe UI', 9, 'bold'), padx=12, pady=4, cursor='hand2'
                  ).pack(side=tk.LEFT, padx=2, pady=3)

 

        tk.Frame(ctrl, **sep_style).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=4)

        self.hand_mode_var = tk.BooleanVar(value=False)
        self.hand_btn = tk.Checkbutton(
            ctrl, text="Pan", variable=self.hand_mode_var,
            indicatoron=False, command=self._toggle_hand_mode,
            bg='#16213e', fg='#e0e0e0', selectcolor='#0f3460',
            activebackground='#0f3460', relief=tk.FLAT, bd=0,
            font=('Segoe UI', 9), padx=8, pady=4, cursor='hand2'
        )
        self.hand_btn.pack(side=tk.LEFT, padx=2, pady=3)

        self._snap_var = tk.BooleanVar(value=True)
        snap_btn = tk.Checkbutton(
            ctrl, text="Snap", variable=self._snap_var,
            indicatoron=False, command=self._toggle_snap,
            bg='#16213e', fg='#e0e0e0', selectcolor='#0f3460',
            activebackground='#0f3460',relief=tk.FLAT, bd=0,
            font=('Segoe UI', 9), padx=8, pady=4, cursor='hand2'
        )
        snap_btn.pack(side=tk.LEFT, padx=2, pady=3)

        self.zoom_label = tk.Label(ctrl, text=f"{int(self.zoom*100)}%", bg='#1a1a2e', fg='#7f8c8d', font=('Segoe UI', 9), cursor='hand2')
        self.zoom_label.pack(side=tk.LEFT, padx=8)
        self.zoom_label.bind('<Button-1>', lambda e: self._on_zoom_label_clicked())

        tk.Frame(ctrl, **sep_style).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=4)

        tk.Button(ctrl, text='Import', command=self.import_layout, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(ctrl, text='Export', command=self.export_layout, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(ctrl, text="Dataset", command=self._on_dataset_button, **btn_style).pack(side=tk.LEFT, padx=2, pady=3)

        # --- Status Bar ------------------------------------------------------
        self._status_bar = tk.Frame(tab_layout, bg='#1a1a2e', height=22)
        self._status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._cursor_pos_label = tk.Label(self._status_bar, text="X: — Y: —",
                                           bg='#1a1a2e', fg='#7f8c8d',
                                           font=('Consolas', 8), anchor='w')
        self._cursor_pos_label.pack(side=tk.LEFT, padx=8)
        self._selection_info_label = tk.Label(self._status_bar, text="",
                                              bg='#1a1a2e', fg='#7f8c8d',
                                              font=('Consolas', 8), anchor='w')
        self._selection_info_label.pack(side=tk.LEFT, padx=16)

                       # --- Layout Tab Canvas -------------------------------------------------
        if self.fig is None or self.ax is None:
            # A plain Figure, not pyplot: under TkAgg a pyplot figure creates
            # its own hidden Tk root, which keeps mainloop() running after the
            # main window is closed.
            self.fig = Figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot()

        # --- Floor selector side panel (RIGHT of the layout) ------------
        floor_panel = tk.Frame(tab_layout, bg='#0f3460', width=60)
        floor_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 0))
        floor_panel.pack_propagate(False)   # keep its fixed width

        # Spacer so buttons sit roughly vertically centered next to layout
        tk.Frame(floor_panel, bg='#0f3460').pack(expand=True)

        ovl_btn_style = dict(
            bg='#16213e', fg='#e0e0e0',
            activebackground='#0f3460', activeforeground='white',
            relief=tk.FLAT, bd=0, font=('Segoe UI', 11, 'bold'),
            width=3, cursor='hand2'
        )

        tk.Label(floor_panel, text='Floors', bg='#0f3460', fg='#7f8c8d',
                 font=('Segoe UI', 8)).pack(pady=(0, 2))

        tk.Button(floor_panel, text='▲', command=lambda: self._cycle_floor(+1),
                  **ovl_btn_style).pack(padx=4, pady=(0, 1))

        self._floor_label = tk.Label(
            floor_panel,
            text=f"F{self.current_floor}/{self.num_floors}",
            bg='#16213e', fg='white',
            font=('Segoe UI', 10, 'bold'), width=5, pady=4
        )
        self._floor_label.pack(padx=4, pady=2)

        tk.Button(floor_panel, text='▼', command=lambda: self._cycle_floor(-1),
                  **ovl_btn_style).pack(padx=4, pady=(0, 6))

        tk.Frame(floor_panel, bg='#1a1a2e', height=1).pack(fill=tk.X, padx=6, pady=4)

        tk.Button(floor_panel, text='+', command=self._on_add_floor_clicked,
                  bg='#27ae60', fg='white', activebackground='#1e8449',
                  relief=tk.FLAT, bd=0, font=('Segoe UI', 11, 'bold'),
                  width=3, cursor='hand2').pack(padx=4, pady=2)

        tk.Button(floor_panel, text='–', command=self._on_remove_floor_clicked,
                  bg='#e94560', fg='white', activebackground='#c81e45',
                  relief=tk.FLAT, bd=0, font=('Segoe UI', 11, 'bold'),
                  width=3, cursor='hand2').pack(padx=4, pady=2)

        tk.Button(floor_panel, text='⇄', command=self._on_add_connector_clicked,
                  bg='#9b59b6', fg='white', activebackground='#7d3c98',
                  relief=tk.FLAT, bd=0, font=('Segoe UI', 11, 'bold'),
                  width=3, cursor='hand2').pack(padx=4, pady=2)

        # Spacer below to balance vertical centering
        tk.Frame(floor_panel, bg='#0f3460').pack(expand=True)

        # --- Now pack the canvas; it will fill the remaining space (left of the panel) ---
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_layout)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('pick_event', self._on_patch_picked)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.canvas.mpl_connect('button_press_event', self._pan_press)
        self.canvas.mpl_connect('motion_notify_event', self._pan_motion)
        self.canvas.mpl_connect('button_release_event', self._pan_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)

        # --- Metrics Tab Styling ------------------------------------------------
        self.metrics_label = tk.Label(
            tab_metrics,
            text="",
            justify='left',
            anchor='nw',
            bg='#002747',
            fg='white'
        )
        self.metrics_label.pack(fill=tk.BOTH, expand=True)

        # --- Final draw & enter mainloop ----------------------------------------
        self.redraw()
         # Kick off the analytics refresher once to account for multiple threads
        self._update_simulation_analytics()
         # Start draining the simulation -> GUI handoff queue on the main thread.
        self._sim_gui_drain_after_id = None
        self._drain_simulation_gui_queue()
        # Cancel scheduled callbacks and stop the worker thread before Tk
        # tears the root down, otherwise the next firing of
        # _update_simulation_analytics / _drain_simulation_gui_queue lands
        # on a destroyed widget and Tk emits "invalid command name ...".
        self.tk_root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        # Enter the Tk main loop
        self.tk_root.mainloop()

    def _on_window_close(self):
        """Tear down scheduled callbacks + worker thread, then destroy root."""
        for attr in ('_analytics_after_id', '_sim_gui_drain_after_id',
                     '_resize_after_id', '_opt_after_id', '_heat_after_id',
                     '_metrics_after_id'):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    self.tk_root.after_cancel(after_id)
                except Exception:
                    pass
                setattr(self, attr, None)
        try:
            top = getattr(self, 'heat_toplevel', None)
            if top is not None:
                try: top.destroy()
                except Exception: pass
        except Exception:
            pass
        try:
            sim = getattr(self, 'customer_simulation', None)
            if sim is not None:
                sim.hard_stop()
        except Exception:
            pass
        # Any figure still created through pyplot owns a hidden Tk root that
        # would keep mainloop() alive after the main window is gone.
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            self.tk_root.destroy()
        except Exception:
            pass

    def _drain_simulation_gui_queue(self):
        """Periodically drain the simulation worker's GUI handoff queue
        and execute callables on the Tk main thread."""
        import tkinter as _tk
        sim = getattr(self, 'customer_simulation', None)
        if sim is not None and getattr(sim, '_gui_queue', None) is not None:
            q = sim._gui_queue
            # Drain up to N callables per tick so we don't starve the
            # event loop when the queue is saturated.
            for _ in range(64):
                try:
                    fn = q.get_nowait()
                except Exception:
                    break
                try:
                    fn()
                except _tk.TclError:
                    # Widget destroyed mid-call -- stop draining for this tick.
                    break
                except Exception:
                    pass
        try:
            if self.tk_root.winfo_exists():
                self._sim_gui_drain_after_id = self.tk_root.after(
                    16, self._drain_simulation_gui_queue
                )
        except Exception:
            pass

    def _cycle_floor(self, delta):
        ids = sorted(self.floors.keys())
        idx = ids.index(self.current_floor)
        new_idx = max(0, min(len(ids) - 1, idx + delta))
        self.switch_floor(ids[new_idx])

    def _on_add_floor_clicked(self):
        new_id = self.add_floor()
        if getattr(self, '_floor_label', None) is not None:
            try:
                self._floor_label.config(text=f"F{new_id}/{self.num_floors}")
            except Exception:
                pass
        messagebox.showinfo("Floor added",
            f"Floor {new_id} created. Add at least one Connector to make it reachable.")
        self.switch_floor(new_id)

    def _on_remove_floor_clicked(self):
        if self.current_floor == 1:
            messagebox.showerror("Cannot remove", "Floor 1 is required.")
            return
        if messagebox.askyesno("Remove floor", f"Delete floor {self.current_floor}?"):
            self.remove_floor(self.current_floor)



    @staticmethod
    def _rectangles_overlap(a, b, padding=0.0):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw + padding <= bx or ax >= bx + bw + padding or
                    ay + ah + padding <= by or ay >= by + bh + padding)

    def _connector_position_is_free(self, floor_id, x, y, w=1.0, h=1.0,
                                    ignore_wall_name=None):
        """Return True when a connector rectangle can be placed here.

        Sections are visual/category overlays and intentionally do not block
        connector placement; all physical walls, checkout/WC blocks, items, and
        other connectors do block it.
        """
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return False
        floor = self.floors.get(floor_id, {})
        candidate = (x, y, w, h)
        for nm, wall in floor.get('walls', {}).items():
            if nm == ignore_wall_name or nm.startswith('Section_'):
                continue
            wx, wy = wall.get('position', (0, 0))
            ww, wh = wall.get('size', (0, 0))
            if self._rectangles_overlap(candidate, (wx, wy, ww, wh), padding=0.02):
                return False
        for nm, item in floor.get('items', {}).items():
            ix, iy = item.get('position', (0, 0))
            iw, ih = item.get('size', (0, 0))
            if self._rectangles_overlap(candidate, (ix, iy, iw, ih), padding=0.02):
                return False
        return True

    def _find_connector_position(self, floor_id, target_x=None, target_y=None,
                                 size=(1.0, 1.0), ignore_wall_name=None):
        """Find the closest empty connector position to the requested point."""
        w, h = size
        if target_x is None:
            target_x = self.width / 2 - w / 2
        if target_y is None:
            target_y = self.height / 2 - h / 2
        target_x = max(0.05, min(target_x, self.width - w - 0.05))
        target_y = max(0.05, min(target_y, self.height - h - 0.05))

        if self._connector_position_is_free(floor_id, target_x, target_y, w, h,
                                            ignore_wall_name=ignore_wall_name):
            return (target_x, target_y)

        step = 0.25
        max_radius = max(self.width, self.height)
        best = None
        best_d2 = float('inf')
        radius = step
        while radius <= max_radius + step:
            samples = max(16, int(2 * math.pi * radius / step))
            for idx in range(samples):
                ang = 2 * math.pi * idx / samples
                x = target_x + math.cos(ang) * radius
                y = target_y + math.sin(ang) * radius
                x = max(0.05, min(x, self.width - w - 0.05))
                y = max(0.05, min(y, self.height - h - 0.05))
                if self._connector_position_is_free(floor_id, x, y, w, h,
                                                    ignore_wall_name=ignore_wall_name):
                    d2 = (x - target_x) ** 2 + (y - target_y) ** 2
                    if d2 < best_d2:
                        best = (x, y)
                        best_d2 = d2
            if best is not None:
                return best
            radius += step

        # Last-resort deterministic grid scan; keeps the app usable even in very
        # dense layouts by finding the first feasible cell.
        grid_x = np.arange(0.05, max(0.06, self.width - w), step)
        grid_y = np.arange(0.05, max(0.06, self.height - h), step)
        for x in grid_x:
            for y in grid_y:
                if self._connector_position_is_free(floor_id, float(x), float(y), w, h,
                                                    ignore_wall_name=ignore_wall_name):
                    return (float(x), float(y))
        return (target_x, target_y)

    def _unique_connector_display_name(self, requested_name, floor_ids):
        base = (requested_name or "connector").strip() or "connector"
        # Keep names simple and visible in the layout while allowing spaces.
        candidate = base
        idx = 2
        while any(candidate in self.floors.get(fid, {}).get('walls', {}) or
                  candidate in self.floors.get(fid, {}).get('items', {})
                  for fid in floor_ids):
            candidate = f"{base}_{idx}"
            idx += 1
        return candidate

    def _on_add_connector_clicked(self):
        """Create a named connector on the current floor and selected floors."""
        if self.num_floors < 2:
            messagebox.showerror("Need 2+ floors",
                "Add another floor first before creating a connector.")
            return

        others = sorted(f for f in self.floors if f != self.current_floor)
        if not others:
            return

        dlg = tk.Toplevel(self.tk_root)
        dlg.transient(self.tk_root)
        dlg.title("Connector — Target Floors")
        tk.Label(dlg, text=f"Connect floor {self.current_floor} to which floors?\n"
                           "(select one or more)").pack(padx=10, pady=(10, 5))

        vars_map = {}
        frm = tk.Frame(dlg)
        frm.pack(padx=10, pady=5, anchor='w')
        for f in others:
            v = tk.BooleanVar(value=False)
            tk.Checkbutton(frm, text=f"Floor {f}", variable=v).pack(anchor='w')
            vars_map[f] = v

        result = {'targets': None}
        def _ok():
            result['targets'] = [f for f, v in vars_map.items() if v.get()]
            dlg.destroy()
        def _cancel():
            dlg.destroy()

        btns = tk.Frame(dlg); btns.pack(pady=8)
        tk.Button(btns, text="OK",     width=10, command=_ok).pack(side='left', padx=4)
        tk.Button(btns, text="Cancel", width=10, command=_cancel).pack(side='left', padx=4)

        dlg.grab_set()
        self.tk_root.wait_window(dlg)

        targets = result['targets']
        if not targets:
            return

        floor_ids = [self.current_floor] + list(targets)
        requested_name = simpledialog.askstring(
            "Connector Name",
            "Connector name (e.g. stairs, elevator):",
            initialvalue="stairs",
            parent=self.tk_root,
        )
        if requested_name is None:
            return
        display_name = self._unique_connector_display_name(requested_name, floor_ids)
        if display_name != requested_name.strip():
            messagebox.showinfo(
                "Connector Name Adjusted",
                f"'{requested_name.strip()}' already exists on one of these floors.\n"
                f"Using '{display_name}' on all connected floors.",
                parent=self.tk_root,
            )

        # Counting connectors reuses a live id once one has been deleted, which
        # would overwrite that connector's floor mapping; take the first id no
        # connector or connector wall still holds.
        used_ids = set(self.connectors)
        for fdata in self.floors.values():
            for wdata in fdata.get('walls', {}).values():
                if isinstance(wdata, dict) and wdata.get('connector_id'):
                    used_ids.add(wdata['connector_id'])
        n = 1
        while f"connector_{n}" in used_ids:
            n += 1
        connector_id = f"connector_{n}"

        def _place(fid):
            px, py = self._find_connector_position(fid, self.width / 2 - 0.5,
                                                   self.height / 2 - 0.5,
                                                   size=(1.0, 1.0))
            self.floors[fid]['walls'][display_name] = {
                'position': [px, py],
                'size':     [1.0, 1.0],
                'color':    '#9b59b6',
                'category': 'Connector',
                'connector_id': connector_id,
                'connector_kind': display_name,
            }
            return display_name

        mapping = {self.current_floor: _place(self.current_floor)}
        for tf in targets:
            mapping[tf] = _place(tf)

        self.connectors[connector_id] = mapping
        self._invalidate_sim_geometry()
        self.redraw()




    # -------------------------------------------------------------
# PROJECTIONS TAB  -  Monte Carlo daily / monthly forecasting
# -------------------------------------------------------------
