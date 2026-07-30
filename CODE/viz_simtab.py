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


class SimTabMixin:
    """Simulation tab UI plus start/stop/clear and customer-legend helpers."""

    def _create_simulation_tab(self, notebook):
        """
        Create the customer simulation tab with controls and analytics.
        """

        tab_simulation = tk.Frame(notebook, bg='#002747')
        notebook.add(tab_simulation, text="Customer Flow")

        # NOTE: the live heat-map preview that used to live at the bottom of
        # this tab has been removed. The full-screen heat map (popup) remains
        # available via the "Show Heat Map" button below. ``heat_im`` /
        # ``heat_canvas`` are intentionally not created here so any leftover
        # ``hasattr(self, 'heat_im')`` checks elsewhere become safe no-ops.

        main_frame = tk.Frame(tab_simulation, bg='#002747')
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(main_frame, bg='#002747')
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        legend_frame = tk.Frame(main_frame, bg='#002747', relief=tk.RAISED, bd=2)
        legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

        tk.Label(controls_frame,
                 text="Simulation Controls",
                 fg='white', bg='#002747', font=('Arial', 12, 'bold')
                 ).grid(row=0, column=0, columnspan=5, pady=5)

        self.start_sim_btn = tk.Button(controls_frame,
                                       text="Start Simulation",
                                       command=self._start_simulation,
                                       bg='green', fg='white')
        self.start_sim_btn.grid(row=1, column=0, padx=5, pady=2)

        self.stop_sim_btn = tk.Button(controls_frame,
                                      text="Stop Simulation",
                                      command=self._stop_simulation,
                                      bg='red', fg='white')
        self.stop_sim_btn.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(controls_frame,
                 text="Speed:", fg='white', bg='#002747'
                 ).grid(row=1, column=2, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(controls_frame,
                 from_=0.1, to=5.0, resolution=0.1,
                 orient=tk.HORIZONTAL,
                 variable=self.speed_var,
                 command=self._update_simulation_speed,
                 bg='#002747', fg='white'
                 ).grid(row=1, column=3, padx=5)

        tk.Label(controls_frame,
                 text="Spawn Rate:", fg='white', bg='#002747'
                 ).grid(row=2, column=0, padx=5)
        self.spawn_var = tk.DoubleVar(value=0.2)
        tk.Scale(controls_frame,
                 from_=0.05, to=1.0, resolution=0.05,
                 orient=tk.HORIZONTAL,
                 variable=self.spawn_var,
                 command=self._update_spawn_rate,
                 bg='#002747', fg='white'
                 ).grid(row=2, column=1, padx=5)

        tk.Button(controls_frame,
                  text="Clear Customers",
                  command=self._clear_simulation,
                  bg='orange', fg='white'
                  ).grid(row=2, column=2, padx=5, pady=2)
        tk.Button(controls_frame,
                  text="Clear Current Data",
                  command=self._clear_current_data,
                  bg='orange', fg='white'
                  ).grid(row=2, column=3, padx=5, pady=2)

        tk.Button(controls_frame,
                  text="Show Heat Map",
                  command=self._show_heat_map,
                  bg='purple', fg='white'
                  ).grid(row=2, column=4, padx=5, pady=2)

        # -- Max Customers in Shop (replaces old hard-coded cap of 40) -----
        # Bound to ``simulation.max_customers``. Default 20 matches the new
        # academic-paper baseline; values > 20 trigger a one-time warning
        # because simulation + optimization runtime scale ~linearly with the
        # concurrent population.
        tk.Label(controls_frame,
                 text="Max Customers in Shop:",
                 fg='white', bg='#002747'
                 ).grid(row=3, column=0, padx=5, pady=(8, 2), sticky='e')

        default_max = int(getattr(self.customer_simulation, 'max_customers', 20))
        if default_max <= 0:
            default_max = 20
        self.max_customers_var = tk.IntVar(value=default_max)
        self._max_customers_last_warned = default_max  # remember last warned level
        self._max_customers_warning_threshold = 20

        self.max_customers_spin = tk.Spinbox(
            controls_frame,
            from_=1, to=100, increment=1,
            width=6,
            textvariable=self.max_customers_var,
            command=self._update_max_customers,
            bg='#001a33', fg='white', insertbackground='white',
            highlightthickness=0, relief=tk.FLAT, justify='center',
        )
        self.max_customers_spin.grid(row=3, column=1, padx=5, pady=(8, 2), sticky='w')
        # Also commit when the user types a number and tabs/clicks out.
        self.max_customers_spin.bind('<Return>',   lambda e: self._update_max_customers())
        self.max_customers_spin.bind('<FocusOut>', lambda e: self._update_max_customers())

        tk.Label(controls_frame,
                 text="(1–100; warning above 20)",
                 fg='#9fb6c9', bg='#002747', font=('Arial', 8, 'italic')
                 ).grid(row=3, column=2, columnspan=3, padx=5, pady=(8, 2), sticky='w')

        # Push the new cap into the simulation at startup so the model and
        # the UI agree before the user touches anything.
        try:
            self.customer_simulation.max_customers = int(default_max)
        except Exception:
            pass

        analytics_frame = tk.Frame(controls_frame, bg='#002747')
        analytics_frame.grid(row=4, column=0, columnspan=5,
                             sticky='nsew', pady=10)
        analytics_frame.columnconfigure(0, weight=1)
        analytics_frame.columnconfigure(1, weight=1)

        tk.Label(analytics_frame,
                 text="Real-time Analytics",
                 fg='white', bg='#002747',
                 font=('Arial', 12, 'bold')
                 ).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 5))

        self.simulation_analytics_label = tk.Label(
            analytics_frame,
            text="Start simulation to see analytics...",
            justify='left', anchor='nw',
            bg='#002747', fg='white',
            font=('Courier', 10)
        )
        self.simulation_analytics_label.grid(row=1, column=0,
                                             sticky='nsew', padx=(5, 2))

        self.product_metrics_label = tk.Label(
            analytics_frame,
            text="",
            justify='left', anchor='nw',
            bg='#002747', fg='white',
            font=('Courier', 10)
        )
        self.product_metrics_label.grid(row=1, column=1,
                                        sticky='nsew', padx=(2, 5))

        self._create_customer_legend(legend_frame)

        self._update_simulation_analytics()

    def _update_max_customers(self, *_args):
        """Push the spin-box value into ``simulation.max_customers``.

        - Clamps the input to ``[1, 100]`` and writes it back to the
          Spinbox so the UI never displays an out-of-range value.
        - Shows a warning the *first* time the user crosses the
          ``_max_customers_warning_threshold`` (default 20) since the
          last warning, because runtime scales with population.
        """
        try:
            raw = int(self.max_customers_var.get())
        except (tk.TclError, ValueError):
            # Bad text input -- revert to the last accepted value.
            try:
                raw = int(getattr(self.customer_simulation, 'max_customers', 20))
            except Exception:
                raw = 20

        clamped = max(1, min(100, raw))
        if clamped != raw:
            try:
                self.max_customers_var.set(clamped)
            except Exception:
                pass

        try:
            self.customer_simulation.max_customers = int(clamped)
        except Exception:
            pass

        threshold = getattr(self, '_max_customers_warning_threshold', 20)
        last = getattr(self, '_max_customers_last_warned', threshold)
        if clamped > threshold and last <= threshold:
            try:
                messagebox.showwarning(
                    "High customer cap",
                    f"Max Customers in Shop is set to {clamped}.\n\n"
                    "Values above 20 may significantly increase simulation "
                    "and optimization runtime, because every concurrent "
                    "customer is fully simulated (path-finding, basket "
                    "selection, checkout) each tick.\n\n"
                    "Recommended ≤ 20 for fast GA optimization.",
                    parent=self.tk_root
                )
            except Exception:
                pass
        # Remember the new level so further bumps inside the warning band
        # don't keep nagging the user.
        self._max_customers_last_warned = clamped

    def _clear_current_data(self):
        """Reset all simulation analytics and heat-map data to zero."""
        sim = self.customer_simulation

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
            'item_conversion_rates':      defaultdict(lambda: {'visits': 0, 'purchases': 0}),
            'foot_traffic_density':       defaultdict(float),
            'bottlenecks':                defaultdict(int),
            'queue_wait_times':           [],
            'processing_times':           [],
            'data_points_collected':      0,
            'accuracy_metrics':           [],
            'optimization_history':       [],
            'pre_optimization_revenue':   0.0,
            'post_optimization_revenue':  0.0,
            'optimization_impact':        0.0,
            'measurement_duration':       90,
            'pre_window_revenue':         0.0,
            'post_window_revenue':        0.0,
            'pre_window_start':           None,
            'post_window_start':          None,
            'phase':                      None,
            'floor_visits':               defaultdict(int),
        }

        self.bottleneck_threshold = 50
        sim.bottleneck_threshold = 50

        sim.reset_heatmap()
        sim.reset_heatmap_on_start = True

        self._refresh_embedded_heatmap(zero=True)

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

        sim.run_time = 0.0
        sim.prev_state_by_cust.clear()
        sim.simulation_start_time = time.time()

        messagebox.showinfo(
            "Data Cleared",
            "All simulation metrics and heat-map data have been reset.",
            parent=self.tk_root
        )

    def _detect_shop_type(self):
        """Detect shop type based on existing item categories."""
        category_counts = {}
        for item_data in self.items.values():
            if item_data.get('category') != 'Impulse':
                category = item_data.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1

        if not category_counts:
            return 'Generic Store'

        category_to_shop = {
            'Fresh Produce': 'Grocery Store',
            'Dairy & Refrigerated': 'Grocery Store',
            'Pantry & Dry Goods': 'Grocery Store',
            'Beverages': 'Grocery Store',
            'Snacks': 'Grocery Store',

            'Fiction': 'Bookstore',
            'Non-Fiction': 'Bookstore',
            'Children': 'Bookstore',
            'Magazines': 'Bookstore',
            'Stationery': 'Bookstore',

            'Mobile Phones': 'Electronics Store',
            'Computers': 'Electronics Store',
            'Audio': 'Electronics Store',
            'TV & Theatre': 'Electronics Store',
            'Accessories': 'Electronics Store',

            'Men': 'Clothing Store',
            'Women': 'Clothing Store',
            'Kids': 'Clothing Store',
            'Shoes & Acc.': 'Clothing Store',
            'Sale': 'Clothing Store',

            'Prescription': 'Pharmacy',
            'OTC': 'Pharmacy',
            'Wellness': 'Pharmacy',
            'Personal Care': 'Pharmacy',
            'Equipment': 'Pharmacy',

            'Tools': 'Hardware Store',
            'Paint': 'Hardware Store',
            'Electrical': 'Hardware Store',
            'Plumbing': 'Hardware Store',
            'Garden': 'Hardware Store',

            'Seating': 'Cafe',
            'Counter': 'Cafe',
            'Kitchen': 'Cafe',
            'Restroom': 'Cafe',
            'Pastry': 'Cafe'
        }

        shop_type_counts = {}
        for category in category_counts.keys():
            shop_type = category_to_shop.get(category, 'Generic Store')
            shop_type_counts[shop_type] = shop_type_counts.get(shop_type, 0) + category_counts[category]

        if shop_type_counts:
            return max(shop_type_counts.items(), key=lambda x: x[1])[0]
        else:
            return 'Generic Store'

    def _create_customer_legend(self, parent_frame):
        """Create a legend showing customer state colors."""
        title_label = tk.Label(parent_frame, text="Customer States",
                               fg='white', bg='#002747',
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(10, 5))

        state_colors = {
            'Entering': '#00ff00',
            'Moving': '#0000ff',
            'Shopping': '#ffa500',
            'Checking Out': '#ff0000',
            'Exiting': '#800080'
        }

        state_descriptions = {
            'Entering': 'Just entered the shop',
            'Moving': 'Moving to next item',
            'Shopping': 'Browsing/selecting items',
            'Checking Out': 'At checkout counter',
            'Exiting': 'Leaving the shop'
        }

        for state, color in state_colors.items():
            entry_frame = tk.Frame(parent_frame, bg='#002747')
            entry_frame.pack(fill=tk.X, padx=5, pady=2)

            color_label = tk.Label(entry_frame, text="●",
                                   fg=color, bg='#002747',
                                   font=('Arial', 16))
            color_label.pack(side=tk.LEFT, padx=5)

            text_frame = tk.Frame(entry_frame, bg='#002747')
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            state_label = tk.Label(text_frame, text=state,
                                   fg='white', bg='#002747',
                                   font=('Arial', 10, 'bold'))
            state_label.pack(anchor='w')

            desc_label = tk.Label(text_frame, text=state_descriptions[state],
                                  fg='lightgray', bg='#002747',
                                  font=('Arial', 8), wraplength=150)
            desc_label.pack(anchor='w')

        separator = tk.Frame(parent_frame, height=2, bg='white')
        separator.pack(fill=tk.X, padx=10, pady=10)

        info_label = tk.Label(parent_frame,
                              text="Customer Behavior:\n• Color changes with state\n• Customers follow shopping lists\n• Must checkout before exiting",
                              fg='lightblue', bg='#002747',
                              font=('Arial', 9), justify=tk.LEFT)
        info_label.pack(padx=5, pady=5)

    def _start_simulation(self):
        """Start the customer flow simulation.
        Validates floor 1 (ground floor) has the required elements
        regardless of which floor is currently displayed."""
        # Always check floor 1 -- entrance, checkout, and items live there
        floor1 = self.floors.get(1, {})
        floor1_items = floor1.get('items', {})
        floor1_walls = floor1.get('walls', {})

        # Synchronize the global simulation entrance from floor 1 metadata or
        # infer it from a generated wall gap before validating.
        try:
            self.customer_simulation._sync_entrance_from_shop()
            self.door_position = self.customer_simulation.door_position
            self.door_side = self.customer_simulation.door_side
        except Exception:
            pass

        # Check items across ALL reachable floors, not just current
        all_items = self.all_items_across_floors()
        if not all_items:
            messagebox.showwarning("No Items",
                                   "Please add some items before starting simulation.",
                                   parent=self.tk_root)
            return
        if not self.door_position:
            messagebox.showerror("Missing Entrance",
                                 "Please define an entrance (on floor 1) before starting.",
                                 parent=self.tk_root)
            return
        if 'Checkout' not in floor1_walls and 'Checkout' in floor1_items:
            # Legacy/dataset layouts stored Checkout as an item. Promote it to a
            # wall/special area so customer routing can find it consistently.
            floor1_walls['Checkout'] = floor1_items.pop('Checkout')
        if 'Checkout' not in floor1_walls:
            messagebox.showerror("Missing Checkout",
                                 "Please define a checkout area (on floor 1) before starting.",
                                 parent=self.tk_root)
            return

        if getattr(self.customer_simulation, 'reset_heatmap_on_start', False):
            self._refresh_embedded_heatmap(zero=True)

        self.customer_simulation.start_simulation()
        self.start_sim_btn.config(state=tk.DISABLED)
        self.stop_sim_btn.config(state=tk.NORMAL)

    def _stop_simulation(self):
        """Stop (pause) the customer flow simulation. Customers stay visible."""
        self.customer_simulation.stop_simulation()
        self.start_sim_btn.config(state=tk.NORMAL)
        self.stop_sim_btn.config(state=tk.DISABLED)

        try:
            from visualizer import export_analytics_to_json
            export_analytics_to_json(self)
        except Exception as e:
            print(f"Failed to export analytics: {e}")

    def _clear_simulation(self):
        """Clear all customers from simulation."""
        self.customer_simulation.customers.clear()

        if hasattr(self.customer_simulation, 'customer_patches_by_id'):
            for patch in self.customer_simulation.customer_patches_by_id.values():
                try:
                    patch.remove()
                except:
                    pass
            self.customer_simulation.customer_patches_by_id.clear()

        if hasattr(self.customer_simulation, 'customer_patches'):
            for patch in self.customer_simulation.customer_patches:
                try:
                    patch.remove()
                except:
                    pass
            self.customer_simulation.customer_patches.clear()

        self.canvas.draw_idle()