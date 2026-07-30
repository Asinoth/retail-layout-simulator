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

from shop_architecture import generate_architecture, FULL_CATALOG, scale_catalog


class GenerateMixin:
    """Auto-generate a realistic full layout.

    The geometry itself comes from ``shop_architecture.generate_architecture``,
    which builds the floor plan according to the canonical retail layout
    archetypes from the cited literature (Sorensen 2009; Larson 2005):
    grid (supermarket gondola aisles), racetrack (department loop), and
    freeform (boutique). This mixin owns only the Tk dialog, the
    multi-floor bookkeeping, the impulse-item zone, and the price
    assignment.
    """

    def _on_generate_layout(self):
        """Auto-generate layout.

        Floor 1 is always treated as the main floor automatically.  Other floors
        are generated as secondary floors and skip Checkout, WC, entrance and
        impulse items.
        """

        if self.zoom != 1.0:
            messagebox.showwarning(
                "Zoom Reset Required",
                "Please reset zoom to 100% before generating a new layout.",
                parent=self.tk_root
            )
            return

        W, H = self.width, self.height

        # Section/item assortment lives in shop_architecture.FULL_CATALOG;
        # scale_catalog sizes it to the floor area (bigger shop -> more
        # sections and more items per section, like a real retail space).
        shop_types = FULL_CATALOG

        # 2) Prompt user to pick shop type
        dlg = tk.Toplevel(self.tk_root)
        dlg.transient(self.tk_root)
        dlg.title("Select Shop Type")
        tk.Label(dlg, text="Shop Type:")\
            .grid(row=0, column=0, padx=5, pady=5, sticky='e')
        shop_var = tk.StringVar(value=list(shop_types.keys())[0])
        ttk.Combobox(
            dlg, textvariable=shop_var,
            values=list(shop_types.keys()),
            state='readonly', width=25
        ).grid(row=0, column=1, padx=5, pady=5)

        selection = {'shop': None}
        def _on_ok():
            selection['shop'] = shop_var.get()
            dlg.destroy()
        def _on_cancel():
            dlg.destroy()

        tk.Button(dlg, text="OK",     width=10, command=_on_ok)\
            .grid(row=1, column=0, padx=5, pady=5)
        tk.Button(dlg, text="Cancel", width=10, command=_on_cancel)\
            .grid(row=1, column=1, padx=5, pady=5)

        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        shop = selection['shop']
        if not shop:
            return
        # rng=random -> the assortment itself varies between generates
        # (sampled sections/items + count jitter), on top of the engine's
        # structural variation (slot shuffle, WC corner, front depth).
        sections = scale_catalog(shop, W, H, rng=random)

        # Floor 1 is always the main/ground floor.  No prompt.
        cur_floor = getattr(self, 'current_floor', 1)
        is_main_floor = (cur_floor == 1)

        # Preserve any connectors that already exist on this floor BEFORE
        # clearing walls; connectors are user-created vertical access points.
        preserved_connectors = {}
        try:
            fdata = self.floors[self.current_floor]
            for nm, w in list(fdata.get('walls', {}).items()):
                if w.get('category') == 'Connector':
                    preserved_connectors[nm] = dict(w)
        except Exception:
            pass

        self.items.clear()
        self.walls.clear()

        # 3) Generate the realistic floor plan (grid / racetrack / freeform
        #    per shop type). The engine guarantees no overlaps, proper
        #    aisle widths, and door-to-item reachability.
        arch = generate_architecture(W, H, shop, sections,
                                     rng=random, is_main_floor=is_main_floor)
        for nm, wd in arch['walls'].items():
            self.walls[nm] = wd
        # Restore preserved connectors (may overwrite a generated wall of
        # the same name; user-created access points win).
        for nm, wd in preserved_connectors.items():
            self.walls[nm] = wd
        for nm, it in arch['items'].items():
            self.items[nm] = it

        if is_main_floor and arch.get('door_position'):
            self.door_position = arch['door_position']
            self.door_side = arch['door_side']

        # Impulse Purchases -- ONLY main floor (needs Checkout)
        if is_main_floor and 'Checkout' in self.walls:
            checkout_pos = self.walls['Checkout']['position']
            checkout_size = self.walls['Checkout']['size']
            impulse_zone_x = max(0.1, checkout_pos[0] - 2.0)
            impulse_zone_y = checkout_pos[1]
            impulse_zone_w = min(4.0, self.width - impulse_zone_x - 0.1)
            impulse_zone_h = checkout_size[1]

            try:
                shop_type = self._detect_shop_type()
            except Exception:
                shop_type = shop

            impulse_items_by_shop = {
                'Grocery Store': ["Candy Bars","Gum","Mints","Energy Drinks","Beef Jerky","Travel Tissues"],
                'Bookstore': ["Bookmarks","Reading Glasses","Coffee Mugs","Sticky Notes","Pens","Magazines","Journals","Book Lights","Gift Cards","Postcards"],
                'Electronics Store': ["Phone Cases","Cables","Screen Protectors","USB Drives","Earbuds","Power Banks","Phone Stands","Cleaning Wipes","Adapters","Memory Cards"],
                'Clothing Store': ["Belts","Socks","Accessories","Hair Ties","Sunglasses","Scarves","Jewelry","Perfume Samples","Gift Cards","Shoe Care"],
                'Pharmacy': ["Travel Size Items","Band-Aids","Hand Sanitizer","Lip Balm","Eye Drops","Throat Lozenges","Travel Packets","Vitamins","Tissues","Hand Cream"],
                'Hardware Store': ["Screws Pack","Small Tools","Tape","Glue","Measuring Tape","Work Gloves","Safety Glasses","Utility Knife","Flashlight","Batteries"],
                'Cafe': ["Coffee Beans","Sugar Packets","Pastries","Bottled Water","Mints","Napkins","Coffee Mugs","Gift Cards","Magazines","Cookies"]
            }

            available_impulse_items = impulse_items_by_shop.get(shop_type,
                ["Candy", "Gum", "Magazines", "Gift Cards", "Mints"])

            if impulse_zone_w > 0.5 and impulse_zone_h > 0.3:
                num_impulse_items = np.random.randint(3, 6)
                selected_impulse_items = np.random.choice(available_impulse_items,
                    size=min(num_impulse_items, len(available_impulse_items)),
                    replace=False)

                placed_impulse_items = []
                for item_name in selected_impulse_items:
                    item_w = np.random.uniform(0.2, 0.4)
                    item_h = np.random.uniform(0.3, 0.5)
                    placed = False
                    for attempt in range(20):
                        if impulse_zone_w - item_w <= 0 or impulse_zone_h - item_h <= 0:
                            break
                        item_x = np.random.uniform(impulse_zone_x, impulse_zone_x + impulse_zone_w - item_w)
                        item_y = np.random.uniform(impulse_zone_y, impulse_zone_y + impulse_zone_h - item_h)
                        collision = False
                        for wall_name, wall_data in self.walls.items():
                            if wall_name.startswith('Section_'):
                                continue
                            if wall_data.get('category') == 'Connector':
                                continue
                            wx, wy = wall_data['position']; ww, wh = wall_data['size']
                            if not (item_x + item_w <= wx or item_x >= wx + ww
                                    or item_y + item_h <= wy or item_y >= wy + wh):
                                collision = True; break
                        if not collision:
                            for ox, oy, ow, oh in placed_impulse_items:
                                if not (item_x + item_w <= ox or item_x >= ox + ow
                                        or item_y + item_h <= oy or item_y >= oy + oh):
                                    collision = True; break
                        if not collision:
                            for existing_item in self.items.values():
                                ex, ey = existing_item['position']; ew, eh = existing_item['size']
                                if not (item_x + item_w <= ex or item_x >= ex + ew
                                        or item_y + item_h <= ey or item_y >= ey + eh):
                                    collision = True; break
                        if not collision:
                            self.items[item_name] = {
                                'position': (item_x, item_y),
                                'size': (item_w, item_h),
                                'category': 'Impulse',
                                'color': '#FFD700',
                                'shop_type': shop_type
                            }
                            placed_impulse_items.append((item_x, item_y, item_w, item_h))
                            placed = True; break
                    if not placed:
                        item_w *= 0.7; item_h *= 0.7
                        for attempt in range(10):
                            if impulse_zone_w - item_w <= 0 or impulse_zone_h - item_h <= 0:
                                break
                            item_x = np.random.uniform(impulse_zone_x, impulse_zone_x + impulse_zone_w - item_w)
                            item_y = np.random.uniform(impulse_zone_y, impulse_zone_y + impulse_zone_h - item_h)
                            collision = any(
                                not (item_x + item_w <= ox or item_x >= ox + ow
                                     or item_y + item_h <= oy or item_y >= oy + oh)
                                for ox, oy, ow, oh in placed_impulse_items
                            )
                            if not collision:
                                self.items[item_name] = {
                                    'position': (item_x, item_y),
                                    'size': (item_w, item_h),
                                    'category': 'Impulse',
                                    'color': '#FFD700',
                                    'shop_type': shop_type
                                }
                                placed_impulse_items.append((item_x, item_y, item_w, item_h))
                                break

        # Store per-floor main/entrance metadata. Floor 1 remains the global
        # simulation entrance even if the user later generates upper floors.
        try:
            fmeta = self.floors[self.current_floor]
            fmeta['is_main_floor'] = bool(is_main_floor)
            if is_main_floor:
                fmeta['door_position'] = tuple(self.door_position) if self.door_position else None
                fmeta['door_side'] = self.door_side
            else:
                fmeta['door_position'] = None
                fmeta['door_side'] = None
            if 1 in self.floors:
                self.floors[1]['is_main_floor'] = True
        except Exception:
            pass

        # Sync simulation door info from floor 1 (only floor 1 is main).
        try:
            floor1 = self.floors.get(1, {})
            main_door = floor1.get('door_position') or self.door_position
            main_side = floor1.get('door_side') or self.door_side
            if main_door:
                self.door_position = tuple(main_door)
                self.door_side = main_side
            if hasattr(self, 'customer_simulation') and self.customer_simulation is not None:
                self.customer_simulation.door_position = self.door_position
                self.customer_simulation.door_side = self.door_side
                self.customer_simulation.geometry_dirty = True
        except Exception:
            pass

        try:
            self._assign_default_prices()
        except Exception:
            pass

        self.redraw()
