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


def _rects_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or ax >= bx + bw
                or ay + ah <= by or ay >= by + bh)


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

        # 3) Generate the realistic floor plan (grid / racetrack / freeform
        #    per shop type). The engine guarantees no overlaps, proper
        #    aisle widths, and door-to-item reachability; fit_assortment
        #    keeps trimming the assortment until the floor really carries
        #    it, so the plan the user gets is not a truncated one.
        #    Built before the floor is cleared: a floor the engine refuses
        #    must leave the existing layout untouched.
        try:
            arch = generate_architecture(W, H, shop, sections,
                                         rng=random,
                                         is_main_floor=is_main_floor,
                                         fit_assortment=True)
        except ValueError as exc:
            messagebox.showerror(
                "Floor Too Small",
                f"No layout was generated: {exc}.",
                parent=self.tk_root
            )
            return

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

        for nm, wd in arch['walls'].items():
            self.walls[nm] = wd
        # Restore preserved connectors (may overwrite a generated wall of
        # the same name; user-created access points win).
        for nm, wd in preserved_connectors.items():
            self.walls[nm] = wd
        # Item dicts carry 'zone', the Section_ wall each fixture sits in
        # (a department split over two gondolas has two zones).
        for nm, it in arch['items'].items():
            self.items[nm] = dict(it, zone=it.get('zone'))

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

            # Use the type the user just picked: inferring it from item
            # categories misses most catalog section names and collapses
            # to 'Generic Store'.
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

            # The engine keeps the doorway approach clear so customers can
            # walk in; a rack there would stand in the entrance. The gaps
            # between the checkout lanes are left available on purpose --
            # that is exactly where impulse merchandise belongs.
            door_keepout = arch.get('door_keepout')

            def _impulse_collides(x, y, w, h):
                """True when a rack of this footprint cannot go here."""
                rect = (x, y, w, h)
                if door_keepout is not None and _rects_overlap(rect, door_keepout):
                    return True
                for wall_name, wall_data in self.walls.items():
                    if wall_name.startswith('Section_'):
                        continue
                    if wall_data.get('category') == 'Connector':
                        continue
                    wx, wy = wall_data['position']; ww, wh = wall_data['size']
                    if _rects_overlap(rect, (wx, wy, ww, wh)):
                        return True
                # Racks placed earlier are already in self.items, so this
                # one loop covers the catalog fixtures and them alike.
                for existing_item in self.items.values():
                    ex, ey = existing_item['position']; ew, eh = existing_item['size']
                    if _rects_overlap(rect, (ex, ey, ew, eh)):
                        return True
                return False

            if impulse_zone_w > 0.5 and impulse_zone_h > 0.3:
                num_impulse_items = np.random.randint(3, 6)
                selected_impulse_items = np.random.choice(available_impulse_items,
                    size=min(num_impulse_items, len(available_impulse_items)),
                    replace=False)

                for item_name in selected_impulse_items:
                    item_w = np.random.uniform(0.2, 0.4)
                    item_h = np.random.uniform(0.3, 0.5)
                    placed = False
                    # A crowded lane zone gets a second pass with a smaller
                    # rack; a rack that fits nowhere is left out.
                    for shrink, attempts in ((1.0, 20), (0.7, 10)):
                        rack_w, rack_h = item_w * shrink, item_h * shrink
                        for attempt in range(attempts):
                            if impulse_zone_w - rack_w <= 0 or impulse_zone_h - rack_h <= 0:
                                break
                            item_x = np.random.uniform(impulse_zone_x, impulse_zone_x + impulse_zone_w - rack_w)
                            item_y = np.random.uniform(impulse_zone_y, impulse_zone_y + impulse_zone_h - rack_h)
                            if _impulse_collides(item_x, item_y, rack_w, rack_h):
                                continue
                            self.items[item_name] = {
                                'position': (item_x, item_y),
                                'size': (rack_w, rack_h),
                                'category': 'Impulse',
                                'color': '#FFD700',
                                'shop_type': shop_type
                            }
                            placed = True
                            break
                        if placed:
                            break

        # The engine lays the floor out without knowing about the stairs
        # and lifts the user placed, so a preserved connector can end up
        # under a fixture. Move each one to the nearest free spot now that
        # every fixture, impulse racks included, is down; connectors are
        # keyed per floor, so the same connector on other floors stays put.
        for nm, wd in preserved_connectors.items():
            cx, cy = wd.get('position', (0.0, 0.0))
            cw, ch = wd.get('size', (1.0, 1.0))
            px, py = self._find_connector_position(
                cur_floor, cx, cy, (cw, ch), ignore_wall_name=nm)
            if (px, py) != (cx, cy):
                self.walls[nm] = dict(wd, position=(px, py))

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
                # Every section, Checkout and WC was replaced; zone
                # attribution must be rebuilt from the new walls.
                self.customer_simulation.invalidate_zones_cache()
        except Exception:
            pass

        try:
            self._assign_default_prices()
        except Exception:
            pass

        self.redraw()

        # A shallow or crowded floor can run out of fixture length; name
        # the items the engine could not fit instead of leaving them out
        # silently.
        dropped = arch.get('dropped') or []
        if dropped:
            messagebox.showwarning(
                "Assortment Trimmed",
                f"{len(dropped)} item(s) did not fit on this "
                f"{W:.0f}x{H:.0f} m floor and were left out: "
                f"{', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}",
                parent=self.tk_root
            )
