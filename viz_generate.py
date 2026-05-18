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


class GenerateMixin:
    """Auto-generate full layout (single very large method)."""

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
        t = 0.2
        threshold = 0.6
        self._iw_counter = 1

        shop_types = {
            'Grocery Store': [
                ('Fresh Produce',    ["Apples","Bananas","Carrots"]),
                ('Dairy & Refrigerated', ["Milk","Cheese","Yogurt"]),
                ('Pantry & Dry Goods',   ["Rice","Pasta","Canned Beans"]),
                ('Beverages',           ["Water","Soda","Juice"]),
                ('Snacks',              ["Chips","Chocolates","Nuts"]),
            ],
            'Bookstore': [
                ('Fiction',        ["Novels","Short Stories","Poetry"]),
                ('Non-Fiction',    ["Biographies","History","Science"]),
                ('Children',       ["Picture Books","Young Adult","Coloring Books"]),
                ('Magazines',      ["Fashion","Tech","Health"]),
                ('Stationery',     ["Notebooks","Pens","Calendars"]),
            ],
            'Electronics Store': [
                ('Mobile Phones', ["Smartphone A","Feature Phone","Smartphone B"]),
                ('Computers',     ["Laptop X","Tablet","Desktop Y"]),
                ('Audio',         ["Headphones","Speakers","Microphone"]),
                ('TV & Theatre',  ["LED TVs","Soundbars","Projectors"]),
                ('Accessories',   ["Chargers","Cables","Cases"]),
            ],
            'Clothing Store': [
                ('Men',       ["Shirts","Trousers","Jackets"]),
                ('Women',     ["Dresses","Blouses","Skirts"]),
                ('Kids',      ["T-Shirts","Shorts","Jeans"]),
                ('Shoes & Acc.', ["Sneakers","Belts","Hats"]),
                ('Sale',      ["Clearance1","Clearance2","Clearance3"]),
            ],
            'Pharmacy': [
                ('Prescription',   ["Drug A","Drug B","Insulin"]),
                ('OTC',            ["Pain Relievers","Cold/Flu","Allergy Meds"]),
                ('Wellness',       ["Vitamins","Supplements","Protein Bars"]),
                ('Personal Care',  ["Shampoo","Toothpaste","Deodorant"]),
                ('Equipment',      ["Bandages","Thermometers","Wheelchair"]),
            ],
            'Hardware Store': [
                ('Tools',      ["Hammers","Screwdrivers","Wrenches"]),
                ('Paint',      ["Paint","Brushes","Rollers"]),
                ('Electrical', ["Wiring","Switches","Outlets"]),
                ('Plumbing',   ["Pipes","Fittings","Valves"]),
                ('Garden',     ["Seeds","Soil","Plants"]),
            ],
            'Cafe': [
                ('Seating',  ["Tables","Chairs","Benches"]),
                ('Counter',  ["Cash Register","Display","POS System"]),
                ('Kitchen',  ["Coffee Machine","Oven","Refrigerator"]),
                ('Restroom', ["Sink","Stall","Toilet"]),
                ('Pastry',   ["Cakes","Pastries","Bread"]),
            ],
        }

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
        sections = shop_types[shop]

        # 2b) Floor 1 is always the main/ground floor.  No prompt.
        cur_floor = getattr(self, 'current_floor', 1)
        is_main_floor = (cur_floor == 1)

        # 3) Build & retry until occupancy >= threshold
        max_outer_attempts = 8
        outer_attempt = 0
        while True:
            outer_attempt += 1

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

            # a) Outer walls
            door_len = 2.0
            gap_x = gap_y = None
            if is_main_floor:
                side = random.choice(['bottom','top','left','right'])
                if side == 'bottom':
                    gap_x = random.uniform(t, W - t - door_len)
                    self.door_position = (gap_x + door_len/2, 0)
                    self.door_side = side
                    self.walls['Wall_Left']  = {'position':(0,0),    'size':(t,H)}
                    self.walls['Wall_Right'] = {'position':(W-t,0),  'size':(t,H)}
                    self.walls['Wall_Top']   = {'position':(0,H-t),  'size':(W,t)}
                    self.walls['Wall_Bot1']  = {'position':(0,0),    'size':(gap_x, t)}
                    self.walls['Wall_Bot2']  = {'position':(gap_x+door_len,0),
                                                'size':(W-gap_x-door_len, t)}
                elif side == 'top':
                    gap_x = random.uniform(t, W - t - door_len)
                    self.door_position = (gap_x + door_len/2, H)
                    self.door_side = side
                    self.walls['Wall_Left']  = {'position':(0,0),    'size':(t,H)}
                    self.walls['Wall_Right'] = {'position':(W-t,0),  'size':(t,H)}
                    self.walls['Wall_Bot']   = {'position':(0,0),    'size':(W,t)}
                    self.walls['Wall_Top1']  = {'position':(0,H-t),  'size':(gap_x, t)}
                    self.walls['Wall_Top2']  = {'position':(gap_x+door_len,H-t),
                                                'size':(W-gap_x-door_len, t)}
                elif side == 'left':
                    gap_y = random.uniform(t, H - t - door_len)
                    self.door_position = (0, gap_y + door_len/2)
                    self.door_side = side
                    self.walls['Wall_Bot']   = {'position':(0,0),    'size':(W,t)}
                    self.walls['Wall_Top']   = {'position':(0,H-t),  'size':(W,t)}
                    self.walls['Wall_Right'] = {'position':(W-t,0),  'size':(t,H)}
                    self.walls['Wall_Left1'] = {'position':(0,gap_y+door_len),
                                                'size':(t, H-gap_y-door_len)}
                    self.walls['Wall_Left2'] = {'position':(0,0),    'size':(t,gap_y)}
                else:
                    gap_y = random.uniform(t, H - t - door_len)
                    self.door_position = (W, gap_y + door_len/2)
                    self.door_side = side
                    self.walls['Wall_Bot']    = {'position':(0,0),    'size':(W,t)}
                    self.walls['Wall_Top']    = {'position':(0,H-t),  'size':(W,t)}
                    self.walls['Wall_Left']   = {'position':(0,0),    'size':(t,H)}
                    self.walls['Wall_Right1'] = {'position':(W-t,0),  'size':(t,gap_y)}
                    self.walls['Wall_Right2'] = {'position':(W-t,gap_y+door_len),
                                                'size':(t, H-gap_y-door_len)}
            else:
                # Non-main floor: solid outer walls, no door, no entrance.
                # Keep the global floor-1 door intact for simulation.
                side = None
                self.walls['Wall_Left']  = {'position':(0,0),    'size':(t,H)}
                self.walls['Wall_Right'] = {'position':(W-t,0),  'size':(t,H)}
                self.walls['Wall_Bot']   = {'position':(0,0),    'size':(W,t)}
                self.walls['Wall_Top']   = {'position':(0,H-t),  'size':(W,t)}

            # Restore preserved connectors
            for nm, w in preserved_connectors.items():
                self.walls[nm] = w

            all_walls = []
            for name, wall in self.walls.items():
                if name.startswith("Section_"):
                    continue
                # Connectors are walkable for customers but still occupy layout
                # space; generated shelves/checkout/WC must not overlap them.
                x, y = wall['position']
                w, h = wall['size']
                all_walls.append((x, y, w, h))

            # b) Sections
            secs = list(sections)
            random.shuffle(secs)
            orientation = random.choice(['vertical','horizontal'])
            if orientation == 'vertical':
                container_w = (W - 2*t) / len(secs)
                container_h = H - 2*t
            else:
                container_w = W - 2*t
                container_h = (H - 2*t) / len(secs)

            self._layout_variation = (self._layout_variation + 1) % 5
            variation = self._layout_variation

            def check_collision(x, y, w, h, walls_list, items_list=None):
                for wx, wy, ww, wh in walls_list:
                    if not (x + w <= wx or x >= wx + ww or y + h <= wy or y >= wy + wh):
                        return True
                if items_list:
                    for ix, iy, iw, ih in items_list:
                        if not (x + w <= ix or x >= ix + iw or y + h <= iy or y >= iy + ih):
                            return True
                return False

            for idx, (sec_name, items) in enumerate(secs):
                if orientation == 'vertical':
                    x_sec = t + idx * container_w
                    y_sec = t
                else:
                    x_sec = t
                    y_sec = t + idx * container_h

                sec_color = "#{:02x}{:02x}{:02x}".format(
                    random.randint(150,255),
                    random.randint(150,255),
                    random.randint(150,255)
                )
                name_sec = f"Section_{sec_name}"
                self.walls[name_sec] = {
                    'position':  (x_sec, y_sec),
                    'size':      (container_w, container_h),
                    'color':     sec_color,
                    'label_loc': 'top'
                }

                section_walls = []
                num_parts = random.randint(1, 2)
                part_thick = t * 0.3

                for _ in range(num_parts):
                    orient_i = random.choice(['vertical', 'horizontal'])
                    wall_name = f"IW{self._iw_counter}"
                    self._iw_counter += 1

                    if orient_i == 'vertical':
                        x_div = random.uniform(x_sec + 0.2*container_w,
                                            x_sec + 0.8*container_w)
                        length = container_h * random.uniform(0.5, 0.8)
                        y_offset = y_sec + random.uniform(0.1*container_h,
                                                        0.3*container_h)
                        wall_pos = (x_div - part_thick/2, y_offset)
                        wall_size = (part_thick, length)
                        self.walls[wall_name] = {
                            'position': wall_pos, 'size': wall_size, 'color': 'black'
                        }
                        section_walls.append((wall_pos[0], wall_pos[1], wall_size[0], wall_size[1]))
                        all_walls.append((wall_pos[0], wall_pos[1], wall_size[0], wall_size[1]))

                        wall_start = x_div - part_thick/2
                        wall_end = x_div + part_thick/2
                        left_bound = (x_sec, y_sec, wall_start - x_sec, container_h)
                        right_bound = (wall_end, y_sec, x_sec + container_w - wall_end, container_h)

                        lw, rw = left_bound[2], right_bound[2]
                        items_copy = list(items)
                        if lw >= rw:
                            two_group = set(random.sample(items_copy, 2))
                            one_group = set(items_copy) - two_group
                            left_items, right_items = two_group, one_group
                        else:
                            two_group = set(random.sample(items_copy, 2))
                            one_group = set(items_copy) - two_group
                            left_items, right_items = one_group, two_group

                        for area_items, (ax, ay, aw, ah) in (
                                (left_items, left_bound),
                                (right_items, right_bound)):
                            placed = []
                            for nm in area_items:
                                size_multiplier = random.uniform(0.5, 0.8) if random.random() < 0.6 else random.uniform(0.8, 1.2)
                                aspect_ratio = random.uniform(0.5, 2.0)
                                base_size = min(aw, ah) * size_multiplier
                                item_w = min(base_size * (2/aspect_ratio if aspect_ratio > 1 else 1), aw * 0.95)
                                item_h = min(base_size * (aspect_ratio if aspect_ratio < 1 else 1), ah * 0.95)
                                found_position = False
                                pad = 0.05
                                for _ in range(50):
                                    nx = random.uniform(ax + pad, ax + aw - item_w - pad)
                                    ny = random.uniform(ay + pad, ay + ah - item_h - pad)
                                    if not check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                        found_position = True
                                        self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                        placed.append((nx, ny, item_w, item_h))
                                        break
                                if not found_position:
                                    grid_size = 6
                                    for i in range(grid_size):
                                        for j in range(grid_size):
                                            nx = ax + pad + (aw - item_w - 2*pad) * i / (grid_size-1)
                                            ny = ay + pad + (ah - item_h - 2*pad) * j / (grid_size-1)
                                            if not check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                                found_position = True
                                                self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                                placed.append((nx, ny, item_w, item_h))
                                                break
                                        if found_position:
                                            break
                                if not found_position:
                                    item_w *= 0.7; item_h *= 0.7
                                    corners = [
                                        (ax + pad, ay + pad),
                                        (ax + aw - item_w - pad, ay + pad),
                                        (ax + pad, ay + ah - item_h - pad),
                                        (ax + aw - item_w - pad, ay + ah - item_h - pad)
                                    ]
                                    for nx, ny in corners:
                                        if not check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                            found_position = True
                                            self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                            placed.append((nx, ny, item_w, item_h))
                                            break
                                if not found_position:
                                    item_w = min(aw * 0.15, 0.5)
                                    item_h = min(ah * 0.15, 0.5)
                                    nx = ax + aw/2 - item_w/2
                                    ny = ay + ah/2 - item_h/2
                                    if check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                        safe_found = False
                                        for i in range(10):
                                            for j in range(10):
                                                test_x = ax + (aw - item_w) * i / 9
                                                test_y = ay + (ah - item_h) * j / 9
                                                if not check_collision(test_x, test_y, item_w, item_h, all_walls, placed):
                                                    nx, ny = test_x, test_y
                                                    safe_found = True
                                                    break
                                            if safe_found:
                                                break
                                        if not safe_found:
                                            item_w *= 0.5; item_h *= 0.5
                                            nx = ax + pad; ny = ay + pad
                                    self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                    placed.append((nx, ny, item_w, item_h))
                    else:
                        y_div = random.uniform(y_sec + 0.2*container_h,
                                            y_sec + 0.8*container_h)
                        length = container_w * random.uniform(0.5, 0.8)
                        x_offset = x_sec + random.uniform(0.1*container_w,
                                                        0.3*container_w)
                        wall_pos = (x_offset, y_div - part_thick/2)
                        wall_size = (length, part_thick)
                        self.walls[wall_name] = {
                            'position': wall_pos, 'size': wall_size, 'color': 'black'
                        }
                        section_walls.append((wall_pos[0], wall_pos[1], wall_size[0], wall_size[1]))
                        all_walls.append((wall_pos[0], wall_pos[1], wall_size[0], wall_size[1]))

                        wall_start = y_div - part_thick/2
                        wall_end = y_div + part_thick/2
                        bottom_bound = (x_sec, y_sec, container_w, wall_start - y_sec)
                        top_bound = (x_sec, wall_end, container_w, y_sec + container_h - wall_end)

                        bh, th = bottom_bound[3], top_bound[3]
                        items_copy = list(items)
                        if bh >= th:
                            two_group = set(random.sample(items_copy, 2))
                            one_group = set(items_copy) - two_group
                            bottom_items, top_items = two_group, one_group
                        else:
                            two_group = set(random.sample(items_copy, 2))
                            one_group = set(items_copy) - two_group
                            bottom_items, top_items = one_group, two_group

                        for area_items, (ax, ay, aw, ah) in (
                                (bottom_items, bottom_bound),
                                (top_items, top_bound)):
                            placed = []
                            pad = 0.05
                            for nm in area_items:
                                size_multiplier = random.uniform(0.5, 0.8) if random.random() < 0.6 else random.uniform(0.8, 1.2)
                                aspect_ratio = random.uniform(0.5, 2.0)
                                base_size = min(aw, ah) * size_multiplier
                                item_w = min(base_size * (2/aspect_ratio if aspect_ratio > 1 else 1), aw * 0.95)
                                item_h = min(base_size * (aspect_ratio if aspect_ratio < 1 else 1), ah * 0.95)
                                found_position = False
                                for _ in range(50):
                                    nx = random.uniform(ax + pad, ax + aw - item_w - pad)
                                    ny = random.uniform(ay + pad, ay + ah - item_h - pad)
                                    if not check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                        found_position = True
                                        self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                        placed.append((nx, ny, item_w, item_h))
                                        break
                                if not found_position:
                                    grid_size = 6
                                    for i in range(grid_size):
                                        for j in range(grid_size):
                                            nx = ax + pad + (aw - item_w - 2*pad) * i / (grid_size-1)
                                            ny = ay + pad + (ah - item_h - 2*pad) * j / (grid_size-1)
                                            if not check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                                found_position = True
                                                self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                                placed.append((nx, ny, item_w, item_h))
                                                break
                                        if found_position:
                                            break
                                if not found_position:
                                    item_w *= 0.7; item_h *= 0.7
                                    corners = [
                                        (ax + pad, ay + pad),
                                        (ax + aw - item_w - pad, ay + pad),
                                        (ax + pad, ay + ah - item_h - pad),
                                        (ax + aw - item_w - pad, ay + ah - item_h - pad)
                                    ]
                                    for nx, ny in corners:
                                        if not check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                            found_position = True
                                            self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                            placed.append((nx, ny, item_w, item_h))
                                            break
                                if not found_position:
                                    item_w = min(aw * 0.15, 0.5)
                                    item_h = min(ah * 0.15, 0.5)
                                    nx = ax + aw/2 - item_w/2
                                    ny = ay + ah/2 - item_h/2
                                    if check_collision(nx, ny, item_w, item_h, all_walls, placed):
                                        safe_found = False
                                        for i in range(10):
                                            for j in range(10):
                                                test_x = ax + (aw - item_w) * i / 9
                                                test_y = ay + (ah - item_h) * j / 9
                                                if not check_collision(test_x, test_y, item_w, item_h, all_walls, placed):
                                                    nx, ny = test_x, test_y
                                                    safe_found = True
                                                    break
                                            if safe_found:
                                                break
                                        if not safe_found:
                                            item_w *= 0.5; item_h *= 0.5
                                            nx = ax + pad; ny = ay + pad
                                    self.items[nm] = {'position':(nx,ny),'size':(item_w,item_h),'category':sec_name}
                                    placed.append((nx, ny, item_w, item_h))

                if not section_walls:
                    placed_items = []
                    count = len(items)
                    if variation == 0:
                        rows = math.ceil(math.sqrt(count))
                        cols = math.ceil(count / rows)
                        cw = container_w / cols
                        ch = container_h / rows
                        for i, itm in enumerate(items):
                            r, c = divmod(i, cols)
                            size_mult = random.uniform(0.85, 0.98) if random.random() < 0.4 else random.uniform(0.7, 0.85)
                            aspect_ratio = random.uniform(0.6, 1.7)
                            base_size = min(cw, ch) * size_mult
                            iw = min(base_size * (2/aspect_ratio if aspect_ratio > 1 else 1), cw * 0.95)
                            ih = min(base_size * (aspect_ratio if aspect_ratio < 1 else 1), ch * 0.95)
                            nx = x_sec + c*cw + (cw - iw)/2
                            ny = y_sec + r*ch + (ch - ih)/2
                            if check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                found_safe = False
                                pad = 0.05
                                for ox in range(-2, 3):
                                    for oy in range(-2, 3):
                                        tx = nx + ox*0.1; ty = ny + oy*0.1
                                        if (tx >= x_sec+pad and tx+iw <= x_sec+container_w-pad
                                            and ty >= y_sec+pad and ty+ih <= y_sec+container_h-pad):
                                            if not check_collision(tx, ty, iw, ih, all_walls, placed_items):
                                                nx, ny = tx, ty
                                                found_safe = True
                                                break
                                    if found_safe:
                                        break
                                if not found_safe:
                                    iw *= 0.7; ih *= 0.7
                                    nx = x_sec + c*cw + (cw - iw)/2
                                    ny = y_sec + r*ch + (ch - ih)/2
                                    if check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                        iw *= 0.7; ih *= 0.7
                            self.items[itm] = {'position':(nx,ny),'size':(iw,ih),'category':sec_name}
                            placed_items.append((nx, ny, iw, ih))
                    elif variation == 1:
                        cw = container_w / count
                        for j, itm in enumerate(items):
                            size_mult = random.uniform(0.7, 0.9)
                            aspect_ratio = random.uniform(0.6, 1.4)
                            iw = min(cw * 0.95, cw * size_mult * (1.5/aspect_ratio))
                            ih = min(container_h * 0.7, container_h * size_mult * aspect_ratio * 0.7)
                            nx = x_sec + j*cw + (cw - iw)/2
                            ny = y_sec + (container_h - ih)/2
                            if check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                found_safe = False
                                for y_pos in [0.25, 0.4, 0.6, 0.75]:
                                    test_y = y_sec + container_h * y_pos
                                    if not check_collision(nx, test_y, iw, ih, all_walls, placed_items):
                                        ny = test_y; found_safe = True; break
                                if not found_safe:
                                    iw *= 0.7; ih *= 0.7
                            self.items[itm] = {'position':(nx,ny),'size':(iw,ih),'category':sec_name}
                            placed_items.append((nx, ny, iw, ih))
                    elif variation == 2:
                        ch = container_h / count
                        for j, itm in enumerate(items):
                            size_mult = random.uniform(0.7, 0.9)
                            aspect_ratio = random.uniform(0.8, 1.8)
                            iw = min(container_w * 0.7, container_w * size_mult * 0.7)
                            ih = min(ch * 0.95, ch * size_mult * aspect_ratio)
                            nx = x_sec + (container_w - iw)/2
                            ny = y_sec + j*ch + (ch - ih)/2
                            if check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                found_safe = False
                                for x_pos in [0.25, 0.4, 0.6, 0.75]:
                                    test_x = x_sec + container_w * x_pos
                                    if not check_collision(test_x, ny, iw, ih, all_walls, placed_items):
                                        nx = test_x; found_safe = True; break
                                if not found_safe:
                                    iw *= 0.7; ih *= 0.7
                            self.items[itm] = {'position':(nx,ny),'size':(iw,ih),'category':sec_name}
                            placed_items.append((nx, ny, iw, ih))
                    elif variation == 3:
                        pad = 0.05
                        for itm in items:
                            size_mult = random.uniform(0.45, 0.85)
                            aspect_ratio = random.uniform(0.5, 2.0)
                            base_size = min(container_w, container_h) * size_mult
                            iw = min(base_size * (2/aspect_ratio if aspect_ratio > 1 else 1), container_w * 0.5)
                            ih = min(base_size * (aspect_ratio if aspect_ratio < 1 else 1), container_h * 0.5)
                            placed = False
                            nx = ny = 0
                            for _ in range(80):
                                nx = random.uniform(x_sec + pad, x_sec + container_w - iw - pad)
                                ny = random.uniform(y_sec + pad, y_sec + container_h - ih - pad)
                                if not check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                    placed = True; break
                            if not placed:
                                iw *= 0.7; ih *= 0.7
                                for _ in range(50):
                                    nx = random.uniform(x_sec + pad, x_sec + container_w - iw - pad)
                                    ny = random.uniform(y_sec + pad, y_sec + container_h - ih - pad)
                                    if not check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                        placed = True; break
                            if not placed:
                                nx = x_sec + container_w/2 - iw/2
                                ny = y_sec + container_h/2 - ih/2
                            self.items[itm] = {'position':(nx,ny),'size':(iw,ih),'category':sec_name}
                            placed_items.append((nx, ny, iw, ih))
                    else:
                        cx, cy = x_sec + container_w/2, y_sec + container_h/2
                        radius = min(container_w, container_h) * 0.25
                        for j, itm in enumerate(items):
                            size_mult = random.uniform(0.25, 0.4)
                            aspect_ratio = random.uniform(0.7, 1.5)
                            base_size = min(container_w, container_h) * size_mult
                            iw = base_size * (1.5/aspect_ratio if aspect_ratio > 1 else 1)
                            ih = base_size * (aspect_ratio if aspect_ratio < 1 else 1)
                            angle = 2*math.pi * j / count
                            nx = cx + math.cos(angle)*radius - iw/2
                            ny = cy + math.sin(angle)*radius - ih/2
                            if check_collision(nx, ny, iw, ih, all_walls, placed_items):
                                found_safe = False
                                for r_mult in [0.9, 0.8, 0.7, 0.6, 1.1, 1.2]:
                                    tx = cx + math.cos(angle)*radius*r_mult - iw/2
                                    ty = cy + math.sin(angle)*radius*r_mult - ih/2
                                    if not check_collision(tx, ty, iw, ih, all_walls, placed_items):
                                        nx, ny = tx, ty; found_safe = True; break
                                if not found_safe:
                                    iw *= 0.7; ih *= 0.7
                                    nx = cx + math.cos(angle)*radius - iw/2
                                    ny = cy + math.sin(angle)*radius - ih/2
                            self.items[itm] = {'position':(nx,ny),'size':(iw,ih),'category':sec_name}
                            placed_items.append((nx, ny, iw, ih))

            # c) Checkout & WC – ONLY for main floor
            if is_main_floor:
                cw = min(2.0, W * 0.2)
                ch = min(1.0, H * 0.1)
                margin = 0.1

                all_items = []
                for item in self.items.values():
                    ix, iy = item['position']
                    iw, ih = item['size']
                    all_items.append((ix, iy, iw, ih))

                checkout_placed = False
                cx = cy = t
                if side in ('bottom', 'top'):
                    ref = gap_x
                    for attempt in range(10):
                        cx = ref - cw - margin if random.choice([True, False]) else ref + door_len + margin
                        cx = max(t, min(cx, W - t - cw))
                        cy = t if side == 'bottom' else (H - t - ch)
                        if not check_collision(cx, cy, cw, ch, all_walls, all_items):
                            checkout_placed = True; break
                        ref += random.uniform(-0.5, 0.5)
                else:
                    ref = gap_y
                    for attempt in range(10):
                        cy = ref - ch - margin if random.choice([True, False]) else ref + door_len + margin
                        cy = max(t, min(cy, H - t - ch))
                        cx = t if side == 'left' else (W - t - cw)
                        if not check_collision(cx, cy, cw, ch, all_walls, all_items):
                            checkout_placed = True; break
                        ref += random.uniform(-0.5, 0.5)

                if not checkout_placed:
                    positions = []
                    if side in ('bottom', 'top'):
                        y_pos = t if side == 'bottom' else (H - t - ch)
                        for x_pos in [t, W/4 - cw/2, W/2 - cw/2, 3*W/4 - cw/2, W - t - cw]:
                            positions.append((x_pos, y_pos))
                    else:
                        x_pos = t if side == 'left' else (W - t - cw)
                        for y_pos in [t, H/4 - ch/2, H/2 - ch/2, 3*H/4 - ch/2, H - t - ch]:
                            positions.append((x_pos, y_pos))
                    for cx_, cy_ in positions:
                        if not check_collision(cx_, cy_, cw, ch, all_walls, all_items):
                            cx, cy = cx_, cy_
                            checkout_placed = True; break

                if not checkout_placed:
                    cw *= 0.7; ch *= 0.7
                    if side in ('bottom', 'top'):
                        cx = t
                        cy = t if side == 'bottom' else (H - t - ch)
                    else:
                        cx = t if side == 'left' else (W - t - cw)
                        cy = t

                self.walls['Checkout'] = {'position': (cx, cy), 'size': (cw, ch)}
                all_walls.append((cx, cy, cw, ch))

                # Place WC
                wc_w = cw; wc_h = ch
                wc_placed = False
                wc_x = wc_y = t
                corners = [(t, t), (W - t - wc_w, t), (t, H - t - wc_h), (W - t - wc_w, H - t - wc_h)]
                for cx_, cy_ in corners:
                    if not check_collision(cx_, cy_, wc_w, wc_h, all_walls, all_items):
                        wc_x, wc_y = cx_, cy_
                        wc_placed = True; break

                if not wc_placed:
                    positions = []
                    for x_pos in [t, W/4, W/2, 3*W/4, W - t - wc_w]:
                        positions.append((x_pos, t))
                        positions.append((x_pos, H - t - wc_h))
                    for y_pos in [t, H/4, H/2, 3*H/4, H - t - wc_h]:
                        positions.append((t, y_pos))
                        positions.append((W - t - wc_w, y_pos))
                    for cx_, cy_ in positions:
                        if not check_collision(cx_, cy_, wc_w, wc_h, all_walls, all_items):
                            wc_x, wc_y = cx_, cy_
                            wc_placed = True; break

                if not wc_placed:
                    wc_w *= 0.7; wc_h *= 0.7
                    wc_x, wc_y = t, H - t - wc_h
                    if check_collision(wc_x, wc_y, wc_w, wc_h, all_walls, all_items):
                        for i in range(10):
                            for j in range(10):
                                test_x = t + (W - 2*t - wc_w) * i / 9
                                test_y = t + (H - 2*t - wc_h) * j / 9
                                if not check_collision(test_x, test_y, wc_w, wc_h, all_walls, all_items):
                                    wc_x, wc_y = test_x, test_y
                                    wc_placed = True; break
                            if wc_placed:
                                break
                        if not wc_placed:
                            wc_w *= 0.5; wc_h *= 0.5
                            wc_x, wc_y = W - t - wc_w, t

                self.walls['WC'] = {'position': (wc_x, wc_y), 'size': (wc_w, wc_h)}

            # d) Occupancy check
            total_area = W * H
            iarea = sum(w*h for w,h in (it['size'] for it in self.items.values()))
            warea = sum(w*h for w,h in (w['size'] for w in self.walls.values()))
            if (iarea + warea) / total_area >= threshold:
                break
            if outer_attempt >= max_outer_attempts:
                break  # accept current layout to avoid infinite loop

        # Impulse Purchases — ONLY main floor (needs Checkout)
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