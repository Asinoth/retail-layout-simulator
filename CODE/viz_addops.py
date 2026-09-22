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


class AddOpsMixin:
    """Add/remove items, walls, sections, entrance, checkout, WC, impulse + zoom controls."""

    def add_item(self, name, position, size, category):
         """Validate + insert an item, then give it a random default price."""
         x, y = position
         w, h = size
         # bounds check
         if x < 0 or y < 0 or w <= 0 or h <= 0 \
            or x + w > self.width or y + h > self.height:
             messagebox.showerror(
                 "Error", "Item must fit within shop bounds.", parent=self.tk_root
             )
             return

         # 1) Insert into model (no 'price' field here)
         self.items[name] = {
             "position": (x, y),
             "size":     (w, h),
             "category": category
         }

         # 2) Assign a default random price (you can tweak range)
         self._assign_default_prices()

         # 3) Redraw
         self._invalidate_sim_geometry()
         self.redraw()

    def remove_item(self, name):
        self.items.pop(name, None)
        self._invalidate_sim_geometry()
        self.redraw()

    def add_wall(self, name, position, size):
        """
        Add a new wall at the given position and size.
        Walls may overlap other walls but must not overlap items.
        """
        x, y = position
        w, h = size

        # 1) Bounds check
        if x < 0 or y < 0 or w <= 0 or h <= 0 \
        or x + w > self.width or y + h > self.height:
            messagebox.showerror(
                "Error",
                "Wall position and size must fit within shop bounds.",
                parent=self.tk_root
            )
            return

        # 2) Prevent overlapping any existing item
        for item_name, item_data in self.items.items():
            ix, iy = item_data['position']
            iw, ih = item_data['size']
            if not (x + w <= ix or x >= ix + iw or y + h <= iy or y >= iy + ih):
                messagebox.showerror(
                    "Error",
                    f"Wall '{name}' would overlap item '{item_name}'.",
                    parent=self.tk_root
                )
                return

        # 3) All clear -> add the new wall
        self.walls[name] = {
            "position": (x, y),
            "size":     (w, h)
        }
        self._invalidate_sim_geometry()
        self.redraw()

    def remove_wall(self, name):
        data = self.walls.get(name)
        connector_id = data.get('connector_id') if isinstance(data, dict) else None
        mapping = (getattr(self, 'connectors', None) or {}).get(connector_id) \
            if connector_id else None
        if mapping and mapping.get(self.current_floor) == name:
            self.connectors.pop(connector_id, None)
            for fid, wname in list(mapping.items()):
                try:
                    self.floors[fid]['walls'].pop(wname, None)
                except Exception:
                    pass
        elif connector_id:
            # Connector ids are reused after a delete, so this wall's id may
            # now name a different connector. Remove only the walls sharing
            # both this name and id, and leave that connector's mapping and
            # its own walls (which may carry the same name) alone.
            for fid, fdata in self.floors.items():
                if mapping and mapping.get(fid) == name:
                    continue
                w = fdata.get('walls', {}).get(name)
                if isinstance(w, dict) and w.get('connector_id') == connector_id:
                    fdata['walls'].pop(name, None)
        self.walls.pop(name, None)
        self._invalidate_sim_geometry()
        self.redraw()

    def _invalidate_sim_geometry(self):
        # The live loop rebuilds its path grid and wall bins only when
        # geometry_dirty is set, and the zone table only when cleared, so
        # every edit to an item, wall or section must flag both: items are
        # obstacles too, so agents would otherwise walk through fixtures that
        # moved and be attributed to stale zones.
        sim = getattr(self, 'customer_simulation', None)
        if sim is not None:
            sim.geometry_dirty = True
            sim.invalidate_zones_cache()


    def _on_add_item_clicked(self):
        cat = simpledialog.askstring("Add Item", "Category:", parent=self.tk_root)
        if not cat:
            return
        cat = cat.strip()
        name = cat
        if name in self.items or name in self.walls:
            i = 2
            name = f"{cat}_{i}"
            while name in self.items or name in self.walls:
                i += 1
                name = f"{cat}_{i}"
        x = simpledialog.askfloat("Add Item", "x (m):", parent=self.tk_root)
        y = simpledialog.askfloat("Add Item", "y (m):", parent=self.tk_root)
        w = simpledialog.askfloat("Add Item", "width (m):", parent=self.tk_root)
        h = simpledialog.askfloat("Add Item", "height (m):", parent=self.tk_root)
        if None in (x, y, w, h):
            return
        self.add_item(name, (x, y), (w, h), cat)

    def _on_add_wall_clicked(self):
        x = simpledialog.askfloat("Add Wall", "x (m):", parent=self.tk_root)
        y = simpledialog.askfloat("Add Wall", "y (m):", parent=self.tk_root)
        w = simpledialog.askfloat("Add Wall", "width (m):", parent=self.tk_root)
        h = simpledialog.askfloat("Add Wall", "height (m):", parent=self.tk_root)
        if None in (x, y, w, h):
            return
        # default wall naming: always start with "Wall", append a numeric if you need it
        base = "Wall"
        name = base
        idx = 2
        while name in self.walls or name in self.items:
            name = f"{base}{idx}"
            idx += 1
        self.add_wall(name, (x, y), (w, h))

    def _on_zoom_in_clicked(self):
        old_z = self.zoom
        new_z = round(old_z + 0.1, 2)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._apply_zoom(new_z, (x0 + x1) / 2, (y0 + y1) / 2)

    def _on_zoom_out_clicked(self):
        old_z = self.zoom
        new_z = max(1.0, round(old_z - 0.1, 2))
        if new_z == old_z:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._apply_zoom(new_z, (x0 + x1) / 2, (y0 + y1) / 2)

    def _apply_zoom(self, new_z, cx, cy):
        new_z = max(1.0, new_z)
        old_z = self.zoom
        if new_z == old_z:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        if cx is None:
            cx = (x0 + x1) / 2
        if cy is None:
            cy = (y0 + y1) / 2
        rx = (cx - x0) / (x1 - x0) if (x1 - x0) else 0.5
        ry = (cy - y0) / (y1 - y0) if (y1 - y0) else 0.5
        new_w = self.width / new_z
        new_h = self.height / new_z
        new_x0 = cx - rx * new_w
        new_x1 = cx + (1 - rx) * new_w
        new_y0 = cy - ry * new_h
        new_y1 = cy + (1 - ry) * new_h

        # Clamp to shop bounds so no blue corners leak outside
        if new_x0 < 0:
            new_x1 -= new_x0
            new_x0 = 0
        if new_x1 > self.width:
            new_x0 -= (new_x1 - self.width)
            new_x1 = self.width
        if new_y0 < 0:
            new_y1 -= new_y0
            new_y0 = 0
        if new_y1 > self.height:
            new_y0 -= (new_y1 - self.height)
            new_y1 = self.height

        # Final safety clamp (if viewport > shop size, just reset to full)
        new_x0 = max(new_x0, 0)
        new_y0 = max(new_y0, 0)
        new_x1 = min(new_x1, self.width)
        new_y1 = min(new_y1, self.height)

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.zoom = new_z
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")
        self.canvas.draw_idle()

    def _on_zoom_label_clicked(self):
        val = simpledialog.askinteger(
            "Custom Zoom",
            "Enter zoom percentage (100% minimum):",
            initialvalue=int(self.zoom * 100),
            minvalue=100,
            maxvalue=1000,
            parent=self.tk_root
        )
        if val is None:
            return
        new_z = max(1.0, val / 100.0)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._apply_zoom(new_z, (x0 + x1) / 2, (y0 + y1) / 2)



    def _on_add_section_clicked(self):
        """Prompt user to draw a new section grouping rectangle."""
        name = simpledialog.askstring("Add Section", "Section Name:", parent=self.tk_root)
        if not name:
            return
        sec_name = f"Section_{name}"
        idx = 2
        while sec_name in self.walls or sec_name in self.items:
            sec_name = f"Section_{name}_{idx}"
            idx += 1

        x = simpledialog.askfloat("Add Section", "X (m):", parent=self.tk_root)
        y = simpledialog.askfloat("Add Section", "Y (m):", parent=self.tk_root)
        w = simpledialog.askfloat("Add Section", "Width (m):", parent=self.tk_root)
        h = simpledialog.askfloat("Add Section", "Height (m):", parent=self.tk_root)
        if None in (x, y, w, h):
            return

        # Prevent overlapping other sections
        for existing_name, existing in self.walls.items():
            if not existing_name.startswith("Section_"):
                continue
            ex, ey = existing['position']
            ew, eh = existing['size']
            if not (x + w <= ex or x >= ex + ew or y + h <= ey or y >= ey + eh):
                messagebox.showerror(
                    "Error",
                    f"Section '{sec_name}' would overlap existing section '{existing_name}'.",
                    parent=self.tk_root
                )
                return

        # Add the new section (can overlap walls/items if desired)
        self.walls[sec_name] = {
            "position":  (x, y),
            "size":      (w, h),
            "color":     "#cccccc",
            "textcolor": "cyan",
            "fontsize":  14,
            "font":      None,
            "label_loc": "top"
        }
        self._invalidate_sim_geometry()
        self.redraw()

 
   
    def _on_add_entrance_clicked(self):
        """Prompt for entrance point (x,y) and update door_position."""
        x = simpledialog.askfloat("Entrance X", "Enter entrance x (m):", parent=self.tk_root)
        y = simpledialog.askfloat("Entrance Y", "Enter entrance y (m):", parent=self.tk_root)
        if None in (x, y):
            return

        # Store the exact entrance coordinate
        self.door_position = (x, y)

        # Try to infer which side of the shop it's on (optional)
        if abs(y - 0) < 1e-6:
            side = 'bottom'
        elif abs(y - self.height) < 1e-6:
            side = 'top'
        elif abs(x - 0) < 1e-6:
            side = 'left'
        elif abs(x - self.width) < 1e-6:
            side = 'right'
        else:
            side = None
        self.door_side = side
        try:
            self.floors.setdefault(1, {'items': {}, 'walls': {}, 'prices': {}})
            self.floors[1]['is_main_floor'] = True
            self.floors[1]['door_position'] = tuple(self.door_position)
            self.floors[1]['door_side'] = self.door_side
            if hasattr(self, 'customer_simulation'):
                self.customer_simulation.door_position = self.door_position
                self.customer_simulation.door_side = self.door_side
        except Exception:
            pass

        messagebox.showinfo(
            "Entrance Updated",
            f"Entrance set at {self.door_position}"
            + (f" on {self.door_side} side." if side else "."),
            parent=self.tk_root
        )



    def _on_add_checkout_clicked(self):
        """Define checkout by designating an existing item as the checkout counter."""
        # 1) Ask user for the desired checkout centre coordinates
        cx = simpledialog.askfloat(
            "Checkout X",
            "Enter the X coordinate for the checkout centre:",
            parent=self.tk_root
        )
        if cx is None:
            return

        cy = simpledialog.askfloat(
            "Checkout Y",
            "Enter the Y coordinate for the checkout centre:",
            parent=self.tk_root
        )
        if cy is None:
            return

        # 2) Find an existing item exactly under that point
        found = None
        for name, data in self.items.items():
            px, py = data['position']
            w, h   = data['size']
            if px <= cx <= px + w and py <= cy <= py + h:
                found = name
                break

        # 3) If not found, pick nearest and confirm with the user
        if not found:
            best, best_d = None, float('inf')
            for name, data in self.items.items():
                px, py = data['position']
                w, h   = data['size']
                center_x = px + w/2
                center_y = py + h/2
                d = math.hypot(center_x - cx, center_y - cy)
                if d < best_d:
                    best, best_d = name, d
            if best:
                use = messagebox.askyesno(
                    "Confirm Checkout Item",
                    f"No item at ({cx:.2f},{cy:.2f}).\n"
                    f"Nearest item is '{best}' ({best_d:.2f} m away).\n"
                    "Use this as the checkout?",
                    parent=self.tk_root
                )
                if not use:
                    return
                found = best
            else:
                messagebox.showerror(
                    "No Items Found",
                    "There are no items in the layout to assign as checkout.",
                    parent=self.tk_root
                )
                return
        else:
            use = messagebox.askyesno(
                "Confirm Checkout Item",
                f"Use '{found}' as the checkout counter?",
                parent=self.tk_root
            )
            if not use:
                return

        # 4) Remove the selected item from items for promotion
        data = self.items.pop(found)
        orig_x, orig_y = data['position']
        w_item, h_item = data['size']

        # 5) Compute candidate top-left so the rectangle is centred on (cx, cy)
        new_x = cx - w_item/2
        new_y = cy - h_item/2
        # clamp into shop bounds
        new_x = max(0, min(new_x, self.width  - w_item))
        new_y = max(0, min(new_y, self.height - h_item))

        # 6) A generated layout has a checkout BANK: 'Checkout' plus one
        #    'Checkout_LaneN' counter per extra lane, and customers queue at
        #    whichever is least busy. Replacing only the primary counter
        #    would leave the old lanes serving customers across the store
        #    from the counter the user just defined.
        stale_lanes = [nm for nm in self.walls if nm.startswith('Checkout_Lane')]
        if stale_lanes:
            drop = messagebox.askyesno(
                "Replace Checkout Bank",
                f"This layout has {len(stale_lanes)} extra checkout lane(s).\n"
                "Remove them so the new counter is the only checkout?",
                parent=self.tk_root
            )
            if drop:
                for nm in stale_lanes:
                    self.walls.pop(nm, None)

        # 7) Only move to (new_x,new_y) if it doesn't overlap walls or items.
        #    The counter being replaced is not an obstacle for its successor,
        #    so it comes out before the test; step 8 writes the new one back
        #    under the same name on either branch.
        self.walls.pop('Checkout', None)
        if self._is_position_safe(new_x, new_y, w_item, h_item):
            x_item, y_item = new_x, new_y
        else:
            messagebox.showwarning(
                "Overlap Detected",
                "Desired checkout position overlaps an existing wall or item.\n"
                "Using the item's original position instead.",
                parent=self.tk_root
            )
            x_item, y_item = orig_x, orig_y

        # 8) Register the Checkout area at the chosen location
        self.walls['Checkout'] = {
            'position': (x_item, y_item),
            'size':     (w_item, h_item),
            'category': 'Checkout'
        }

        self._invalidate_sim_geometry()

        # 9) Redraw to reflect the new checkout
        self.redraw()



    def _on_add_wc_clicked(self):
        """Define WC by designating an existing item as the restroom area."""
        # 1) Ask user for WC centre
        wx = simpledialog.askfloat(
            "WC X",
            "Enter the X coordinate for the WC centre:",
            parent=self.tk_root
        )
        if wx is None:
            return

        wy = simpledialog.askfloat(
            "WC Y",
            "Enter the Y coordinate for the WC centre:",
            parent=self.tk_root
        )
        if wy is None:
            return

        # 2) Find an existing item under that point
        found = None
        for name, data in self.items.items():
            px, py = data['position']
            w, h   = data['size']
            if px <= wx <= px + w and py <= wy <= py + h:
                found = name
                break

        # 3) If not found, pick nearest and confirm
        if not found:
            best, best_d = None, float('inf')
            for name, data in self.items.items():
                px, py = data['position']
                w, h   = data['size']
                cx = px + w/2
                cy = py + h/2
                d = math.hypot(cx - wx, cy - wy)
                if d < best_d:
                    best, best_d = name, d
            if not best:
                messagebox.showerror(
                    "No Items Found",
                    "There are no items to assign as WC.",
                    parent=self.tk_root
                )
                return
            use = messagebox.askyesno(
                "Confirm WC Area",
                f"No item at ({wx:.2f},{wy:.2f}).\n"
                f"Nearest item is '{best}' ({best_d:.2f} m away).\n"
                "Use this as the restroom?",
                parent=self.tk_root
            )
            if not use:
                return
            found = best
        else:
            use = messagebox.askyesno(
                "Confirm WC Area",
                f"Use '{found}' as the restroom area?",
                parent=self.tk_root
            )
            if not use:
                return

        # 4) Remove item from self.items and promote to WC wall
        data = self.items.pop(found)
        x_item, y_item = data['position']
        w_item, h_item = data['size']

        self.walls['WC'] = {
            'position': (x_item, y_item),
            'size':     (w_item, h_item)
        }

        self._invalidate_sim_geometry()

        # 5) Redraw so we see the new WC, never re-adding to self.items
        self.redraw()


    
    def _on_add_impulse_items_clicked(self):
        """
        Let the user pick existing items by entering X/Y.
        Ensures each is near the Checkout (within 3m), offers to move it if not,
        then marks it as category 'Impulse' with a gold colour.
        """
        # 1) How many to mark?
        count = simpledialog.askinteger(
            "Impulse Items",
            "How many impulse items to define?",
            parent=self.tk_root,
            minvalue=1, maxvalue=20
        )
        if not count:
            return

        # 2) Ensure Checkout exists
        if 'Checkout' not in self.walls:
            messagebox.showerror(
                "Missing Checkout",
                "Please define a Checkout area first.",
                parent=self.tk_root
            )
            return

        # Precompute checkout centre and size
        co_pos = self.walls['Checkout']['position']
        co_size = self.walls['Checkout']['size']
        cx_co = co_pos[0] + co_size[0] / 2
        cy_co = co_pos[1] + co_size[1] / 2

        threshold = 3.0   # max allowed distance (m)
        margin    = 0.1   # when auto-moving

        for i in range(count):
            # 3) Ask for coords
            x = simpledialog.askfloat(
                "Impulse X",
                f"Enter X (m) for impulse item #{i+1}:",
                parent=self.tk_root
            )
            y = simpledialog.askfloat(
                "Impulse Y",
                f"Enter Y (m) for impulse item #{i+1}:",
                parent=self.tk_root
            )
            if x is None or y is None:
                break

            # 4) Find the item under (x,y) or nearest one
            found = None
            for name, data in self.items.items():
                px, py = data['position']
                w, h   = data['size']
                if px <= x <= px+w and py <= y <= py+h:
                    found = name
                    break

            if not found:
                # pick nearest by centre
                best, best_d = None, float('inf')
                for name, data in self.items.items():
                    px, py = data['position']
                    w, h   = data['size']
                    cx = px + w/2
                    cy = py + h/2
                    d = math.hypot(cx - x, cy - y)
                    if d < best_d:
                        best_d, best = d, name
                if best:
                    use = messagebox.askyesno(
                        "Use Nearest Item",
                        f"No item at ({x:.2f},{y:.2f}).\n"
                        f"Nearest is '{best}' at {best_d:.2f} m.\n"
                        "Use this?",
                        parent=self.tk_root
                    )
                    if use:
                        found = best

            if not found:
                continue

            # 5) Check distance from checkout centre
            itm = self.items[found]
            px, py = itm['position']
            w, h   = itm['size']
            cx_it  = px + w/2
            cy_it  = py + h/2
            dist   = math.hypot(cx_it - cx_co, cy_it - cy_co)

            if dist > threshold:
                move = messagebox.askyesno(
                    "Item Too Far",
                    f"'{found}' is {dist:.2f} m from checkout.\n"
                    "Move it next to checkout?",
                    parent=self.tk_root
                )
                if move:
                    # position just to the right of checkout by default
                    new_x = cx_co + co_size[0]/2 + margin
                    new_y = cy_co - h/2
                    # if that runs off the right edge, put it on the left
                    if new_x + w > self.width:
                        new_x = cx_co - co_size[0]/2 - margin - w
                    # clamp into the shop
                    new_x = max(0, min(new_x, self.width  - w))
                    new_y = max(0, min(new_y, self.height - h))
                    # Every relocated item shares this target, so look for a
                    # free spot near it. _find_safe_position returns the
                    # item's current position when nothing nearby is free.
                    pos = self._find_safe_position(new_x, new_y, (w, h),
                                                   exclude_item=found)
                    if tuple(pos) == (px, py) and (new_x, new_y) != (px, py):
                        messagebox.showwarning(
                            "No Free Space",
                            f"No free spot next to checkout for '{found}'.\n"
                            "It was left in place and not marked as impulse.",
                            parent=self.tk_root
                        )
                        continue
                    itm['position'] = pos
                else:
                    # skip marking if user declines to move
                    continue

            # 6) Mark as impulse. Generated and dataset-built fixtures carry
            #    a 'zone' stamp naming their department's Section_ wall, and
            #    the optimizer clips an item back inside that rectangle. An
            #    impulse item belongs at the checkout, so the department
            #    stamp goes with the old category.
            itm['category'] = 'Impulse'
            itm['color']    = '#FFD700'
            itm.pop('zone', None)

        self._invalidate_sim_geometry()
        self.redraw()

