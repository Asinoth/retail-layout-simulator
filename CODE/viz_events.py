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


class EventsMixin:
    """Mouse / keyboard / context-menu / rotate / color / resize event handlers."""

    def _on_patch_picked(self, event):
        if getattr(self, 'hand_mode', False):
            return
        patch = event.artist
        if any(dr.patch is patch for dr in self.resize_handles):
            return
        if self.selected_patch and self.selected_patch is not patch:
            self.selected_patch.set_edgecolor('black')
            self.selected_patch.set_linewidth(1)
            self._remove_resize_handles()
            self._remove_dimension_annotations()
        self.selected_patch = patch
        patch.set_edgecolor('#e94560')
        patch.set_linewidth(2)
        self._show_resize_handles(patch)
        self._update_status_bar()
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if getattr(event, 'inaxes', None) == self.ax and getattr(self, 'hand_mode', False):
            return
        if event.inaxes != self.ax:
            return

        if event.button == 3:
            self._show_context_menu(event)
            return

        if getattr(event, 'dblclick', False):
            for p in self.item_patches.values():
                contains, _ = p.contains(event)
                if contains:
                    self._open_edit_dialog(p)
                    return
            # Walls: prefer connectors, then non-sections, then sections
            wall_hits = [(nm, p) for nm, p in self.wall_patches.items()
                         if p.contains(event)[0]]
            if wall_hits:
                def _prio(nm):
                    w = self.walls.get(nm, {})
                    if w.get('connector_id'):
                        return 0
                    if not nm.startswith('Section_'):
                        return 1
                    return 2
                wall_hits.sort(key=lambda t: _prio(t[0]))
                nm, p = wall_hits[0]
                if self.walls.get(nm, {}).get('connector_id'):
                    self._rename_connector_wall(nm)
                else:
                    self._open_edit_dialog(p)
                return

        for p in list(self.item_patches.values()) + list(self.wall_patches.values()):
            contains, _ = p.contains(event)
            if contains:
                return
        if self.selected_patch:
            self.selected_patch.set_edgecolor('black')
            self.selected_patch.set_linewidth(1)
            self.selected_patch = None
            self._remove_resize_handles()
            self._remove_dimension_annotations()
            self._update_status_bar()
            self.canvas.draw_idle()

    def _on_key_press(self, event):  
        key = (getattr(event, 'key', '') or '').lower()
        if key == 'r' and self.selected_patch:
            self._rotate_item()  
            return  
        if key == 'delete' and self.selected_patch:
            for nm, p in list(self.item_patches.items()):
                if p is self.selected_patch:
                    self.remove_item(nm)
                    break
            for nm, p in list(self.wall_patches.items()):
                if p is self.selected_patch:
                    self.remove_wall(nm)
                    break
            self.selected_patch = None
            self._remove_resize_handles()
            self._remove_dimension_annotations()
            self._update_status_bar()
        elif key == 'escape' and self.selected_patch:
            self.selected_patch.set_edgecolor('black')
            self.selected_patch.set_linewidth(1)
            self.selected_patch = None
            self._remove_resize_handles()
            self._remove_dimension_annotations()
            self._update_status_bar()
            self.canvas.draw_idle()


    def _toggle_snap(self):
        self.snap_to_grid = bool(self._snap_var.get())

    def _remove_dimension_annotations(self):
        for artist in self._dimension_annotations:
            try:
                artist.remove()
            except Exception:
                pass
        self._dimension_annotations.clear()

    def _update_status_bar(self):


        if not self._selection_info_label:
            return
        if self.selected_patch:
            name = None
            for nm, p in self.item_patches.items():
                if p is self.selected_patch:
                    name = nm
                    break
            if not name:
                for nm, p in self.wall_patches.items():
                    if p is self.selected_patch:
                        name = nm
                        break
            if name:
                x, y = self.selected_patch.get_xy()
                w = self.selected_patch.get_width()
                h = self.selected_patch.get_height()
                self._selection_info_label.config(
                    text=f"Selected: {name}  |  Pos: ({x:.2f}, {y:.2f})  |  Size: {w:.2f} x {h:.2f} m"
                )
            else:
                self._selection_info_label.config(text="")
        else:
            self._selection_info_label.config(text="No selection")

    def _on_mouse_move(self, event):


        if event.inaxes == self.ax and self._cursor_pos_label and event.xdata is not None and event.ydata is not None:
            self._cursor_pos_label.config(
                text=f"X: {event.xdata:.2f}  Y: {event.ydata:.2f}"
            )

    def _on_scroll_zoom(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        old_z = self.zoom
        if event.button == 'up':
            new_z = old_z + 0.1
        elif event.button == 'down':
            new_z = max(1.0, old_z - 0.1)
        else:
            return
        new_z = round(new_z, 2)
        if new_z == old_z:
            return
        self._apply_zoom(new_z, event.xdata, event.ydata)




    def _show_context_menu(self, event):
        patch = None
        name = None
        obj_type = None
        # Items first (always on top)
        for nm, p in self.item_patches.items():
            contains, _ = p.contains(event)
            if contains:
                patch, name, obj_type = p, nm, 'item'
                break
        if not patch:
            # Walls: prefer connectors, then non-sections, then sections
            wall_hits = [(nm, p) for nm, p in self.wall_patches.items()
                         if p.contains(event)[0]]
            if wall_hits:
                def _prio(nm):
                    w = self.walls.get(nm, {})
                    if w.get('connector_id'):
                        return 0
                    if not nm.startswith('Section_'):
                        return 1
                    return 2
                wall_hits.sort(key=lambda t: _prio(t[0]))
                name, patch = wall_hits[0]
                obj_type = 'wall'
        if not patch:
            return

        if self.selected_patch and self.selected_patch is not patch:
            self.selected_patch.set_edgecolor('black')
            self.selected_patch.set_linewidth(1)
        self.selected_patch = patch

        patch.set_edgecolor('#e94560')
        patch.set_linewidth(2)
        self._remove_resize_handles()
        self._show_resize_handles(patch)
        self.canvas.draw_idle()

        is_connector = (obj_type == 'wall' and
                        self.walls.get(name, {}).get('connector_id') is not None)

        menu = tk.Menu(self.tk_root, tearoff=0,
                       bg='#16213e', fg='#e0e0e0',
                       activebackground='#e94560', activeforeground='white',
                       font=('Segoe UI', 9))

        if is_connector:
            menu.add_command(label=f"Rename '{name}'",
                             command=lambda n=name: self._rename_connector_wall(n))
        else:
            menu.add_command(label=f"Rename '{name}'",
                             command=lambda: self._open_edit_dialog(patch))
        menu.add_command(label="Change Colour",
                         command=lambda: self._on_color_btn_for(patch, name, obj_type))
        menu.add_command(label="Resize (numeric)",
                         command=lambda: self._open_resize_dialog(patch))
        menu.add_separator()
        if obj_type == 'item':
            menu.add_command(label="Duplicate",
                             command=lambda: self._duplicate_item(name))
        menu.add_command(label="Delete",
                         command=lambda: self._context_delete(name, obj_type))

        widget = self.canvas.get_tk_widget()
        try:
            pixel_x, pixel_y = self.ax.transData.transform((event.xdata, event.ydata))
            canvas_h = widget.winfo_height()
            screen_x = widget.winfo_rootx() + int(pixel_x)
            screen_y = widget.winfo_rooty() + int(canvas_h - pixel_y)
            menu.tk_popup(screen_x, screen_y)
        except Exception:
            menu.tk_popup(widget.winfo_pointerx(), widget.winfo_pointery())
        finally:
            menu.grab_release()

    def _rename_connector_wall(self, wall_name):
        """Rename a connector consistently on every floor it connects."""
        if wall_name not in self.walls:
            return
        data = self.walls.get(wall_name, {})
        connector_id = data.get('connector_id')
        if not connector_id:
            return

        new_name = simpledialog.askstring(
            "Rename Connector",
            f"New name for connector '{wall_name}' on all connected floors:",
            initialvalue=wall_name,
            parent=self.tk_root
        )
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name or new_name == wall_name:
            return

        mapping = getattr(self, 'connectors', {}).get(connector_id, {})
        floor_ids = list(mapping.keys()) or [self.current_floor]
        for fid in floor_ids:
            fdata = self.floors.get(fid, {})
            old = mapping.get(fid, wall_name)
            if new_name == old:
                continue
            if (new_name in fdata.get('walls', {}) and new_name != old) or \
               (new_name in fdata.get('items', {})):
                messagebox.showerror(
                    "Error",
                    f"Name '{new_name}' already exists on floor {fid}.",
                    parent=self.tk_root)
                return

        for fid in floor_ids:
            fdata = self.floors.get(fid, {})
            walls = fdata.get('walls', {})
            old = mapping.get(fid, wall_name)
            if old in walls:
                wdata = walls.pop(old)
                wdata['connector_kind'] = new_name
                walls[new_name] = wdata
                mapping[fid] = new_name

        if self.selected_patch is not None:
            self.selected_patch = None
        self._remove_resize_handles()
        if hasattr(self, 'customer_simulation'):
            self.customer_simulation.geometry_dirty = True
        self.redraw()

    def _on_color_btn_for(self, patch, name, obj_type):
        _, color = colorchooser.askcolor(
            title=f"Pick Colour for '{name}'",
            parent=self.tk_root
        )
        if color:
            target = self.items if obj_type == 'item' else self.walls
            target[name]['color'] = color
            self.redraw()

    def _duplicate_item(self, name):
        if name not in self.items:
            return
        orig = self.items[name]
        x, y = orig['position']
        w, h = orig['size']
        new_x = x + w + 0.2
        if new_x + w > self.width:
            new_x = max(0, x - w - 0.2)
        new_name = f"{name}_copy"
        idx = 2
        while new_name in self.items:
            new_name = f"{name}_copy{idx}"
            idx += 1
        self.items[new_name] = {
            "position": (new_x, y),
            "size":     (w, h),
            "category": orig.get('category', ''),
            "color":     orig.get('color'),
            "textcolor": orig.get('textcolor'),
            "font":      orig.get('font'),
            "fontsize":  orig.get('fontsize'),
            "price":     orig.get('price'),
        }
        self.redraw()

    def _context_delete(self, name, obj_type):
        if obj_type == 'item':
            self.remove_item(name)
        else:
            self.remove_wall(name)
        self.selected_patch = None
        self._remove_resize_handles()
        self._remove_dimension_annotations()
        self._update_status_bar()



    def _rotate_item(self, name=None, obj_type=None):
        """Rotate an item or wall 90deg by swapping width <-> height.
        If name/obj_type not given, operates on self.selected_patch."""
        if name is None:
            patch = self.selected_patch
            if not patch:
                return
            for nm, p in self.item_patches.items():
                if p is patch:
                    name, obj_type = nm, 'item'
                    break
            if name is None:
                for nm, p in self.wall_patches.items():
                    if p is patch:
                        name, obj_type = nm, 'wall'
                        break
            if name is None:
                return

        target = self.items if obj_type == 'item' else self.walls
        entry = target[name]
        w, h = entry['size']
        x, y = entry['position']

        # Check rotated dimensions still fit in shop bounds
        if x + h > self.width or y + w > self.height:
            messagebox.showwarning(
                "Rotate",
                f"Rotating '{name}' would exceed shop bounds.",
                parent=self.tk_root
            )
            return

        entry['size'] = (h, w)
        self.redraw()



 

    def _on_rename_btn(self):
        """Toolbar callbacks that open the respective edit or resize dialogs."""
        if self.selected_patch:
            self._open_edit_dialog(self.selected_patch)

    def _on_color_btn(self):
        patch = self.selected_patch
        if not patch:
            return
        name, obj_type = None, None
        for nm, p in self.item_patches.items():
            if p is patch:
                name, obj_type = nm, 'item'
                break
        if not name:
            for nm, p in self.wall_patches.items():
                if p is patch:
                    name, obj_type = nm, 'wall'
                    break
        if not name:
            return
        _, hexclr = colorchooser.askcolor(title=f"Pick Colour for '{name}'", parent=self.tk_root)
        if hexclr:
            if obj_type == 'item':
                self.items[name]['color'] = hexclr
            else:
                self.walls[name]['color'] = hexclr
            self.redraw()

    def _open_resize_dialog(self, patch):
        """
        Open a width/height dialog for an item or wall.
        """
        # figure out which object we're resizing
        obj_name = None
        obj_type = None
        for nm, p in self.item_patches.items():
            if p is patch:
                obj_name, obj_type = nm, 'item'
                break
        if obj_name is None:
            for nm, p in self.wall_patches.items():
                if p is patch:
                    obj_name, obj_type = nm, 'wall'
                    break
        if obj_name is None:
            return

        # fetch current size & position
        if obj_type == 'item':
            curr_w, curr_h = self.items[obj_name]['size']
            x, y = self.items[obj_name]['position']
        else:
            curr_w, curr_h = self.walls[obj_name]['size']
            x, y = self.walls[obj_name]['position']

        # build dialog
        dlg = tk.Toplevel(self.tk_root)
        dlg.transient(self.tk_root)
        dlg.title(f"Resize '{obj_name}'")

        tk.Label(dlg, text="Width (m):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        wvar = tk.DoubleVar(value=curr_w)
        tk.Entry(dlg, textvariable=wvar).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(dlg, text="Height (m):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        hvar = tk.DoubleVar(value=curr_h)
        tk.Entry(dlg, textvariable=hvar).grid(row=1, column=1, padx=5, pady=5)

        def on_ok():
            new_w = wvar.get()
            new_h = hvar.get()
            # must fit in shop
            if new_w <= 0 or new_h <= 0 or x + new_w > self.width or y + new_h > self.height:
                messagebox.showerror("Error",
                                     "Size must fit within shop bounds.",
                                     parent=dlg)
                return
            # collision check: items vs walls+items, walls vs items
            if obj_type == 'item':
                for wname, wdict in self.walls.items():
                    if wname.startswith('Section_'):
                        continue
                    wx, wy = wdict['position']
                    ww, wh = wdict['size']
                    if not (x + new_w <= wx or x >= wx + ww or
                            y + new_h <= wy or y >= wy + wh):
                        messagebox.showerror("Error",
                            f"Resize overlaps wall '{wname}'.", parent=dlg)
                        return
                for iname, idata in self.items.items():
                    if iname == obj_name:
                        continue
                    ix, iy = idata['position']
                    iw, ih = idata['size']
                    if not (x + new_w <= ix or x >= ix + iw or
                            y + new_h <= iy or y >= iy + ih):
                        messagebox.showerror("Error",
                            f"Resize overlaps item '{iname}'.", parent=dlg)
                        return
            elif obj_type == 'wall':
                for iname, idata in self.items.items():
                    ix, iy = idata['position']
                    iw, ih = idata['size']
                    if not (x + new_w <= ix or x >= ix + iw or
                            y + new_h <= iy or y >= iy + ih):
                        messagebox.showerror("Error",
                            f"Resize overlaps item '{iname}'.", parent=dlg)
                        return
            # commit new size
            if obj_type == 'item':
                self.items[obj_name]['size'] = (new_w, new_h)
            else:
                self.walls[obj_name]['size'] = (new_w, new_h)
            dlg.destroy()
            self.redraw()

        tk.Button(dlg, text="OK",     width=10, command=on_ok)   \
           .grid(row=2, column=0, padx=5, pady=5)
        tk.Button(dlg, text="Cancel", width=10, command=dlg.destroy) \
           .grid(row=2, column=1, padx=5, pady=5)

        # center dialog and grab focus
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x0 = (dlg.winfo_screenwidth()  // 2) - (w // 2)
        y0 = (dlg.winfo_screenheight() // 2) - (h // 2)
        dlg.geometry(f"{w}x{h}+{x0}+{y0}")
        dlg.grab_set()
        self.tk_root.wait_window(dlg)

    def _on_resize_btn(self):
        """
        Toolbar "Resize" button handler.
        If nothing is selected it warns; otherwise
        it opens the unified resize dialog.
        """
        # Nothing selected, check case
        if not self.selected_patch:
            messagebox.showwarning(
                "Resize",
                "Please select an item or wall first.",
                parent=self.tk_root
            )
            return

        # Open the shared resize dialog for items OR walls(new implement works)
        self._open_resize_dialog(self.selected_patch)

