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


class EditMixin:
    """Edit dialog and resize-handle interaction."""

    def _open_edit_dialog(self, patch):
        """Pop up a Toplevel allowing rename, font, colours, and resize commands for a patch."""
        # identify object name & type, item or wall
        obj_name, obj_type = None, None
        for nm, p in self.item_patches.items():
            if p is patch:
                obj_name, obj_type = nm, 'item'
                break
        if not obj_name:
            for nm, p in self.wall_patches.items():
                if p is patch:
                    obj_name, obj_type = nm, 'wall'
                    break
        if not obj_name:
            return

        # prepare current values
        target = self.items if obj_type == 'item' else self.walls
        current_color      = target[obj_name].get('color')
        current_text_color = target[obj_name].get('textcolor',
                    'white' if obj_type=='item' else 'black')
        current_font       = target[obj_name].get('font', tkfont.families()[0])
        current_size       = target[obj_name].get('fontsize',
                    10 if obj_type=='item' else 14)
        new_color      = current_color
        new_text_color = current_text_color

        # build dialog
        dlg = tk.Toplevel(self.tk_root)
        dlg.transient(self.tk_root)
        dlg.title(f"Edit '{obj_name}'")

        # 1) Name
        tk.Label(dlg, text="Name:")\
            .grid(row=0, column=0, sticky='e', padx=5, pady=5)
        name_var = tk.StringVar(value=obj_name)
        tk.Entry(dlg, textvariable=name_var)\
            .grid(row=0, column=1, padx=5, pady=5)

        # 2) Font family
        tk.Label(dlg, text="Font:")\
            .grid(row=1, column=0, sticky='e', padx=5, pady=5)
        font_var = tk.StringVar(value=current_font)
        ttk.Combobox(
            dlg,
            textvariable=font_var,
            values=tkfont.families(),
            state='readonly'
        ).grid(row=1, column=1, padx=5, pady=5)

        # 3) Font size
        tk.Label(dlg, text="Font Size:")\
            .grid(row=2, column=0, sticky='e', padx=5, pady=5)
        size_var = tk.IntVar(value=current_size)
        tk.Spinbox(dlg, from_=8, to=72, textvariable=size_var)\
            .grid(row=2, column=1, padx=5, pady=5)

        # 4) Text colour
        def choose_text_color():
            nonlocal new_text_color
            _, c = colorchooser.askcolor(
                title=f"Pick Text Colour for '{obj_name}'",
                parent=dlg
            )
            if c:
                new_text_color = c
        tk.Button(dlg, text="Text Colour", command=choose_text_color)\
            .grid(row=3, column=0, columnspan=2, pady=5)

        # 5) Background colour
        def choose_color():
            nonlocal new_color
            _, c = colorchooser.askcolor(
                title=f"Pick Background Colour for '{obj_name}'",
                parent=dlg
            )
            if c:
                new_color = c
        tk.Button(dlg, text="Change Background Colour", command=choose_color)\
            .grid(row=4, column=0, columnspan=2, pady=5)

        # 6) Resize
        tk.Button(
            dlg,
            text="Resize",
            width=10,
            command=lambda: self._open_resize_dialog(patch)
        ).grid(row=5, column=0, columnspan=2, pady=5)

        # 7) Rotate 90deg
        tk.Button(
            dlg,
            text="Rotate 90°",
            width=10,
            command=lambda: [self._rotate_item(), dlg.destroy()]
        ).grid(row=6, column=0, columnspan=2, pady=5)

        # 8) Duplicate & Delete
        if obj_type == 'item':
            tk.Button(
                dlg, text="Duplicate", width=10,
                command=lambda: [self._duplicate_item(obj_name), dlg.destroy()]
            ).grid(row=7, column=0, padx=5, pady=5)
            tk.Button(
                dlg, text="Delete", width=10, fg='red',
                command=lambda: [self._context_delete(obj_name, obj_type), dlg.destroy()]
            ).grid(row=7, column=1, padx=5, pady=5)
        else:
            tk.Button(
                dlg, text="Delete", width=10, fg='red',
                command=lambda: [self._context_delete(obj_name, obj_type), dlg.destroy()]
            ).grid(row=7, column=0, columnspan=2, pady=5)

        # 9) OK / Cancel
        def on_ok():
            new_nm = name_var.get().strip()
            if not new_nm:
                messagebox.showerror("Error",
                    "Name cannot be empty",
                    parent=dlg)
                return
            if new_nm != obj_name and new_nm in target:
                messagebox.showerror("Error",
                    f"Name '{new_nm}' already exists",
                    parent=dlg)
                return

            # apply changes
            if new_nm != obj_name:
                target[new_nm] = target.pop(obj_name)
            target[new_nm]['color']     = new_color
            target[new_nm]['textcolor'] = new_text_color
            target[new_nm]['font']      = font_var.get()
            target[new_nm]['fontsize']  = size_var.get()

            dlg.destroy()
            self.redraw()

        tk.Button(dlg, text="OK",     width=10, command=on_ok)\
            .grid(row=8, column=0, padx=5, pady=5)
        tk.Button(dlg, text="Cancel", width=10, command=dlg.destroy)\
            .grid(row=8, column=1, padx=5, pady=5)

        # center & grab focus
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x0 = (dlg.winfo_screenwidth()  // 2) - (w // 2)
        y0 = (dlg.winfo_screenheight() // 2) - (h // 2)
        dlg.geometry(f"{w}x{h}+{x0}+{y0}")
        dlg.grab_set()
        self.tk_root.wait_window(dlg)
        

    def _on_handle_drag(self, dr, mx, my):
        """Resize with collision detection and opposite-side fallback.
        
        If expanding in the natural direction collides, tries expanding
        from the opposite side. For corners: natural -> flip X -> flip Y -> flip both.
        Also properly handles edge handles (tm, bm, ml, mr).
        """
        patch = dr.main_patch
        if patch is None:
            return
        x0, y0 = patch.get_xy()
        w0, h0 = patch.get_width(), patch.get_height()
        corner = dr.corner
        right = x0 + w0
        top = y0 + h0

        # -- X axis: compute desired width + natural/opposite origins --
        if corner in ('lr', 'mr', 'ur'):
            # Right edge follows mouse
            des_w = max(mx - x0, 0.1)
            nat_x = x0
            opp_x = right - des_w
        elif corner in ('ll', 'ml', 'ul'):
            # Left edge follows mouse
            des_w = max(right - mx, 0.1)
            nat_x = mx
            opp_x = x0
        else:  # 'tm', 'bm' -- width unchanged
            des_w = w0
            nat_x = x0
            opp_x = x0

        # -- Y axis: compute desired height + natural/opposite origins --
        if corner in ('lr', 'll', 'bm'):
            # Lower edge follows mouse
            des_h = max(my - y0, 0.1)
            nat_y = y0
            opp_y = top - des_h
        elif corner in ('ur', 'ul', 'tm'):
            # Upper edge follows mouse
            des_h = max(top - my, 0.1)
            nat_y = my
            opp_y = y0
        else:  # 'ml', 'mr' -- height unchanged
            des_h = h0
            nat_y = y0
            opp_y = y0

        # -- Clamp to shop bounds --
        def clamp(nx, ny, nw, nh):
            nw = max(nw, 0.1)
            nh = max(nh, 0.1)
            nx = max(nx, 0)
            ny = max(ny, 0)
            if nx + nw > self.width:
                nw = self.width - nx
            if ny + nh > self.height:
                nh = self.height - ny
            return nx, ny, nw, nh

        # -- Try candidates: natural -> flip X -> flip Y -> flip both --
        candidates = [
            clamp(nat_x, nat_y, des_w, des_h),
            clamp(opp_x, nat_y, des_w, des_h),
            clamp(nat_x, opp_y, des_w, des_h),
            clamp(opp_x, opp_y, des_w, des_h),
        ]

        for nx, ny, nw, nh in candidates:
            if not self._resize_collides(patch, nx, ny, nw, nh):
                patch.set_xy((nx, ny))
                patch.set_width(nw)
                patch.set_height(nh)
                self._on_patch_moved(patch)
                self._remove_resize_handles()
                self._show_resize_handles(patch)
                self.canvas.draw_idle()
                return
        # All candidates collide -- don't resize

    def _resize_collides(self, patch, nx, ny, nw, nh):
        """Check if resizing patch to (nx, ny, nw, nh) would collide with anything."""
        if nx < 0 or ny < 0 or nx + nw > self.width or ny + nh > self.height:
            return True

        # Section vs other sections
        if getattr(patch, 'is_section', False):
            for nm, wpatch in self.wall_patches.items():
                if not getattr(wpatch, 'is_section', False) or wpatch is patch:
                    continue
                x2, y2 = wpatch.get_xy()
                w2, h2 = wpatch.get_width(), wpatch.get_height()
                if not (nx + nw <= x2 or nx >= x2 + w2) and not (ny + nh <= y2 or ny >= y2 + h2):
                    return True
            return False

        # Item vs walls + other items
        if patch in self.item_patches.values():
            for nm, wdict in self.walls.items():
                if nm.startswith('Section_'):
                    continue
                wx, wy = wdict['position']
                ww, wh = wdict['size']
                if not (nx + nw <= wx or nx >= wx + ww) and not (ny + nh <= wy or ny >= wy + wh):
                    return True
            for inm, ipatch in self.item_patches.items():
                if ipatch is patch:
                    continue
                x2, y2 = ipatch.get_xy()
                w2, h2 = ipatch.get_width(), ipatch.get_height()
                if not (nx + nw <= x2 or nx >= x2 + w2) and not (ny + nh <= y2 or ny >= y2 + h2):
                    return True
            return False

        # Wall vs items
        if patch in self.wall_patches.values():
            for inm, idata in self.items.items():
                ix, iy = idata['position']
                iw, ih = idata['size']
                if not (nx + nw <= ix or nx >= ix + iw) and not (ny + nh <= iy or ny >= iy + ih):
                    return True
            return False

        return False

    def _remove_resize_handles(self):
        for dr in self.resize_handles:
            try:
                dr.disconnect()
            except Exception:
                pass
            try:
                if dr.patch.axes is not None:
                    dr.patch.remove()
            except Exception:
                pass
        self.resize_handles.clear()

    def _show_resize_handles(self, patch):
        self._remove_resize_handles()
        self._remove_dimension_annotations()
        x, y = patch.get_xy()
        w, h = patch.get_width(), patch.get_height()

        handle_size = max(min(w, h) * 0.08, 0.12)
        hs = handle_size

        handle_positions = {
            'ul': (x - hs/2, y + h - hs/2),
            'ur': (x + w - hs/2, y + h - hs/2),
            'll': (x - hs/2, y - hs/2),
            'lr': (x + w - hs/2, y - hs/2),
            'tm': (x + w/2 - hs/2, y + h - hs/2),
            'bm': (x + w/2 - hs/2, y - hs/2),
            'ml': (x - hs/2, y + h/2 - hs/2),
            'mr': (x + w - hs/2, y + h/2 - hs/2),
        }

        for corner, (cx, cy) in handle_positions.items():
            is_edge = corner in ('tm', 'bm', 'ml', 'mr')
            handle_patch = plt.Rectangle(
                (cx, cy), hs, hs,


                facecolor='white', edgecolor='#e94560',
                linewidth=1.0, zorder=patch.get_zorder() + 3
            )
            handle_patch.main_patch = patch
            self.ax.add_patch(handle_patch)
            dr = DraggableRectangle(handle_patch, owner=self, corner=corner)
            self.resize_handles.append(dr)

        sel_rect = plt.Rectangle(
            (x, y), w, h, fill=False, edgecolor='#e94560',
            linewidth=1.0, linestyle='--', zorder=patch.get_zorder() + 2
        )
        self.ax.add_patch(sel_rect)
        self._dimension_annotations.append(sel_rect)

        dim_w_txt = self.ax.text(
            x + w/2, y - 0.3, f"{w:.2f}m",
            ha='center', va='top', color='#e94560',
            fontsize=7, fontfamily='Consolas',
            zorder=patch.get_zorder() + 3,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a1a2e',
                      edgecolor='#e94560', alpha=0.9, linewidth=0.5)
        )
        self._dimension_annotations.append(dim_w_txt)

        dim_h_txt = self.ax.text(
            x + w + 0.3, y + h/2, f"{h:.2f}m",
            ha='left', va='center', color='#e94560',
            fontsize=7, fontfamily='Consolas', rotation=90,
            zorder=patch.get_zorder() + 3,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a1a2e',
                      edgecolor='#e94560', alpha=0.9, linewidth=0.5)
        )
        self._dimension_annotations.append(dim_h_txt)

        self.canvas.draw_idle()
        self._update_status_bar()

