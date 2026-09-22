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


class MetricsMixin:
    """Live metrics tab updater."""

    def update_metrics(self):
        """
        Method to produce the metrics.
    
    Uses a grid-based approach to precisely measure occupied and free areas.
    Calculates total area, usable area, occupied area by items and interior walls,
    and updates the metrics display with the results.
    
    The metrics include:
    - Total shop area (entire rectangle)
    - Usable area (excluding outer walls)
    - Items area (sum of all item areas)
    - Interior walls area
    - Occupied area (grid-based calculation)
    - Free area
    - Occupancy percentage
    - Item count
    - Interior wall count
    """
        if not getattr(self, '_metrics_dirty', True):
            return
        if not self.metrics_label:
            return

        # Nothing else refreshes the panel, so a dirty update that lands inside
        # the throttle window is retried once the window has passed instead of
        # being dropped.
        remaining = (getattr(self, '_metrics_interval', 1.0)
                     - (time.time() - getattr(self, '_last_metrics_time', 0.0)))
        if remaining > 0:
            if (getattr(self, '_metrics_after_id', None) is None
                    and getattr(self, 'tk_root', None) is not None):
                try:
                    self._metrics_after_id = self.tk_root.after(
                        int(remaining * 1000) + 1, self._metrics_retry
                    )
                except Exception:
                    self._metrics_after_id = None
            return

        W, H = self.width, self.height

        # Classify walls by geometry, not name: '+ Wall' names every wall
        # "Wall", "Wall2", ... wherever it is drawn, and the generator emits
        # checkout banks as Checkout, Checkout_Lane2, ... A boundary wall lies
        # entirely inside a thin band along one shop edge; any other physical
        # wall is an interior partition. Sections and connectors are overlays.
        edge_band = 0.5
        outer_walls, interior_walls, obstructions = [], [], []
        for name, wall in self.walls.items():
            if name.startswith("Section_") or wall.get('category') == 'Connector':
                continue
            if name.startswith("Checkout") or name == "WC":
                obstructions.append(wall)
                continue
            x, y = wall['position']
            w, h = wall['size']
            if (x + w <= edge_band or x >= W - edge_band
                    or y + h <= edge_band or y >= H - edge_band):
                outer_walls.append(wall)
            else:
                interior_walls.append(wall)

        # Calculate areas by type for reference
        items_area = sum(w*h for item in self.items.values() for w, h in [item['size']])

        # Only subtract true outer boundary walls from usable area
        outer_walls_area = sum(w * h for wall in outer_walls for w, h in [wall['size']])
        interior_walls_area = sum(w * h for wall in interior_walls for w, h in [wall['size']])

        # Usable area is total minus outer walls
        total_area = W * H
        usable_area = total_area - outer_walls_area

        # Grid-based occupancy
        grid_size = 0.1  # 10cm grid
        nx = int(round(W / grid_size))
        ny = int(round(H / grid_size))
        outer_cells = np.zeros((nx, ny), dtype=bool)
        occupied_cells = np.zeros((nx, ny), dtype=bool)

        def _mark(cells, pos, size):
            # Half-open cell ranges between the rounded edges, so a 1x1 m
            # rectangle covers exactly 10x10 cells rather than 11x11.
            x, y = pos
            w, h = size
            i0 = min(nx, max(0, int(round(x / grid_size))))
            i1 = min(nx, max(i0, int(round((x + w) / grid_size))))
            j0 = min(ny, max(0, int(round(y / grid_size))))
            j1 = min(ny, max(j0, int(round((y + h) / grid_size))))
            cells[i0:i1, j0:j1] = True

        for wall in outer_walls:
            _mark(outer_cells, wall['position'], wall['size'])

        # Interior walls, checkout lanes, WC and items are occupied space
        for wall in interior_walls + obstructions:
            _mark(occupied_cells, wall['position'], wall['size'])
        for item in self.items.values():
            _mark(occupied_cells, item['position'], item['size'])
        occupied_cells &= ~outer_cells

        total_positions = nx * ny
        usable_positions = total_positions - int(outer_cells.sum())
        occupied_positions = int(occupied_cells.sum())
        occupied_pct = (occupied_positions / usable_positions * 100) if usable_positions > 0 else 0

        occupied_area = (occupied_pct / 100) * usable_area
        free_area = usable_area - occupied_area

        txt = (
            f"Total shop area:    {total_area:.2f} m²\n"
            f"Usable area:    {usable_area:.2f} m²\n"
            f"Items area:    {items_area:.2f} m²\n"
            f"Interior walls:    {interior_walls_area:.2f} m²\n"
            f"Occupied area:    {occupied_area:.2f} m²\n"
            f"Free area:    {free_area:.2f} m²\n"
            f"Occupancy:    {occupied_pct:.1f}%\n\n"
            f"Items count:    {len(self.items)}\n"
            f"Interior walls:    {len(interior_walls)}"
        )
        self.metrics_label.config(text=txt)
        self._metrics_dirty = False
        self._last_metrics_time = time.time()

    def _metrics_retry(self):
        """Run a throttled metrics update if the metrics tab is still shown."""
        self._metrics_after_id = None
        if getattr(self, 'current_tab', None) == "Shop Area Metrics":
            self.update_metrics()
   

