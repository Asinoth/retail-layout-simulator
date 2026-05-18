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
        now = time.time()
        if now - getattr(self, '_last_metrics_time', 0.0) < getattr(self, '_metrics_interval', 1.0):
            return
        self._last_metrics_time = now

        if not getattr(self, '_metrics_dirty', True):
            return
        if not self.metrics_label:
            return

        # Calculate areas by type for reference
        items_area = sum(w*h for item in self.items.values() for w, h in [item['size']])

        # Only subtract true outer boundary walls from usable area
        outer_walls_area = sum(
            w * h
            for name, wall in self.walls.items()
            if name.startswith("Wall") and not name.startswith("IW")
            for w, h in [wall['size']]
        )

        interior_walls_area = sum(w*h for name, wall in self.walls.items() 
                                  if name.startswith("IW")
                                  for w, h in [wall['size']])

        # Usable area is total minus outer walls
        total_area = self.width * self.height
        usable_area = total_area - outer_walls_area

        # Grid-based occupancy
        grid_size = 0.1  # 10cm grid
        outer_wall_positions = set()
        occupied_positions = set()

        # Mark outer walls
        for name, wall in self.walls.items():
            if name.startswith("Wall") and not name.startswith("IW"):
                x, y = wall['position']
                w, h = wall['size']
                start_x = max(0, int(x / grid_size))
                start_y = max(0, int(y / grid_size))
                end_x = min(int((x + w) / grid_size) + 1, int(self.width / grid_size))
                end_y = min(int((y + h) / grid_size) + 1, int(self.height / grid_size))
                for gx in range(start_x, end_x):
                    for gy in range(start_y, end_y):
                        outer_wall_positions.add((gx, gy))

        # Mark interior walls (IW) and other obstructions like Checkout/WC as occupied
        for name, wall in self.walls.items():
            if name.startswith("IW") or name in ("Checkout", "WC"):
                x, y = wall['position']
                w, h = wall['size']
                start_x = max(0, int(x / grid_size))
                start_y = max(0, int(y / grid_size))
                end_x = min(int((x + w) / grid_size) + 1, int(self.width / grid_size))
                end_y = min(int((y + h) / grid_size) + 1, int(self.height / grid_size))
                for gx in range(start_x, end_x):
                    for gy in range(start_y, end_y):
                        if (gx, gy) not in outer_wall_positions:
                            occupied_positions.add((gx, gy))

        # Mark items
        for item in self.items.values():
            x, y = item['position']
            w, h = item['size']
            start_x = max(0, int(x / grid_size))
            start_y = max(0, int(y / grid_size))
            end_x = min(int((x + w) / grid_size) + 1, int(self.width / grid_size))
            end_y = min(int((y + h) / grid_size) + 1, int(self.height / grid_size))
            for gx in range(start_x, end_x):
                for gy in range(start_y, end_y):
                    if (gx, gy) not in outer_wall_positions:
                        occupied_positions.add((gx, gy))

        total_positions = int(self.width / grid_size) * int(self.height / grid_size)
        usable_positions = total_positions - len(outer_wall_positions)
        free_positions = usable_positions - len(occupied_positions)
        occupied_pct = (len(occupied_positions) / usable_positions * 100) if usable_positions > 0 else 0

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
            f"Interior walls:    {sum(1 for name in self.walls if name.startswith('IW'))}"
        )
        self.metrics_label.config(text=txt)
        self._metrics_dirty = False
        self._last_metrics_time = time.time()
   

