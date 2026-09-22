import logging

class _DropFindfont(logging.Filter):
    def filter(self, record):
        try:
            return 'findfont' not in record.getMessage()
        except Exception:
            return True

# Attach to BOTH the named logger and the root, so basicConfig later can't bypass us
_fm_logger = logging.getLogger('matplotlib.font_manager')
_fm_logger.addFilter(_DropFindfont())
_fm_logger.setLevel(logging.ERROR)
_fm_logger.propagate = False
logging.getLogger().addFilter(_DropFindfont())

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=r".*findfont.*")

import sys
import tkinter as tk
from tkinter import simpledialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams['font.family']  = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

from visualizer import ShopVisualizer

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # The dimensions size the heat-map arrays and every geometry cache, so a
    # zero or negative entry either opens a degenerate shop or fails while the
    # simulation is being built. askfloat keeps re-asking below its minimum.
    area  = simpledialog.askfloat("Shop Size", "Total area (m²):", parent=root,
                                  minvalue=1.0)
    width = simpledialog.askfloat("Shop Size", "Width (m):",        parent=root,
                                  minvalue=1.0)
    depth = simpledialog.askfloat("Shop Size", "Depth (m):",        parent=root,
                                  minvalue=1.0)

    if None in (area, width, depth):
        print("Cancelled.")
        sys.exit()

    if abs(width * depth - area) > 0.1:
        messagebox.showwarning("Warning",
            "Width×Depth = {:.2f} ≠ Area = {:.2f}. Using width & depth.".format(width * depth, area),
            parent=root)

    root.destroy()

    shop = ShopVisualizer((width, depth))
    shop.show()