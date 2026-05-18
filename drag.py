# drag n drop for rectangles

class DraggableRectangle:
    """Click-drag for matplotlib rects. Uses blitting for smooth drags."""

    def __init__(self,patch,on_update=None, owner=None, corner=None):
        self.patch=patch
        self.press = None
        self.on_update = on_update
        self.owner = owner
        self.corner = corner
        self.main_patch = getattr(patch, 'main_patch', None)
        self.origin = None
        self._orig_fc = None
        self._orig_ec = None
        self._orig_alpha=None
        self._background = None
        self._drag_text = None
        self.connect()

    def connect(self):
        fig = self.patch.figure
        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release= fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event',self.on_motion)

    def on_press(self, event):
        if self.owner and getattr(self.owner, 'hand_mode', False) == True:
            return
        if event.inaxes != self.patch.axes:
            return
        if not event.button == 1:
            return

        if self.owner:
            if self.corner:
                if self.owner.selected_patch not in (self.main_patch, self.patch):
                    return
            else:
                if self.owner.selected_patch is not self.patch:
                    return

        contains, _ = self.patch.contains(event)
        if contains == False:
            return

        x0, y0 = self.patch.get_xy()
        self.press = (x0, y0, event.xdata, event.ydata)
        self.origin = (x0,y0)
        self._orig_fc = self.patch.get_facecolor()
        self._orig_ec = self.patch.get_edgecolor()
        self._orig_alpha = self.patch.get_alpha()

        # setup blitting for main patches
        if not self.corner:
            self._setup_blit()

    def _setup_blit(self):
        """Capture background for blitting"""
        try:
            owner = self.owner
            ax = self.patch.axes
            fig = self.patch.figure
            canvas = fig.canvas

            if owner:
                owner._remove_resize_handles()
                owner._remove_dimension_annotations()

            self._drag_text = None
            if owner:
                self._drag_text = owner.annotation_patches.get(self.patch)

            self.patch.set_visible(False)
            if self._drag_text:
                self._drag_text.set_visible(False)

            canvas.draw()
            self._background = canvas.copy_from_bbox(ax.bbox)

            self.patch.set_visible(True)
            if self._drag_text:
                self._drag_text.set_visible(True)
        except Exception:
            self._background = None
            self._drag_text = None

    def _restore_appearance(self):
        """Restore patch to original look"""
        if self._orig_fc is not None:
            self.patch.set_facecolor(self._orig_fc)
        if self._orig_ec is not None:
            self.patch.set_edgecolor(self._orig_ec)
        if self._orig_alpha is not None:
            self.patch.set_alpha(self._orig_alpha)
        else:
            self.patch.set_alpha(1.0)

    def _apply_invalid_tint(self):
        """Red tint for invalid pos"""
        self.patch.set_facecolor((1.0, 0.2, 0.2, 0.5))
        self.patch.set_edgecolor((1.0, 0.0, 0.0, 0.9))
        self.patch.set_alpha(0.6)

    def on_motion(self, event):
        if self.owner and getattr(self.owner, 'hand_mode', False):
            return
        if self.press is None or event.inaxes != self.patch.axes:
            return

        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        if self.corner:
            self.owner._on_handle_drag(self, event.xdata, event.ydata)
            return

        xnew = x0 + dx
        ynew = y0 + dy
        owner = self.owner

        if owner and getattr(owner, 'snap_to_grid', False):
            snap = owner._snap_size
            xnew = round(xnew / snap) * snap
            ynew = round(ynew / snap) * snap

        self.patch.set_xy((xnew, ynew))

        if self.on_update:
            self.on_update(self.patch)

        # red if invalid
        if self._check_collision() == True:
            self._apply_invalid_tint()
        else:
            self._restore_appearance()

        # render
        if self._background is not None:
            canvas = self.patch.figure.canvas
            ax = self.patch.axes
            canvas.restore_region(self._background)
            ax.draw_artist(self.patch)
            if self._drag_text:
                ax.draw_artist(self._drag_text)
            canvas.blit(ax.bbox)
        else:
            self.patch.figure.canvas.draw_idle()

        if owner:
            owner._update_status_bar()

    def _check_collision(self):
        """Check if patch pos is invalid"""
        owner = self.owner
        if owner is None:
            return False

        px, py = self.patch.get_xy()
        pw = self.patch.get_width()
        ph = self.patch.get_height()

        # bounds check
        if px < 0 or py < 0 or px + pw > owner.width or py + ph > owner.height:
            return True

        # section patches
        if getattr(self.patch, 'is_section', False) == True:
            for nm, wpatch in owner.wall_patches.items():
                if not getattr(wpatch, 'is_section', False) or wpatch is self.patch:
                    continue
                x2, y2 = wpatch.get_xy()
                w2 = wpatch.get_width()
                h2 = wpatch.get_height()
                if not (px + pw <= x2 or px >= x2 + w2) and not (py + ph <= y2 or py >= y2 + h2):
                    return True
            return False

        # item patches
        if self.patch in owner.item_patches.values():
            for nm, wdict in owner.walls.items():
                if nm.startswith('Section_'):
                    continue
                wx, wy = wdict['position']
                ww, wh = wdict['size']
                if not (px + pw <= wx or px >= wx + ww) and not (py + ph <= wy or py >= wy + wh):
                    return True
            for inm, ipatch in owner.item_patches.items():
                if ipatch is self.patch:
                    continue
                x2, y2 = ipatch.get_xy()
                w2 = ipatch.get_width()
                h2 = ipatch.get_height()
                if not (px + pw <= x2 or px >= x2 + w2) and not (py + ph <= y2 or py >= y2 + h2):
                    return True
            return False

        # wall patches
        if self.patch in owner.wall_patches.values():
            for inm, idata in owner.items.items():
                ix, iy = idata['position']
                iw, ih = idata['size']
                if not (px + pw <= ix or px >= ix + iw) and not (py + ph <= iy or py >= iy + ih):
                    return True
            return False

        return False

    def on_release(self, event):
        if self.press is not None and self.owner:
            self._restore_appearance()

            # revert if invalid
            if self.origin and self._check_collision():
                self.patch.set_xy(self.origin)
                if self.on_update:
                    self.on_update(self.patch)

            self.patch.figure.canvas.draw()

            self.owner._remove_resize_handles()
            if self.owner.selected_patch:
                self.owner._show_resize_handles(self.owner.selected_patch)

        self.press = None
        self.origin = None
        self._orig_fc = None
        self._orig_ec = None
        self._orig_alpha = None
        self._background = None
        self._drag_text = None

    def disconnect(self):
        fig = self.patch.figure
        fig.canvas.mpl_disconnect(self.cid_press)
        fig.canvas.mpl_disconnect(self.cid_release)
        fig.canvas.mpl_disconnect(self.cid_motion)
