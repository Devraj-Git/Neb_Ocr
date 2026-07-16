"""
Image preview interaction – extracted from main.py.

``ImageEditor`` owns drag-to-pan, scroll-wheel zoom, physical rotation,
crop selection, undo history, export, print, and canvas-clearing logic.
"""

import os
from tkinter import filedialog, messagebox

from PIL import Image
import customtkinter as ctk


class ImageEditor:
    """Handles all user interactions with the image preview."""

    def __init__(self, app):
        self.app = app

    # ── Dock toolbar helper ─────────────────────────────────────

    def create_dock_btn(self, parent, icon, command, color):
        """Create a floating toolbar button."""
        btn = ctk.CTkButton(
            parent,
            text=icon,
            width=60,
            height=40,
            corner_radius=12,
            fg_color="transparent",
            text_color=color,
            hover_color="#334155",
            font=("Inter", 22),
            command=command,
        )
        btn.pack(side="left", padx=10, pady=5)

    # ── Drag (pan) ──────────────────────────────────────────────

    def start_drag(self, event):
        """Record the global mouse position and current label position."""
        if self.app.selected_file:
            self.app.drag_data["mouse_x"] = event.x_root
            self.app.drag_data["mouse_y"] = event.y_root
            self.app.drag_data["label_x"] = self.app.display_label.winfo_x()
            self.app.drag_data["label_y"] = self.app.display_label.winfo_y()
            self.app.display_label.configure(cursor="fleur")

    def do_drag(self, event):
        """Move the image label based on mouse delta."""
        if self.app.selected_file:
            dx = event.x_root - self.app.drag_data["mouse_x"]
            dy = event.y_root - self.app.drag_data["mouse_y"]
            new_x = self.app.drag_data["label_x"] + dx
            new_y = self.app.drag_data["label_y"] + dy
            self.app.display_label.place(x=new_x, y=new_y, anchor="nw")

    # ── Zoom ────────────────────────────────────────────────────

    def zoom_image(self, event):
        """Resize the image preview on scroll-wheel delta."""
        if not self.app.selected_file:
            return
        if event.delta > 0:
            self.app.zoom_level *= 1.1
        else:
            self.app.zoom_level /= 1.1
        self.app.zoom_level = max(0.2, min(self.app.zoom_level, 4.0))
        self.app.display_image()

    # ── Rotation ────────────────────────────────────────────────

    def rotate_image(self):
        """Physically rotate the source image by -90 degrees and refresh."""
        if self.app.selected_file:
            try:
                with Image.open(self.app.selected_file) as img:
                    rotated = img.rotate(-90, expand=True)
                    rotated.save(self.app.selected_file)
                self.app.display_image()
                self.app.log_message("Image physically rotated and saved.")
            except Exception as e:
                self.app.log_message(f"Rotation Error: {str(e)}")

    # ── Crop ────────────────────────────────────────────────────

    def toggle_crop_mode(self):
        """Enter / exit interactive crop mode, swapping event bindings."""
        if not self.app.selected_file:
            return
        self.app.is_crop_mode = not self.app.is_crop_mode

        if self.app.is_crop_mode:
            self.app.display_label.configure(cursor="cross")
            self.app.log_message("CROP MODE: Draw a box over the image to cut.")
            self.app.display_label.unbind("<ButtonPress-1>")
            self.app.display_label.unbind("<B1-Motion>")
            self.app.display_label.bind("<ButtonPress-1>", self.start_crop_select)
            self.app.display_label.bind("<B1-Motion>", self.draw_crop_rect)
            self.app.display_label.bind("<ButtonRelease-1>", self.execute_crop)
        else:
            self.app.display_label.configure(cursor="hand2")
            self.app.display_label.bind("<ButtonPress-1>", self.start_drag)
            self.app.display_label.bind("<B1-Motion>", self.do_drag)
            self.app.log_message("CROP MODE: Deactivated.")

    def start_crop_select(self, event):
        """Record the starting point of the crop rectangle."""
        self.app.crop_start_x = event.x
        self.app.crop_start_y = event.y
        self.app.crop_rect = ctk.CTkFrame(
            self.app.display_label,
            fg_color="transparent",
            bg_color="transparent",
            border_width=2,
            border_color=self.app.COLORS["accent"],
            width=0,
            height=0,
        )

    def draw_crop_rect(self, event):
        """Update the crop rectangle dimensions as the user drags."""
        if not self.app.crop_rect or not self.app.crop_rect.winfo_exists():
            return
        cur_x, cur_y = event.x, event.y
        x = min(self.app.crop_start_x, cur_x)
        y = min(self.app.crop_start_y, cur_y)
        w = abs(cur_x - self.app.crop_start_x)
        h = abs(cur_y - self.app.crop_start_y)
        self.app.crop_rect.configure(width=w, height=h)
        self.app.crop_rect.place(x=x, y=y)

    def execute_crop(self, event):
        """Perform the actual crop, save to disk, and refresh the preview."""
        if not self.app.crop_rect or not self.app.crop_rect.winfo_exists():
            return

        try:
            with Image.open(self.app.selected_file) as current_img:
                self.app.image_history.append(current_img.copy())
                if len(self.app.image_history) > 5:
                    self.app.image_history.pop(0)

            ui_w = self.app.display_label.winfo_width()
            ui_h = self.app.display_label.winfo_height()
            rx = self.app.crop_rect.winfo_x()
            ry = self.app.crop_rect.winfo_y()
            rw = self.app.crop_rect.winfo_width()
            rh = self.app.crop_rect.winfo_height()

            if rw < 10 or rh < 10:
                return

            with Image.open(self.app.selected_file) as img:
                img = img.rotate(-self.app.rotation, expand=True)
                sx, sy = img.size[0] / ui_w, img.size[1] / ui_h
                cropped = img.crop(
                    (rx * sx, ry * sy, (rx + rw) * sx, (ry + rh) * sy)
                )
                cropped.save(self.app.selected_file, quality=95)
                self.app.rotation = 0

            self.app.log_message("Auto-Save: Image cropped and synchronized to disk.")

        except Exception as e:
            self.app.log_message(f"Crop Error: {str(e)}")
        finally:
            if self.app.crop_rect and self.app.crop_rect.winfo_exists():
                self.app.crop_rect.destroy()
            self.toggle_crop_mode()
            self.app.display_image()

    # ── Undo ────────────────────────────────────────────────────

    def undo_last_action(self):
        """Restore the previous image state from the history stack."""
        if not self.app.image_history:
            self.app.log_message("Undo Warning: No further history available.")
            return

        try:
            last_img = self.app.image_history.pop()
            last_img.save(self.app.selected_file, quality=95)
            self.app.rotation = 0
            self.app.display_image()
            self.app.log_message("Undo Success: Previous state restored and saved.")
        except Exception as e:
            self.app.log_message(f"Undo Error: {str(e)}")

    # ── Export / Print / Clear ──────────────────────────────────

    def save_image(self):
        """Export the current preview as a new image file."""
        if self.app.selected_file:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                with Image.open(self.app.selected_file) as img:
                    img.rotate(-self.app.rotation, expand=True).save(path)
                self.app.log_message(f"Export Success: {os.path.basename(path)}")

    def print_image(self):
        """Send the current image to the system print dialog."""
        if self.app.selected_file:
            try:
                os.startfile(self.app.selected_file, "print")
            except Exception:
                messagebox.showerror("Hardware", "No printing device detected.")

    def clear_canvas(self):
        """Clear the preview and reset the image buffer."""
        self.app.selected_file = None
        self.app.current_preview_img = None
        self.app.display_label.configure(
            image=None, text="CORE IDLE\nWaiting for document input..."
        )
        self.app.update()
        self.app.log_message("Image buffer flushed.")
