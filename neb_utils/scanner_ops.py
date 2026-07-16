"""
Scanner hardware operations – extracted from main.py.

``ScannerProcessor`` handles WIA device detection,
hardware scanning with COM threading, and scan UI cleanup.
"""

import os
import threading
from tkinter import messagebox

import pythoncom
from wia_scan import get_device_manager, connect_to_device_by_uid, scan_side

from neb_utils.utils import get_next_filename, to_roman, save_image_smart


class ScannerProcessor:
    """WIA scanner device discovery and image acquisition."""

    def __init__(self, app):
        self.app = app

    # ── Device detection ────────────────────────────────────────

    def load_hardware_devices(self):
        """Enumerate WIA scanners and populate the scanner dropdown."""
        try:
            manager = get_device_manager()
            device_names = []
            self.app.uids = []
            for i in range(1, manager.DeviceInfos.Count + 1):
                info = manager.DeviceInfos(i)
                device_names.append(info.Properties("Name").Value)
                self.app.uids.append(info.DeviceID)

            if device_names:
                self.app.scanner_menu.configure(values=device_names)
                self.app.scanner_menu.set(device_names[0])
                self.app.log_message(f"🔍 Scanner detected: {device_names[0]}")
            else:
                self.app.scanner_menu.configure(values=["No Scanner Detected"])
                self.app.log_message(
                    "ℹ️ No WIA scanner device found. Use 'IMPORT FROM DISK' to load an image."
                )
        except Exception as e:
            self.app.scanner_menu.configure(values=["No Scanner Detected"])
            self.app.scanner_menu.set("No Scanner Detected")
            self.app.log_message(
                "ℹ️ No scanner hardware available. Use 'IMPORT FROM DISK' to load an image."
            )

    # ── Scan workflow ───────────────────────────────────────────

    def run_hardware_scan(self):
        """Show the scanning overlay and start the hardware scan thread."""
        self.app.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.app.load_info.pack(expand=True, pady=(0, 10))
        self.app.prog.pack(pady=10)
        self.app.prog.start()
        self.app.update()
        self.app.log_message("Initializing WIA hardware...")
        scan_thread = threading.Thread(target=self.scan_thread_logic, daemon=True)
        scan_thread.start()

    def scan_thread_logic(self):
        """COM-initialised scan thread — capture, save, and notify the UI."""
        pythoncom.CoInitialize()
        try:
            selected_year = self.app.year_box.get()
            selected_exam = self.app.exam_btn.get()
            selected_grade = self.app.grade_btn.get()
            roman_grade = to_roman(selected_grade)
            year_folder = os.path.join(self.app.BASE_SAVE_PATH, selected_year)
            exam_folder_name = f"{selected_year} {roman_grade} {selected_exam}"
            final_folder = os.path.join(year_folder, exam_folder_name)
            os.makedirs(final_folder, exist_ok=True)
            save_path = get_next_filename(final_folder)

            selected_name = self.app.scanner_menu.get()
            target_uid = self.app.uids[
                self.app.scanner_menu._values.index(selected_name)
            ]
            connected_device = connect_to_device_by_uid(device_uid=target_uid)
            img = scan_side(device=connected_device)
            img.save(save_path, "JPEG", quality=95)
            self.app.selected_file = save_path
            self.app.after(0, self.finish_scan_ui)
        except Exception as e:
            error_msg = str(e)
            self.app.after(
                0, lambda m=error_msg: messagebox.showerror("Hardware Error", m)
            )
            self.app.after(0, lambda: self.app.overlay.place_forget())
        finally:
            pythoncom.CoUninitialize()

    def finish_scan_ui(self):
        """Remove the scanning overlay and display the captured image."""
        self.app.overlay.place_forget()
        self.app.prog.stop()
        if self.app.selected_file:
            self.app.display_image()
            self.app.log_message("Buffer digitized: [SUCCESS]")
