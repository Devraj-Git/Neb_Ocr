import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import datetime
import threading
import time
from wia_scan import get_device_manager, connect_to_device_by_uid, scan_side
from neb_utils.utils import (
    get_next_filename, save_image_smart, to_roman,
    build_image_filename,
    load_checkpoint, save_checkpoint, delete_checkpoint,
)
from neb_utils.vlm_front import get_vlm_result, save_vlm_to_database
from neb_utils.ollama_pipeline import _get_db, get_exam_id, is_student_already_in_db
from neb_utils.webhook_client import WebhookClient
import pythoncom
from dotenv import load_dotenv
load_dotenv()

BASE_SAVE_PATH = os.getenv("BASE_SAVE_PATH")
# Finalized 2026 Enterprise Palette
COLORS = {
    "bg": "#020617",           
    "sidebar": "#0F172A",      
    "accent": "#38BDF8",       
    "accent_hover": "#0EA5E9",
    "card": "#1E293B",         
    "text_main": "#F8FAFC",
    "text_dim": "#94A3B8",     
    "success": "#22C55E",      
    "border": "#334155",
    "dock": "#111827",
    "danger": "#EF4444"
}

ctk.set_appearance_mode("Dark")

class RadiantUltraScanner(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NEB | CORE OPTICAL CHARACTER READER")
        self.geometry("1300x900")
        self.configure(fg_color=COLORS["bg"])

        self.selected_file = None
        self.uids = []
        self.rotation = 0 
        self.crop_start_x = 0
        self.crop_start_y = 0
        self.crop_rect = None
        self.is_crop_mode = False
        self.image_history = []
        self.current_ocr_data = None
        self.current_vlm_raw = None
        self.is_batch_mode = False
        self.batch_files = []
        self.batch_index = 0
        self.batch_total = 0
        self.batch_success = 0
        self.batch_fail = 0
        self.auto_commit = ctk.BooleanVar(value=False)
        self.batch_abort = False
        self.batch_folder = None
        self.batch_file_map = {}
        self.batch_completed_originals = set()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Webhook client reference (initialized after sidebar) ──
        self.webhook_client: WebhookClient = None
        self._webhook_status = "DISABLED"

        self.setup_sidebar()
        self.setup_main_view()
        self.setup_loading_overlay()
        self.update_live_elements()
        
        # FINAL UPGRADE: Professional Initial Log
        self.load_hardware_devices()

        # ── Start webhook polling client ──
        self._init_webhook_client()

    # ── Webhook Client ──────────────────────────────────────────

    def _init_webhook_client(self):
        """Initialize and start the webhook polling client."""
        self.webhook_client = WebhookClient(
            status_callback=self._on_webhook_status,
            log_callback=self._on_webhook_log,
        )
        self.webhook_client.start()

    def _on_webhook_status(self, status: str):
        """Update the webhook status indicator in the sidebar (thread-safe)."""
        self._webhook_status = status
        self.after(0, self._refresh_webhook_ui)

    def _on_webhook_log(self, msg: str):
        """Forward webhook log messages to the console (thread-safe)."""
        self.after(0, lambda m=msg: self.log_message(m))

    def _refresh_webhook_ui(self):
        """Refresh the webhook status UI elements based on current state."""
        s = self._webhook_status

        # ── Header indicator ──
        if s == "CONNECTED":
            text = "🌐 REMOTE: ● ONLINE"
            color = COLORS["success"]
            bg = "#14532D"
        elif s == "DISCONNECTED":
            text = "🌐 REMOTE: ○ OFFLINE"
            color = COLORS["danger"]
            bg = "#7F1D1D"
        elif s == "DISABLED":
            text = "🌐 REMOTE: ○ DISABLED"
            color = COLORS["text_dim"]
            bg = COLORS["sidebar"]
        elif s == "AUTH_ERROR":
            text = "🌐 REMOTE: ● AUTH FAIL"
            color = COLORS["danger"]
            bg = "#7F1D1D"
        elif s == "ERROR":
            text = "🌐 REMOTE: ● ERROR"
            color = COLORS["danger"]
            bg = "#7F1D1D"
        elif s.startswith("PROCESSING"):
            text = f"🌐 REMOTE: ⏳ {s}"
            color = COLORS["accent"]
            bg = "#0C4A6E"
        else:
            text = f"🌐 REMOTE: {s}"
            color = COLORS["text_dim"]
            bg = COLORS["sidebar"]

        self.webhook_status_lbl.configure(text=text, text_color=color, fg_color=bg)

        # ── Info frame labels ──
        if hasattr(self, 'webhook_url_lbl'):
            from neb_utils.webhook_client import API_BASE_URL, API_CLIENT_ID
            self.webhook_url_lbl.configure(
                text=f"Server: {API_BASE_URL or 'not set'}"
            )
            self.webhook_client_id_lbl.configure(
                text=f"Client: {API_CLIENT_ID or 'default'}"
            )
            self.webhook_interval_lbl.configure(
                text=f"Transport: SSE"
            )

    def load_hardware_devices(self):
        try:
            manager = get_device_manager()
            device_names = []
            self.uids = []
            for i in range(1, manager.DeviceInfos.Count + 1):
                info = manager.DeviceInfos(i)
                device_names.append(info.Properties("Name").Value)
                self.uids.append(info.DeviceID)

            if device_names:
                self.scanner_menu.configure(values=device_names)
                self.scanner_menu.set(device_names[0])
                self.log_message(f"🔍 Scanner detected: {device_names[0]}")
            else:
                self.scanner_menu.configure(values=["No Scanner Detected"])
                self.log_message("ℹ️ No WIA scanner device found. Use 'IMPORT FROM DISK' to load an image.")
        except Exception as e:
            self.scanner_menu.configure(values=["No Scanner Detected"])
            self.scanner_menu.set("No Scanner Detected")
            self.log_message(f"ℹ️ No scanner hardware available. Use 'IMPORT FROM DISK' to load an image.")

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=COLORS["sidebar"], border_width=2, border_color=COLORS["border"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.pack_propagate(False)

        # # Brand Header
        # self.brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        # self.brand_frame.pack(pady=(40, 20), fill="x")
        # self.logo = ctk.CTkLabel(self.brand_frame, text="NEB", font=ctk.CTkFont(family="Inter", size=32, weight="bold"), text_color=COLORS["accent"])
        # self.logo.pack()
        # Brand Header
        self.brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.brand_frame.pack(pady=(40, 20), fill="x")

        # Load Image
        try:
            from PIL import Image
            img = Image.open("media/nepal_oemblem.png")  # make sure path is correct
            self.emblem_icon = ctk.CTkImage(light_image=img, dark_image=img, size=(40, 40))
        except Exception as e:
            print("Logo Load Failed:", e)
            self.emblem_icon = None

        # Create inner frame for horizontal alignment
        self.logo_row = ctk.CTkFrame(self.brand_frame, fg_color="transparent")
        self.logo_row.pack()

        # Image (LEFT)
        self.emblem_label = ctk.CTkLabel(self.logo_row, text="", image=self.emblem_icon)
        self.emblem_label.pack(side="left", padx=(0, 8))

        # Text (RIGHT)
        self.logo = ctk.CTkLabel(
            self.logo_row,
            text="NEB",
            font=ctk.CTkFont(family="Inter", size=32, weight="bold"),
            text_color=COLORS["accent"]
        )
        self.logo.pack(side="left")
        
        self.status_pill = ctk.CTkLabel(self.brand_frame, text="● SYSTEM ONLINE", font=("Inter", 10, "bold"), text_color=COLORS["success"], fg_color="#14532D", corner_radius=20, width=120, height=24)
        self.status_pill.pack(pady=(10, 2))

        # ── Webhook Connection Status ──
        self.webhook_status_lbl = ctk.CTkLabel(
            self.brand_frame,
            text="🌐 REMOTE: ⏳",
            font=("Inter", 9, "bold"),
            text_color=COLORS["text_dim"],
            fg_color="#1E293B",
            corner_radius=20,
            width=140,
            height=20
        )
        self.webhook_status_lbl.pack(pady=(0, 10))

        self.ctrl_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ctrl_box.pack(fill="both", expand=True, padx=30)

        # Preserved Section Headers
        self.add_section_header("HARDWARE SOURCE")
        self.scanner_menu = ctk.CTkOptionMenu(self.ctrl_box, values=["No Scanner Detected"], fg_color="#1E293B", button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        self.scanner_menu.pack(fill="x", pady=(0, 20))

        self.add_section_header("ACADEMIC YEAR (BS)")
        current_bs = datetime.datetime.now().year + 57
        years = [str(y) for y in range(current_bs, current_bs - 15, -1)]
        self.year_box = ctk.CTkComboBox(self.ctrl_box, values=years, border_color=COLORS["border"], fg_color="#1E293B")
        self.year_box.pack(fill="x", pady=(0, 20))

        self.add_section_header("EXAMINATION TYPE")
        self.exam_btn = ctk.CTkSegmentedButton(self.ctrl_box, values=["Regular", "Partial", "Supplementary"], selected_color=COLORS["accent"])
        self.exam_btn.pack(fill="x", pady=(0, 20))
        self.exam_btn.set("Regular")

        self.add_section_header("GRADE")
        self.grade_btn = ctk.CTkSegmentedButton(self.ctrl_box, values=["11", "12"], selected_color=COLORS["accent"])
        self.grade_btn.pack(fill="x", pady=(0, 20))
        self.grade_btn.set("11")

        # Action Buttons
        self.scan_btn = ctk.CTkButton(self.sidebar, text="START SCAN", command=self.run_hardware_scan, height=55, corner_radius=12, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], font=("Inter", 14, "bold"))
        self.scan_btn.pack(side="bottom", fill="x", padx=30, pady=(10, 20))

        self.import_btn = ctk.CTkButton(self.sidebar, text="IMPORT FROM DISK", command=self.select_file, height=45, corner_radius=12, fg_color="transparent", border_width=1, border_color=COLORS["border"])
        self.import_btn.pack(side="bottom", fill="x", padx=30, pady=(0, 5))

        self.import_folder_btn = ctk.CTkButton(
            self.sidebar, text="📁 IMPORT FOLDER", command=self.select_folder,
            height=45, corner_radius=12, fg_color="transparent",
            border_width=1, border_color=COLORS["border"],
            font=("Inter", 13, "bold")
        )
        self.import_folder_btn.pack(side="bottom", fill="x", padx=30, pady=(0, 5))

        # Auto-commit checkbox row
        self.auto_commit_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.auto_commit_frame.pack(side="bottom", fill="x", padx=35, pady=(0, 5))

        self.auto_commit_cb = ctk.CTkCheckBox(
            self.auto_commit_frame, text="Auto-commit to DB",
            variable=self.auto_commit, command=self._toggle_auto_commit,
            font=("Inter", 12, "bold"), text_color=COLORS["text_dim"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            checkmark_color="white", border_color=COLORS["border"]
        )
        self.auto_commit_cb.pack(side="left")

        # ── Webhook Configuration Info ──
        self.webhook_info_frame = ctk.CTkFrame(
            self.sidebar, fg_color="#1E293B", corner_radius=10, height=70
        )
        self.webhook_info_frame.pack(side="bottom", fill="x", padx=20, pady=(0, 5))
        self.webhook_info_frame.pack_propagate(False)

        # Server URL
        self.webhook_url_lbl = ctk.CTkLabel(
            self.webhook_info_frame,
            text="Server: connecting...",
            font=("JetBrains Mono", 9),
            text_color=COLORS["text_dim"],
            anchor="w"
        )
        self.webhook_url_lbl.pack(fill="x", padx=12, pady=(8, 0))

        # Client ID + Poll interval row
        info_row = ctk.CTkFrame(self.webhook_info_frame, fg_color="transparent")
        info_row.pack(fill="x", padx=12, pady=(0, 6))

        self.webhook_client_id_lbl = ctk.CTkLabel(
            info_row,
            text="Client: ...",
            font=("JetBrains Mono", 9),
            text_color=COLORS["text_dim"]
        )
        self.webhook_client_id_lbl.pack(side="left")

        self.webhook_interval_lbl = ctk.CTkLabel(
            info_row,
            text="Poll: ...",
            font=("JetBrains Mono", 9),
            text_color=COLORS["text_dim"]
        )
        self.webhook_interval_lbl.pack(side="right")

        # Time Widget
        self.stats_box = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=15, height=60)
        self.stats_box.pack(side="bottom", fill="x", padx=20, pady=(5, 20))
        self.time_lbl = ctk.CTkLabel(self.stats_box, text="00:00:00", font=("JetBrains Mono", 16, "bold"), text_color=COLORS["text_main"])
        self.time_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def add_section_header(self, text):
        lbl = ctk.CTkLabel(self.ctrl_box, text=text, font=("Inter", 10, "bold"), text_color=COLORS["text_dim"], anchor="w")
        lbl.pack(fill="x", pady=(10, 5))

    def setup_main_view(self):
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, padx=40, pady=40, sticky="nsew")

        # Preview Card
        self.view_card = ctk.CTkFrame(self.main_view, fg_color=COLORS["card"], corner_radius=25, border_width=1, border_color=COLORS["border"])
        self.view_card.pack(expand=True, fill="both", pady=(0, 20))
        
        self.display_label = ctk.CTkLabel(self.view_card, text="CORE IDLE\nFeed image via scanner or disk", font=("Inter", 15), text_color=COLORS["text_dim"])
        self.display_label.pack(expand=True, fill="both", padx=20, pady=20)

        # Bindings
        self.display_label.bind("<ButtonPress-1>", self.start_drag)
        self.display_label.bind("<B1-Motion>", self.do_drag)
        self.view_card.bind("<MouseWheel>", self.zoom_image)
        self.display_label.bind("<MouseWheel>", self.zoom_image)

        # Variables
        self.zoom_level = 1.0
        self.drag_data = {"mouse_x": 0, "mouse_y": 0, "label_x": 0, "label_y": 0}

        # Floating Toolbar
        self.float_bar = ctk.CTkFrame(self.view_card, fg_color=COLORS["dock"], height=55, corner_radius=18, border_width=1, border_color=COLORS["border"])
        self.float_bar.place(relx=0.5, rely=0.92, anchor="center")
        
        self.create_dock_btn(self.float_bar, "⟳", self.rotate_image, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "✂", self.toggle_crop_mode, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "↶", self.undo_last_action, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "📥", self.save_image, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "⎙", self.print_image, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "🗑", self.clear_canvas, COLORS["danger"])

        # Console Log
        self.log_box = ctk.CTkTextbox(self.main_view, height=100, fg_color="#020617", border_width=1, border_color=COLORS["border"], font=("JetBrains Mono", 11), text_color=COLORS["success"])
        self.log_box.pack(fill="x", pady=(0, 20))

        # --- PROFESSIONAL ACTION BAR (Side-by-Side) ---
        self.button_row = ctk.CTkFrame(self.main_view, fg_color="transparent")
        self.button_row.pack(fill="x")

        self.proceed_btn = ctk.CTkButton(
            self.button_row, text="PROCEED TO OCR ENGINE", 
            command=self.run_ocr, width=350, height=60, 
            corner_radius=30, fg_color=COLORS["success"], 
            font=("Inter", 16, "bold")
        )
        self.proceed_btn.pack(side="left", expand=True, padx=(0, 5))

        self.db_save_btn = ctk.CTkButton(
            self.button_row, 
            text="📥 COMMIT TO DATABASE", 
            command=self.save_to_database,
            width=350, height=60, 
            corner_radius=30,
            fg_color="#334155", # Professional slate for disabled state
            state="disabled",
            font=("Inter", 16, "bold")
        )
        self.db_save_btn.pack(side="right", expand=True, padx=(5, 0))


    def start_drag(self, event):
        """Record the global mouse position and the current label position."""
        if self.selected_file:
            self.drag_data["mouse_x"] = event.x_root
            self.drag_data["mouse_y"] = event.y_root
            self.drag_data["label_x"] = self.display_label.winfo_x()
            self.drag_data["label_y"] = self.display_label.winfo_y()
            self.display_label.configure(cursor="fleur")

    def do_drag(self, event):
        """Calculate movement based on global screen delta."""
        if self.selected_file:
            dx = event.x_root - self.drag_data["mouse_x"]
            dy = event.y_root - self.drag_data["mouse_y"]
            new_x = self.drag_data["label_x"] + dx
            new_y = self.drag_data["label_y"] + dy
            self.display_label.place(x=new_x, y=new_y, anchor="nw")

    def zoom_image(self, event):
        """Resizes the image and centers it."""
        if not self.selected_file: return
        if event.delta > 0: self.zoom_level *= 1.1
        else: self.zoom_level /= 1.1
        self.zoom_level = max(0.2, min(self.zoom_level, 4.0))
        self.display_image()

    def create_dock_btn(self, parent, icon, command, color):
        btn = ctk.CTkButton(parent, text=icon, width=60, height=40, corner_radius=12, 
                             fg_color="transparent", text_color=color,
                             hover_color="#334155", font=("Inter", 22), command=command)
        btn.pack(side="left", padx=10, pady=5)

    def run_hardware_scan(self):
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.load_info.pack(expand=True, pady=(0, 10))
        self.prog.pack(pady=10)
        self.prog.start()
        self.update() 
        self.log_message("Initializing WIA hardware...")
        scan_thread = threading.Thread(target=self.scan_thread_logic, daemon=True)
        scan_thread.start()
    
    def scan_thread_logic(self):
        pythoncom.CoInitialize() # Fixes "CoInitialize has not been called"
        try:
            # Get UI selections
            selected_year = self.year_box.get()
            selected_exam = self.exam_btn.get()
            selected_grade = self.grade_btn.get()
            roman_grade = to_roman(selected_grade)
            year_folder = os.path.join(BASE_SAVE_PATH, selected_year)
            exam_folder_name = f"{selected_year} {roman_grade} {selected_exam}"
            final_folder = os.path.join(year_folder, exam_folder_name)
            os.makedirs(final_folder, exist_ok=True)
            save_path = get_next_filename(final_folder)
            
            selected_name = self.scanner_menu.get()
            target_uid = self.uids[self.scanner_menu._values.index(selected_name)]
            connected_device = connect_to_device_by_uid(device_uid=target_uid)
            img = scan_side(device=connected_device)
            # save_path = os.path.join(os.path.dirname(__file__), "scanned_output.jpg")
            img.save(save_path, "JPEG", quality=95)
            self.selected_file = save_path
            self.after(0, self.finish_scan_ui)
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda m=error_msg: messagebox.showerror("Hardware Error", m))
            self.after(0, lambda: self.overlay.place_forget())
        finally:
            pythoncom.CoUninitialize()

    def finish_scan_ui(self):
        self.overlay.place_forget()
        self.prog.stop()
        if self.selected_file:
            self.display_image() # Call your existingPIL/CTK display logic
            self.log_message("Buffer digitized: [SUCCESS]")

    def run_ocr(self):
        if self.selected_file:
            self.proceed_btn.configure(state="disabled", text="", fg_color=COLORS["sidebar"])
            self.btn_loader = ctk.CTkProgressBar(self.proceed_btn, width=280, height=8, 
                                                mode="indeterminate", 
                                                progress_color=COLORS["accent"])
            self.btn_loader.place(relx=0.5, rely=0.5, anchor="center")
            self.btn_loader.start()
            self.log_message("OCR Engine: Extracting high-precision data...")
            threading.Thread(target=self.ocr_thread_logic, daemon=True).start()
        else:
            messagebox.showwarning("System", "Input buffer empty. Please scan a document first.")

    def ocr_thread_logic(self, img_path=None):
        target = img_path or self.selected_file
        try:
            data_dict, summary_text = get_vlm_result(target)

            self.after(0, lambda: self.finalize_ocr_button(data_dict, img_path=target))
            self.after(0, lambda: self.log_message(summary_text))

        except Exception as e:
            err = f"VLM OCR Error on {os.path.basename(target)}: {str(e)}"
            self.after(0, lambda: self.log_message(err))
            self.after(0, lambda: self.finalize_ocr_button(error=True, img_path=target))

    def finalize_ocr_button(self, results=None, error=None, img_path=None):
        if hasattr(self, 'btn_loader') and self.btn_loader.winfo_exists():
            self.btn_loader.stop()
            self.btn_loader.destroy()

        if results:
            self.current_vlm_raw = results
            self.current_ocr_data = results.get("students", [])

            # --- Auto-update UI inputs from VLM extraction ---
            self._sync_ui_from_vlm(results)

            student_count = len(self.current_ocr_data)
            img_name = os.path.basename(img_path) if img_path else self.selected_file
            img_name = os.path.basename(img_name) if img_name else "?"

            self.log_message(f"✅ {img_name}: {student_count} students extracted.")

            # --- Auto-commit if checkbox is checked ---
            if self.auto_commit.get():
                self.log_message("⏳ Auto-committing to database...")
                img_target = img_path or self.selected_file
                threading.Thread(target=self._auto_save_to_db, args=(img_target,), daemon=True).start()
            else:
                self.proceed_btn.configure(state="normal", text="PROCEED TO OCR ENGINE", fg_color=COLORS["success"])
                self.db_save_btn.configure(state="normal", fg_color=COLORS["success"])
                self.log_message("System: Records verified. Ready for Database Commit.")

        if error:
            self.proceed_btn.configure(state="normal", text="PROCEED TO OCR ENGINE", fg_color=COLORS["success"])
            self.log_message(f"VLM OCR Error: {error}")

    def _toggle_auto_commit(self):
        """Enable/disable DB save button based on auto-commit checkbox."""
        if self.auto_commit.get():
            self.db_save_btn.configure(state="disabled", fg_color=COLORS["border"],
                                        text="📥 AUTO-COMMIT ON")
        else:
            if self.current_ocr_data:
                self.db_save_btn.configure(state="normal", fg_color=COLORS["success"],
                                            text="📥 COMMIT TO DATABASE")
            else:
                self.db_save_btn.configure(state="disabled", fg_color=COLORS["border"],
                                            text="📥 COMMIT TO DATABASE")

    def _auto_save_to_db(self, img_path: str):
        """Save OCR result to DB without confirmation dialog (used in auto-commit mode)."""
        if not self.current_vlm_raw:
            return

        meta = {
            "grade": self.grade_btn.get(),
            "exam_type": self.exam_btn.get(),
            "exam_year": self.year_box.get(),
            "book_name": None,
            "qc_check": "0",
            "qc_remarks": None,
            "cluster_id": None,
            "ui": "True",
            "remarks": None,
            "is_legacy_image": False,
        }

        try:
            save_vlm_to_database(self.current_vlm_raw, img_path, meta=meta)
            student_count = len(self.current_ocr_data)
            self.log_message(f"💾 Auto-committed {student_count} records to DB.")
        except Exception as e:
            self.log_message(f"❌ Auto-commit failed: {e}")

        if not self.is_batch_mode:
            self.proceed_btn.configure(state="normal", text="PROCEED TO OCR ENGINE",
                                        fg_color=COLORS["success"])

    def select_folder(self):
        """Open folder dialog and start batch OCR processing with resume support."""
        folder_path = filedialog.askdirectory(title="Select Folder with Mark Sheet Images")
        if not folder_path:
            return

        folder_abs = os.path.abspath(folder_path)

        # Gather all image files, sorted
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = []
        for fname in sorted(os.listdir(folder_abs)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                all_files.append(fname)

        if not all_files:
            messagebox.showwarning("Empty Folder", "No image files found in the selected folder.")
            return

        # --- Check for existing checkpoint (resume support) ---
        checkpoint = load_checkpoint()
        resume_remaining = None
        completed_originals = set()
        file_map = {}

        if checkpoint and checkpoint.get("source_folder") == folder_abs:
            resume_remaining = checkpoint.get("remaining", [])
            completed_originals = set(checkpoint.get("completed", []))
            file_map = checkpoint.get("file_map", {})

            done_count = len(completed_originals)
            total = checkpoint.get("total", len(all_files))
            if resume_remaining:
                answer = messagebox.askyesno(
                    "Resume Batch",
                    f"Found existing checkpoint for this folder.\n"
                    f"{done_count} of {total} images already completed.\n\n"
                    f"Resume processing the remaining {len(resume_remaining)}?",
                    icon="question"
                )
                if not answer:
                    delete_checkpoint()
                    resume_remaining = None
                    completed_originals = set()
                    file_map = {}

        # Build the work queue
        if resume_remaining is not None:
            # Resume mode: use remaining list from checkpoint
            work_queue = list(resume_remaining)
            skip_count = len(completed_originals)
        else:
            # Fresh start: process All files
            work_queue = list(all_files)
            completed_originals = set()
            file_map = {}
            skip_count = 0

        if not work_queue:
            messagebox.showinfo("All Done", "All images in this folder have already been processed!")
            return

        self.batch_files = work_queue
        self.batch_total = len(all_files)
        self.batch_index = 0
        self.batch_success = skip_count
        self.batch_fail = 0
        self.batch_abort = False
        self.is_batch_mode = True
        self.batch_folder = folder_abs
        self.batch_file_map = file_map
        self.batch_completed_originals = completed_originals

        self.log_message(f"\n{'='*60}")
        self.log_message(f"📂 BATCH MODE: '{os.path.basename(folder_abs)}'")
        self.log_message(f"   Total in folder: {self.batch_total}  |  Already completed: {skip_count}  |  Queue: {len(work_queue)}")
        self.log_message(f"{'='*60}")

        # Disable UI during batch
        self.import_btn.configure(state="disabled")
        self.import_folder_btn.configure(state="disabled", text="⏳ PROCESSING...")
        self.proceed_btn.configure(state="disabled", text="BATCH ACTIVE")
        self.scan_btn.configure(state="disabled")

        threading.Thread(target=self.process_folder_thread, daemon=True).start()

    def process_folder_thread(self):
        """Process all images in batch mode sequentially with checkpoint updates."""
        while self.batch_index < len(self.batch_files) and not self.batch_abort:
            orig_filename = self.batch_files[self.batch_index]
            self.batch_index += 1

            # Build source path
            img_path = os.path.join(self.batch_folder, orig_filename)

            idx = self.batch_index
            total = self.batch_total

            try:
                # Copy image to save folder
                selected_year = self.year_box.get()
                selected_exam = self.exam_btn.get()
                selected_grade = self.grade_btn.get()
                roman_grade = to_roman(selected_grade)
                year_folder = os.path.join(BASE_SAVE_PATH, selected_year)
                exam_folder = os.path.join(year_folder, f"{selected_year} {roman_grade} {selected_exam}")
                os.makedirs(exam_folder, exist_ok=True)
                saved_path = save_image_smart(img_path, exam_folder)

                # Update preview
                self.after(0, lambda p=saved_path: self._set_preview(p))
                self.after(0, lambda i=idx, t=total, f=orig_filename:
                    self.log_message(f"\n[{i}/{t}] Processing: {f}"))

                # Run OCR
                data_dict, _ = get_vlm_result(saved_path)

                # --- Rename image to meaningful name ---
                new_name = build_image_filename(data_dict)
                renamed_path = os.path.join(exam_folder, new_name)
                if os.path.normpath(saved_path) != os.path.normpath(renamed_path):
                    # Remove existing file with same name if any
                    if os.path.exists(renamed_path):
                        os.remove(renamed_path)
                    os.rename(saved_path, renamed_path)
                else:
                    renamed_path = saved_path

                # Track file mapping for checkpoint
                self.batch_file_map[orig_filename] = new_name

                self.current_vlm_raw = data_dict
                self.current_ocr_data = data_dict.get("students", [])
                student_count = len(self.current_ocr_data)

                # --- Dedup check: skip students already in DB ---
                saved_count = 0
                skipped_count = 0

                # Batch mode always saves to DB
                meta = {
                    "grade": self.grade_btn.get(),
                    "exam_type": self.exam_btn.get(),
                    "exam_year": self.year_box.get(),
                    "book_name": None,
                    "qc_check": "0",
                    "qc_remarks": None,
                    "cluster_id": None,
                    "ui": "True",
                    "remarks": None,
                    "is_legacy_image": False,
                }

                try:
                    conn = _get_db()
                    exam_id = get_exam_id(conn, meta["exam_year"], meta["grade"], meta["exam_type"])

                    if exam_id:
                        # Check each student — skip if already in DB
                        new_students = []
                        for s in data_dict.get("students", []):
                            reg = s.get("registration_number", "") if isinstance(s, dict) else s.registration_number
                            if is_student_already_in_db(conn, reg, exam_id):
                                skipped_count += 1
                            else:
                                new_students.append(s)

                        conn.close()

                        if new_students:
                            # Save only new students
                            filtered_data = dict(data_dict)
                            filtered_data["students"] = new_students
                            save_vlm_to_database(filtered_data, renamed_path, meta=meta)
                            saved_count = len(new_students)
                    else:
                        conn.close()
                        save_vlm_to_database(data_dict, renamed_path, meta=meta)
                        saved_count = student_count

                except Exception as db_err:
                    self.after(0, lambda e=db_err: self.log_message(f"⚠️ DB check failed (saving anyway): {e}"))
                    save_vlm_to_database(data_dict, renamed_path, meta=meta)
                    saved_count = student_count

                log_msg = f"💾 [{idx}/{total}] {new_name}: {saved_count} saved"
                if skipped_count:
                    log_msg += f", {skipped_count} already in DB (skipped)"
                self.after(0, lambda m=log_msg: self.log_message(m))

                # --- Update checkpoint ---
                self.batch_completed_originals.add(orig_filename)
                remaining = [f for f in self.batch_files if f not in self.batch_completed_originals]
                self.after(0, lambda sf=self.batch_folder, tt=self.batch_total,
                           fm=self.batch_file_map, co=self.batch_completed_originals, rem=remaining:
                    save_checkpoint(sf, tt, dict(fm), list(co), rem))

                self.batch_success += 1

            except Exception as e:
                self.batch_fail += 1
                self.after(0, lambda i=idx, t=total, f=orig_filename, err=str(e):
                    self.log_message(f"❌ [{i}/{t}] {f} FAILED: {err}"))

        # Batch complete — clean up UI
        self.after(0, self._finish_batch)

    def _finish_batch(self):
        """Re-enable UI after batch processing completes and clean up checkpoint."""
        self.is_batch_mode = False
        self.import_btn.configure(state="normal")
        self.import_folder_btn.configure(state="normal", text="📁 IMPORT FOLDER")
        self.proceed_btn.configure(state="normal", text="PROCEED TO OCR ENGINE", fg_color=COLORS["success"])
        self.scan_btn.configure(state="normal")

        total = self.batch_total
        success = self.batch_success
        fail = self.batch_fail

        # Delete checkpoint ONLY if all succeeded (no remaining failures)
        if fail == 0 and success >= total:
            delete_checkpoint()
            self.log_message("🗑️ Checkpoint cleared — batch fully complete.")
        else:
            self.log_message("📝 Checkpoint preserved — resume available on restart.")

        self.log_message(f"\n{'='*60}")
        self.log_message(f"📊 BATCH RESULT: {success}/{total} succeeded")
        if fail > 0:
            self.log_message(f"   ❌ {fail} failed")
        self.log_message(f"{'='*60}")

        self.log_message("System: Batch processing finished.")

    def _set_preview(self, img_path: str):
        """Update the preview display with a specific image (thread-safe)."""
        self.selected_file = img_path
        if img_path and os.path.exists(img_path):
            self.rotation = 0
            self.zoom_level = 1.0
            self.display_image()

    def _sync_ui_from_vlm(self, data: dict):
        """Sync the sidebar input widgets with values extracted by the VLM."""
        # --- Year ---
        extracted_year = str(data.get("examination_year", "")).strip()
        if extracted_year.isdigit():
            try:
                current_years = self.year_box.cget("values")
                if extracted_year in current_years:
                    self.year_box.set(extracted_year)
                else:
                    # Add it and select
                    updated = list(current_years) + [extracted_year]
                    updated = sorted(set(updated), key=lambda x: int(x), reverse=True)
                    self.year_box.configure(values=updated)
                    self.year_box.set(extracted_year)
            except Exception:
                pass

        # --- Grade ---
        extracted_grade = str(data.get("grade", "")).strip().lower()
        grade_map = {"eleven": "11", "twelve": "12", "11": "11", "12": "12"}
        if extracted_grade in grade_map:
            try:
                self.grade_btn.set(grade_map[extracted_grade])
            except Exception:
                pass

        # --- Exam Type ---
        extracted_type = str(data.get("exame_Type", "")).strip()
        exam_type_map = {
            "regular": "Regular",
            "partial": "Partial",
            "supplementary": "Supplementary",
        }
        mapped_type = exam_type_map.get(extracted_type.lower())
        if mapped_type:
            try:
                self.exam_btn.set(mapped_type)
            except Exception:
                pass
    
    def save_to_database(self):
        if not self.current_ocr_data:
            self.log_message("Database Error: No data buffer found to commit.")
            return

        # --- Confirmation dialog ---
        if not self._confirm_db_commit():
            self.log_message("Database Commit: Cancelled by user.")
            return

        self.log_message("Database: Initializing secure handshake...")
        self.db_save_btn.configure(state="disabled", text="⌛ COMMITTING...")
        
        threading.Thread(target=self.db_thread_logic, daemon=True).start()

    def _confirm_db_commit(self) -> bool:
        """Show a confirmation dialog with extracted details before DB commit."""
        raw = self.current_vlm_raw
        if not raw:
            return False

        school = raw.get("school_name", "?")
        school_code = raw.get("school_code", "?")
        grade = raw.get("grade", "?")
        exam_type = raw.get("exame_Type", "?")
        year = raw.get("examination_year", "?")
        page = raw.get("page_number", "?")
        students = raw.get("students", [])

        # Build detailed summary
        detail_lines = [
            f"School : {school} ({school_code})",
            f"Grade  : {grade}",
            f"Type   : {exam_type}",
            f"Year   : {year}",
            f"Page   : {page}",
            f"Students: {len(students)}",
            "",
            "─" * 40,
        ]

        # Show first few students as preview
        preview_count = min(5, len(students))
        for idx in range(preview_count):
            s = students[idx]
            name = s.get("student_name", "?")
            symbol = s.get("symbol_number", "?")
            total = s.get("grand_total", "?")
            remark = s.get("remark", "?")
            detail_lines.append(f"{idx+1}. {name}  [{symbol}]  Total: {total}  {remark}")

        if len(students) > preview_count:
            detail_lines.append(f"   ... and {len(students) - preview_count} more")

        detail_lines.append("")
        detail_lines.append("Override values from sidebar will be used for:")
        detail_lines.append(f"  Grade (UI): {self.grade_btn.get()}")
        detail_lines.append(f"  Type  (UI): {self.exam_btn.get()}")
        detail_lines.append(f"  Year  (UI): {self.year_box.get()}")

        details = "\n".join(detail_lines)

        # Use a custom CTkToplevel dialog for a polished look
        dialog = ctk.CTkToplevel(self)
        dialog.title("Confirm Database Commit")
        dialog.geometry("520x520")
        dialog.configure(fg_color=COLORS["bg"])
        dialog.transient(self)
        dialog.grab_set()

        # Header
        header = ctk.CTkLabel(
            dialog, text="📋 Confirm OCR Data Before Saving",
            font=("Inter", 16, "bold"), text_color=COLORS["accent"]
        )
        header.pack(pady=(20, 10))

        # Separator
        sep = ctk.CTkFrame(dialog, height=2, fg_color=COLORS["border"])
        sep.pack(fill="x", padx=30, pady=(0, 10))

        # Scrollable detail area
        text_box = ctk.CTkTextbox(
            dialog, height=280, fg_color=COLORS["card"],
            border_width=1, border_color=COLORS["border"],
            font=("JetBrains Mono", 12), text_color=COLORS["text_main"]
        )
        text_box.pack(fill="both", expand=True, padx=30, pady=(0, 15))
        text_box.insert("1.0", details)
        text_box.configure(state="disabled")

        # Button row
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(0, 20))

        result = [False]  # mutable capture for closures

        def on_confirm():
            result[0] = True
            dialog.destroy()

        def on_cancel():
            result[0] = False
            dialog.destroy()

        cancel_btn = ctk.CTkButton(
            btn_frame, text="CANCEL", command=on_cancel,
            width=180, height=45, corner_radius=12,
            fg_color="transparent", border_width=1, border_color=COLORS["border"],
            font=("Inter", 13, "bold")
        )
        cancel_btn.pack(side="left", expand=True, padx=(0, 10))

        confirm_btn = ctk.CTkButton(
            btn_frame, text="✅ CONFIRM & SAVE", command=on_confirm,
            width=180, height=45, corner_radius=12,
            fg_color=COLORS["success"], font=("Inter", 13, "bold")
        )
        confirm_btn.pack(side="right", expand=True, padx=(10, 0))

        # Center on parent
        self.wait_window(dialog)

        return result[0]
    
    def db_thread_logic(self):
        try:
            if not self.current_vlm_raw:
                raise ValueError("No VLM OCR data available. Run OCR first.")

            # Prepare metadata from UI selections
            meta = {
                "grade": self.grade_btn.get(),
                "exam_type": self.exam_btn.get(),
                "exam_year": self.year_box.get(),
                "book_name": None,
                "qc_check": "0",
                "qc_remarks": None,
                "cluster_id": None,
                "ui": "True",
                "remarks": None,
                "is_legacy_image": False,
            }

            # Save to normalized 6-table schema
            save_vlm_to_database(
                self.current_vlm_raw,
                self.selected_file,
                meta=meta
            )

            student_count = len(self.current_ocr_data)

            # Finalize UI
            self.after(0, lambda: self.log_message(f"Database: [COMMIT_SUCCESS] {student_count} records saved to normalized tables."))
            self.after(0, lambda: self.db_save_btn.configure(text="📥 COMMIT TO DATABASE", fg_color=COLORS["border"]))
            self.after(0, lambda: messagebox.showinfo("Success", f"Successfully saved {student_count} student records to NEB Database."))

        except Exception as e:
            error_msg = f"Database Critical Failure: {str(e)}"
            self.after(0, lambda: self.log_message(error_msg))
            self.after(0, lambda: self.db_save_btn.configure(state="normal", text="📥 RETRY COMMIT", fg_color=COLORS["danger"]))

    def log_message(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Digital Imaging", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        
        ext = os.path.splitext(path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            messagebox.showerror("Security", "Unsupported file format. Please select an image.")
            self.log_message(f"Security Alert: Blocked unauthorized file type ({ext})")
            return
        
        try:
            # ---- Get selections ----
            selected_year = self.year_box.get()
            selected_exam = self.exam_btn.get()
            selected_grade = self.grade_btn.get()
            roman_grade = to_roman(selected_grade)
            year_folder = os.path.join(BASE_SAVE_PATH, selected_year)
            exam_folder_name = f"{selected_year} {roman_grade} {selected_exam}"
            final_folder = os.path.join(year_folder, exam_folder_name)
            os.makedirs(final_folder, exist_ok=True)
            self.selected_file = save_image_smart(path, final_folder)
            self.rotation = 0
            self.display_image()
            self.log_message(f"Secure Input: {os.path.basename(path)} loaded.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self):
        if self.selected_file:
            try:
                self.display_label.configure(image="") 
                img = Image.open(self.selected_file).rotate(-self.rotation, expand=True)
                w, h = img.size
                base_ratio = min(800/w, 500/h)
                
                final_w = int(w * base_ratio * self.zoom_level)
                final_h = int(h * base_ratio * self.zoom_level)
                self.current_preview_img = ctk.CTkImage(
                    light_image=img, 
                    dark_image=img, 
                    size=(final_w, final_h)
                )
                card_w = self.view_card.winfo_width()
                card_h = self.view_card.winfo_height()
                if card_w <= 1: card_w = 850 
                if card_h <= 1: card_h = 550
                
                pos_x = (card_w - final_w) // 2
                pos_y = (card_h - final_h) // 2
                self.display_label.configure(image=self.current_preview_img, text="", cursor="hand2")
                self.display_label.place(x=pos_x, y=pos_y, anchor="nw")
                
                self.update_idletasks()
                
            except Exception as e:
                self.log_message(f"Display Error: {str(e)}")

    def rotate_image(self):
        if self.selected_file:
            try:
                with Image.open(self.selected_file) as img:
                    rotated_img = img.rotate(-90, expand=True)
                    rotated_img.save(self.selected_file)
                self.display_image()
                self.log_message(f"Image physically rotated and saved.")
                
            except Exception as e:
                self.log_message(f"Rotation Error: {str(e)}")

    def toggle_crop_mode(self):
        if not self.selected_file: return
        self.is_crop_mode = not self.is_crop_mode
        
        if self.is_crop_mode:
            self.display_label.configure(cursor="cross")
            self.log_message("CROP MODE: Draw a box over the image to cut.")
            # 1. UNBIND Dragging so the image stays still
            self.display_label.unbind("<ButtonPress-1>")
            self.display_label.unbind("<B1-Motion>")
            # 2. BIND Crop Events
            self.display_label.bind("<ButtonPress-1>", self.start_crop_select)
            self.display_label.bind("<B1-Motion>", self.draw_crop_rect)
            self.display_label.bind("<ButtonRelease-1>", self.execute_crop)
        else:
            self.display_label.configure(cursor="hand2")
            # 3. REBIND Dragging when Crop is OFF
            self.display_label.bind("<ButtonPress-1>", self.start_drag)
            self.display_label.bind("<B1-Motion>", self.do_drag)
            self.log_message("CROP MODE: Deactivated.")

    def start_crop_select(self, event):
        self.crop_start_x = event.x
        self.crop_start_y = event.y
        
        # Initialize with 0 size, transparent background, and neon border
        self.crop_rect = ctk.CTkFrame(
            self.display_label, 
            fg_color="transparent",
            bg_color="transparent", 
            border_width=2, 
            border_color=COLORS["accent"],
            width=0, 
            height=0
        )

    def draw_crop_rect(self, event):
        if not self.crop_rect or not self.crop_rect.winfo_exists():
            return

        # Calculate dimensions
        cur_x, cur_y = event.x, event.y
        x = min(self.crop_start_x, cur_x)
        y = min(self.crop_start_y, cur_y)
        w = abs(cur_x - self.crop_start_x)
        h = abs(cur_y - self.crop_start_y)

        # UPDATED: Use configure for size and place for position
        self.crop_rect.configure(width=w, height=h)
        self.crop_rect.place(x=x, y=y)

    def execute_crop(self, event):
        if not self.crop_rect or not self.crop_rect.winfo_exists():
            return

        try:
            # 1. PREPARE UNDO: Save current state to history
            with Image.open(self.selected_file) as current_img:
                # We store a copy in memory to keep it fast
                self.image_history.append(current_img.copy())
                # Limit history to 5 steps to save RAM
                if len(self.image_history) > 5: self.image_history.pop(0)

            # 2. PERFORM CROP MATH
            ui_w, ui_h = self.display_label.winfo_width(), self.display_label.winfo_height()
            rect_x, rect_y = self.crop_rect.winfo_x(), self.crop_rect.winfo_y()
            rect_w, rect_h = self.crop_rect.winfo_width(), self.crop_rect.winfo_height()

            if rect_w < 10 or rect_h < 10: return # Ignore accidental clicks

            with Image.open(self.selected_file) as img:
                img = img.rotate(-self.rotation, expand=True)
                scale_x, scale_y = img.size[0] / ui_w, img.size[1] / ui_h
                
                # 3. CROP & AUTO-SAVE
                cropped_img = img.crop((rect_x * scale_x, rect_y * scale_y, 
                                        (rect_x + rect_w) * scale_x, 
                                        (rect_y + rect_h) * scale_y))
                
                # Overwrite the original file automatically
                cropped_img.save(self.selected_file, quality=95)
                self.rotation = 0 # Reset rotation since it's "baked" into the crop now
            
            self.log_message("Auto-Save: Image cropped and synchronized to disk.")
            
        except Exception as e:
            self.log_message(f"Crop Error: {str(e)}")
        finally:
            if self.crop_rect.winfo_exists(): self.crop_rect.destroy()
            self.toggle_crop_mode() # Exit crop mode
            self.display_image()     # Refresh UI

    def undo_last_action(self):
        if not self.image_history:
            self.log_message("Undo Warning: No further history available.")
            return

        try:
            last_img = self.image_history.pop()
            last_img.save(self.selected_file, quality=95)
            self.rotation = 0 # Reset rotation for the restored image
            self.display_image()
            self.log_message("Undo Success: Previous state restored and saved.")
            
        except Exception as e:
            self.log_message(f"Undo Error: {str(e)}")

    def save_image(self):
        if self.selected_file:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                Image.open(self.selected_file).rotate(-self.rotation, expand=True).save(path)
                self.log_message(f"Export Success: {os.path.basename(path)}")

    def print_image(self):
        if self.selected_file:
            try: os.startfile(self.selected_file, "print")
            except: messagebox.showerror("Hardware", "No printing device detected.")

    def clear_canvas(self):
        self.selected_file = None
        self.current_preview_img = None 
        self.display_label.configure(image=None, text="CORE IDLE\nWaiting for document input...")
        self.update()
        self.log_message("Image buffer flushed.")

    def setup_loading_overlay(self):
        self.overlay = ctk.CTkFrame(self.view_card, fg_color=COLORS["bg"], corner_radius=25)
    
        self.load_info = ctk.CTkLabel(self.overlay, text="ACQUIRING HARDWARE SIGNAL...", 
                                    font=("Inter", 16, "bold"), text_color=COLORS["accent"])
        self.prog = ctk.CTkProgressBar(self.overlay, width=300, mode="indeterminate", 
                                    progress_color=COLORS["accent"])

    def update_live_elements(self):
        self.time_lbl.configure(text=datetime.datetime.now().strftime("%I:%M:%S %p"))
        self.after(1000, self.update_live_elements)

if __name__ == "__main__":
    RadiantUltraScanner().mainloop()
