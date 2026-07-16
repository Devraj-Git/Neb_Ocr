import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import datetime
import threading

from neb_utils.utils import (
    save_image_smart, to_roman,
    build_image_filename,
)
from neb_utils.icons import OCR_ICON, DB_ICON, IMAGE_ICON, FOLDER_ICON, GLOBE_ICON
from neb_utils.batch_processor import BatchProcessor
from neb_utils.db_operations import DbAndOcrProcessor
from neb_utils.scanner_ops import ScannerProcessor
from neb_utils.webhook_handler import WebhookHandler
from neb_utils.image_editor import ImageEditor
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



        # ── Shared constants (also used by sub-processors) ──
        self.BASE_SAVE_PATH = BASE_SAVE_PATH
        self.COLORS = COLORS

        # ── Sub-processors (created first — UI methods reference them) ──
        self.batch_processor = BatchProcessor(self)
        self.db_processor = DbAndOcrProcessor(self)
        self.scanner_processor = ScannerProcessor(self)
        self.webhook_handler = WebhookHandler(self)
        self.image_editor = ImageEditor(self)

        self.setup_sidebar()
        self.setup_main_view()
        self.setup_loading_overlay()
        self.update_live_elements()

        # FINAL UPGRADE: Professional Initial Log
        self.scanner_processor.load_hardware_devices()

        # ── Start webhook polling client ──
        self.webhook_handler.init_client()

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
            text="REMOTE: ⏳",
            image=GLOBE_ICON,
            compound="left",
            font=("Inter", 9, "bold"),
            text_color=COLORS["text_dim"],
            fg_color="#1E293B",
            corner_radius=20,
            width=160,
            height=20
        )
        self.webhook_status_lbl.pack(pady=(0, 10))

        self.ctrl_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ctrl_box.pack(fill="both", expand=True, padx=30)

        # ── Hardware Source ──
        self.add_section_header("HARDWARE SOURCE")
        self.scanner_menu = ctk.CTkOptionMenu(self.ctrl_box, values=["No Scanner Detected"], fg_color="#1E293B", button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        self.scanner_menu.pack(fill="x", pady=(0, 6))

        self.add_section_header("ACADEMIC YEAR (BS)")
        current_bs = datetime.datetime.now().year + 57
        years = [str(y) for y in range(current_bs, current_bs - 15, -1)]
        self.year_box = ctk.CTkComboBox(self.ctrl_box, values=years, border_color=COLORS["border"], fg_color="#1E293B")
        self.year_box.pack(fill="x", pady=(0, 6))

        self.add_section_header("EXAMINATION TYPE")
        self.exam_btn = ctk.CTkSegmentedButton(self.ctrl_box, values=["Regular", "Partial", "Supplementary"], selected_color=COLORS["accent"])
        self.exam_btn.pack(fill="x", pady=(0, 6))
        self.exam_btn.set("Regular")

        self.add_section_header("GRADE")
        self.grade_btn = ctk.CTkSegmentedButton(self.ctrl_box, values=["11", "12"], selected_color=COLORS["accent"])
        self.grade_btn.pack(fill="x", pady=(0, 6))
        self.grade_btn.set("11")

        # Action Buttons (packed first → bottommost)
        self.scan_btn = ctk.CTkButton(self.sidebar, text="START SCAN", command=self.scanner_processor.run_hardware_scan, height=55, corner_radius=12, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], font=("Inter", 14, "bold"))
        self.scan_btn.pack(side="bottom", fill="x", padx=30, pady=(10, 20))

        # ── Import row: IMAGE + FOLDER side by side (above START SCAN) ──
        self.import_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.import_row.pack(side="bottom", fill="x", padx=30, pady=(0, 5))

        self.import_btn = ctk.CTkButton(
            self.import_row, text="IMAGE",
            image=IMAGE_ICON, compound="left",
            command=self.select_file, height=40, corner_radius=10,
            fg_color="transparent", border_width=1, border_color=COLORS["border"],
            font=("Inter", 11, "bold")
        )
        self.import_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.import_folder_btn = ctk.CTkButton(
            self.import_row, text="FOLDER",
            image=FOLDER_ICON, compound="left",
            command=self.batch_processor.select_folder,
            height=40, corner_radius=10, fg_color="transparent",
            border_width=1, border_color=COLORS["border"],
            font=("Inter", 11, "bold")
        )
        self.import_folder_btn.pack(side="right", fill="x", expand=True, padx=(3, 0))

        # ── IMPORT IMAGE label (above the row) ──
        self.import_lbl = ctk.CTkLabel(
            self.sidebar, text="SELECT FOR IMPORT",
            font=("Inter", 9, "bold"), text_color=COLORS["text_dim"], anchor="w"
        )
        self.import_lbl.pack(side="bottom", fill="x", padx=30, pady=(0, 3))

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

        # self.webhook_interval_lbl = ctk.CTkLabel(
        #     info_row,
        #     text="Poll: ...",
        #     font=("JetBrains Mono", 9),
        #     text_color=COLORS["text_dim"]
        # )
        # self.webhook_interval_lbl.pack(side="right")

        # Time Widget
        self.stats_box = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=15, height=60)
        self.stats_box.pack(side="bottom", fill="x", padx=20, pady=(5, 20))
        self.time_lbl = ctk.CTkLabel(self.stats_box, text="00:00:00", font=("JetBrains Mono", 16, "bold"), text_color=COLORS["text_main"])
        self.time_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def add_section_header(self, text):
        lbl = ctk.CTkLabel(self.ctrl_box, text=text, font=("Inter", 9, "bold"), text_color=COLORS["text_dim"], anchor="w")
        lbl.pack(fill="x", pady=(6, 3))

    def setup_main_view(self):
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, padx=40, pady=40, sticky="nsew")

        # Preview Card
        self.view_card = ctk.CTkFrame(self.main_view, fg_color=COLORS["card"], corner_radius=25, border_width=1, border_color=COLORS["border"])
        self.view_card.pack(expand=True, fill="both", pady=(0, 20))
        
        self.display_label = ctk.CTkLabel(self.view_card, text="CORE IDLE\nFeed image via scanner or disk", font=("Inter", 15), text_color=COLORS["text_dim"])
        self.display_label.pack(expand=True, fill="both", padx=20, pady=20)

        # Bindings
        e = self.image_editor
        self.display_label.bind("<ButtonPress-1>", e.start_drag)
        self.display_label.bind("<B1-Motion>", e.do_drag)
        self.view_card.bind("<MouseWheel>", e.zoom_image)
        self.display_label.bind("<MouseWheel>", e.zoom_image)

        # Variables
        self.zoom_level = 1.0
        self.drag_data = {"mouse_x": 0, "mouse_y": 0, "label_x": 0, "label_y": 0}

        # Floating Toolbar
        self.float_bar = ctk.CTkFrame(self.view_card, fg_color=COLORS["dock"], height=55, corner_radius=18, border_width=1, border_color=COLORS["border"])
        self.float_bar.place(relx=0.5, rely=0.92, anchor="center")
        
        e.create_dock_btn(self.float_bar, "⟳", e.rotate_image, COLORS["text_main"])
        e.create_dock_btn(self.float_bar, "✂", e.toggle_crop_mode, COLORS["text_main"])
        e.create_dock_btn(self.float_bar, "↶", e.undo_last_action, COLORS["text_main"])
        e.create_dock_btn(self.float_bar, "📥", e.save_image, COLORS["text_main"])
        e.create_dock_btn(self.float_bar, "⎙", e.print_image, COLORS["text_main"])
        e.create_dock_btn(self.float_bar, "🗑", e.clear_canvas, COLORS["danger"])

        # Console Log
        self.log_box = ctk.CTkTextbox(self.main_view, height=100, fg_color="#020617", border_width=1, border_color=COLORS["border"], font=("JetBrains Mono", 11), text_color=COLORS["success"])
        self.log_box.pack(fill="x", pady=(0, 20))

        # --- PROFESSIONAL ACTION BAR (Side-by-Side) ---
        self.button_row = ctk.CTkFrame(self.main_view, fg_color="transparent")
        self.button_row.pack(fill="x")

        self.proceed_btn = ctk.CTkButton(
            self.button_row, text="PROCEED TO OCR ENGINE", 
            image=OCR_ICON, compound="left",
            command=self.db_processor.run_ocr, width=350, height=60, 
            corner_radius=30, fg_color=COLORS["success"], 
            font=("Inter", 16, "bold")
        )
        self.proceed_btn.pack(side="left", expand=True, padx=(0, 5))

        self.db_save_btn = ctk.CTkButton(
            self.button_row, 
            text="COMMIT TO DATABASE", 
            image=DB_ICON, compound="left",
            command=self.db_processor.save_to_database,
            width=350, height=60, 
            corner_radius=30,
            fg_color="#334155", # Professional slate for disabled state
            state="disabled",
            font=("Inter", 16, "bold")
        )
        self.db_save_btn.pack(side="right", expand=True, padx=(5, 0))

        # ── Auto-commit row (below action bar, aligned under commit button) ──
        self.auto_commit_row = ctk.CTkFrame(self.main_view, fg_color="transparent")
        self.auto_commit_row.pack(fill="x", pady=(8, 0))

        # Spacer pushes the checkbox to the right (under the commit button)
        spacer = ctk.CTkLabel(self.auto_commit_row, text="", width=0)
        spacer.pack(side="left", expand=True)

        self.auto_commit_cb = ctk.CTkCheckBox(
            self.auto_commit_row, text="Auto-commit to DB (skip confirmation)",
            variable=self.auto_commit, command=self.db_processor.toggle_auto_commit,
            font=("Inter", 11, "bold"), text_color=COLORS["text_dim"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            checkmark_color="white", border_color=COLORS["border"]
        )
        self.auto_commit_cb.pack(side="right", padx=(0, 10))


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
