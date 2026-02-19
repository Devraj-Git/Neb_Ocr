import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import datetime
import threading
import time

# 2026 Core Palette
COLORS = {
    "bg": "#020617",           
    "sidebar": "#0F172A",      
    "accent": "#38BDF8",       
    "accent_glow": "#0EA5E9",
    "card": "#1E293B",         
    "text_main": "#F8FAFC",
    "text_dim": "#94A3B8",     
    "success": "#22C55E",      
    "border": "#334155"
}

ctk.set_appearance_mode("Dark")

class RadiantUltraScanner(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("RADIANT | CORE OPTICAL CHARACTER READER")
        self.geometry("1280x880")
        self.configure(fg_color=COLORS["bg"])

        self.selected_file = None
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_view()
        self.setup_loading_overlay()
        self.update_live_elements()

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=COLORS["sidebar"], border_width=2, border_color=COLORS["border"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.pack_propagate(False)

        # Brand Header
        self.brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.brand_frame.pack(pady=(40, 20), fill="x")
        
        self.logo = ctk.CTkLabel(self.brand_frame, text="RADIANT", font=ctk.CTkFont(family="Inter", size=32, weight="bold"), text_color=COLORS["accent"])
        self.logo.pack()
        
        self.status_pill = ctk.CTkLabel(self.brand_frame, text="● SYSTEM ONLINE", font=("Inter", 10, "bold"), text_color=COLORS["success"], fg_color="#14532D", corner_radius=20, width=120, height=24)
        self.status_pill.pack(pady=10)

        self.ctrl_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ctrl_box.pack(fill="both", expand=True, padx=30)

        self.add_section_header("HARDWARE SOURCE")
        self.scanner_menu = ctk.CTkOptionMenu(self.ctrl_box, values=["Virtual Engine v5", "WIA Direct-Link"], fg_color="#1E293B", button_color=COLORS["accent"])
        self.scanner_menu.pack(fill="x", pady=(0, 20))

        self.add_section_header("ACADEMIC YEAR (BS)")
        current_bs = datetime.datetime.now().year + 57
        years = [str(y) for y in range(current_bs, current_bs - 15, -1)]
        self.year_box = ctk.CTkComboBox(self.ctrl_box, values=years, border_color=COLORS["border"], fg_color="#1E293B")
        self.year_box.pack(fill="x", pady=(0, 20))

        self.add_section_header("EXAMINATION TYPE")
        self.exam_btn = ctk.CTkSegmentedButton(self.ctrl_box, values=["Regular", "Partial", "Back"], selected_color=COLORS["accent"], unselected_color="#1E293B")
        self.exam_btn.pack(fill="x", pady=(0, 20))
        self.exam_btn.set("Regular")

        self.add_section_header("GRADE")
        self.grade_btn = ctk.CTkSegmentedButton(self.ctrl_box, values=["11", "12"], selected_color=COLORS["accent"], unselected_color="#1E293B")
        self.grade_btn.pack(fill="x", pady=(0, 20))
        self.grade_btn.set("11")

        # Action Buttons
        self.scan_btn = ctk.CTkButton(self.sidebar, text="START SCAN", command=self.run_process, height=55, corner_radius=12, fg_color=COLORS["accent"], font=("Inter", 14, "bold"))
        self.scan_btn.pack(side="bottom", fill="x", padx=30, pady=(10, 20))

        # ADDED: Import File Option
        self.import_btn = ctk.CTkButton(self.sidebar, text="IMPORT FROM DISK", command=self.select_file, height=45, corner_radius=12, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], font=("Inter", 12))
        self.import_btn.pack(side="bottom", fill="x", padx=30, pady=0)

        # Stats Card
        self.stats_box = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=15, height=60)
        self.stats_box.pack(side="bottom", fill="x", padx=20, pady=20)
        self.time_lbl = ctk.CTkLabel(self.stats_box, text="00:00:00", font=("JetBrains Mono", 16, "bold"), text_color=COLORS["text_main"])
        self.time_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def add_section_header(self, text):
        lbl = ctk.CTkLabel(self.ctrl_box, text=text, font=("Inter", 10, "bold"), text_color=COLORS["text_dim"], anchor="w")
        lbl.pack(fill="x", pady=(10, 5))

    def setup_main_view(self):
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, padx=40, pady=40, sticky="nsew")

        # Preview Area
        self.view_card = ctk.CTkFrame(self.main_view, fg_color=COLORS["card"], corner_radius=25, border_width=1, border_color=COLORS["border"])
        self.view_card.pack(expand=True, fill="both", pady=(0, 20))
        
        self.display_label = ctk.CTkLabel(self.view_card, text="SYSTEM IDLE\nWaiting for Signal", font=("Inter", 16), text_color=COLORS["text_dim"])
        self.display_label.pack(expand=True, fill="both", padx=20, pady=20)

        self.log_box = ctk.CTkTextbox(self.main_view, height=100, fg_color="#020617", border_width=1, border_color=COLORS["border"], font=("JetBrains Mono", 11), text_color=COLORS["success"])
        self.log_box.pack(fill="x", pady=(0, 20))
        self.log_message("System kernel initialized... [OK]")

        self.proceed_btn = ctk.CTkButton(self.main_view, text="PROCEED TO OCR", command=self.proceed, width=350, height=60, corner_radius=30, fg_color=COLORS["success"], font=("Inter", 16, "bold"))
        self.proceed_btn.pack()

    def setup_loading_overlay(self):
        self.overlay = ctk.CTkFrame(self, fg_color="#020617", corner_radius=0)
        self.load_info = ctk.CTkLabel(self.overlay, text="ACQUIRING HARDWARE BUFFER...", font=("Inter", 18, "bold"))
        self.prog = ctk.CTkProgressBar(self.overlay, width=400, mode="indeterminate", progress_color=COLORS["accent"])

    def log_message(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{timestamp}] {msg}\n")
        self.log_box.see("end")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.selected_file = path
            self.log_message(f"Local file imported: {os.path.basename(path)}")
            self.display_image()

    def display_image(self):
        if self.selected_file:
            img = Image.open(self.selected_file)
            # Dynamic resizing to fit preview
            w, h = img.size
            ratio = min(800/w, 500/h)
            new_size = (int(w*ratio), int(h*ratio))
            
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
            self.display_label.configure(image=ctk_img, text="")
            self.display_label.image = ctk_img

    def run_process(self):
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.load_info.pack(expand=True, pady=(0, 10))
        self.prog.pack(pady=10)
        self.prog.start()
        self.log_message("Hardware handshake initiated...")
        threading.Thread(target=self.mock_scan, daemon=True).start()

    def mock_scan(self):
        time.sleep(2)
        self.after(0, self.finish_scan)

    def finish_scan(self):
        self.overlay.place_forget()
        self.prog.stop()
        self.log_message("Scan buffer received. Decoding bitmap... [SUCCESS]")
        messagebox.showinfo("Scanner", "Process Complete")

    def proceed(self):
        if not self.selected_file:
            messagebox.showwarning("Warning", "No document in buffer.")
            return
        self.log_message("Sending document to OCR engine...")

    def update_live_elements(self):
        t = datetime.datetime.now().strftime("%I:%M:%S %p")
        self.time_lbl.configure(text=t)
        self.after(1000, self.update_live_elements)

if __name__ == "__main__":
    app = RadiantUltraScanner()
    app.mainloop()
