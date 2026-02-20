import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import datetime
import threading
import time
from wia_scan import get_device_manager, connect_to_device_by_uid, scan_side
from ocr_front import get_ocr_result
import pythoncom
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
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_view()
        self.setup_loading_overlay()
        self.update_live_elements()
        
        # FINAL UPGRADE: Professional Initial Log
        self.load_hardware_devices()
        self.log_message("OCR Engine v5.7 initialized. Secure Image Buffer: READY.")

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
            else:
                self.scanner_menu.configure(values=["No Scanner Detected"])
        except Exception as e:
            self.log_message(f"Hardware Error: {str(e)}")

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=COLORS["sidebar"], border_width=2, border_color=COLORS["border"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.pack_propagate(False)

        # Brand Header
        self.brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.brand_frame.pack(pady=(40, 20), fill="x")
        self.logo = ctk.CTkLabel(self.brand_frame, text="NEB", font=ctk.CTkFont(family="Inter", size=32, weight="bold"), text_color=COLORS["accent"])
        self.logo.pack()
        
        self.status_pill = ctk.CTkLabel(self.brand_frame, text="● SYSTEM ONLINE", font=("Inter", 10, "bold"), text_color=COLORS["success"], fg_color="#14532D", corner_radius=20, width=120, height=24)
        self.status_pill.pack(pady=10)

        self.ctrl_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ctrl_box.pack(fill="both", expand=True, padx=30)

        # Preserved Section Headers
        self.add_section_header("HARDWARE SOURCE")
        self.scanner_menu = ctk.CTkOptionMenu(self.ctrl_box, values=["Virtual Engine v5", "WIA Direct-Link"], fg_color="#1E293B", button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
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
        self.import_btn.pack(side="bottom", fill="x", padx=30, pady=0)

        # Time Widget
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

        # Preview Card
        self.view_card = ctk.CTkFrame(self.main_view, fg_color=COLORS["card"], corner_radius=25, border_width=1, border_color=COLORS["border"])
        self.view_card.pack(expand=True, fill="both", pady=(0, 20))
        
        self.display_label = ctk.CTkLabel(self.view_card, text="CORE IDLE\nFeed image via scanner or disk", font=("Inter", 15), text_color=COLORS["text_dim"])
        self.display_label.pack(expand=True, fill="both", padx=20, pady=20)

        self.display_label.bind("<ButtonPress-1>", self.start_drag)
        self.display_label.bind("<B1-Motion>", self.do_drag)
        
        # Bindings for Zooming (Mouse Wheel)
        self.view_card.bind("<MouseWheel>", self.zoom_image)
        self.display_label.bind("<MouseWheel>", self.zoom_image)

        # Interaction Variables
        self.zoom_level = 1.0
        self.drag_data = {"x": 0, "y": 0}

        # UPGRADED DOCK: Professional High-Contrast Toolbar
        self.float_bar = ctk.CTkFrame(self.view_card, fg_color=COLORS["dock"], height=55, corner_radius=18, border_width=1, border_color=COLORS["border"])
        self.float_bar.place(relx=0.5, rely=0.92, anchor="center")
        
        self.create_dock_btn(self.float_bar, "⟳", self.rotate_image, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "📥", self.save_image, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "⎙", self.print_image, COLORS["text_main"])
        self.create_dock_btn(self.float_bar, "🗑", self.clear_canvas, COLORS["danger"])

        self.log_box = ctk.CTkTextbox(self.main_view, height=100, fg_color="#020617", border_width=1, border_color=COLORS["border"], font=("JetBrains Mono", 11), text_color=COLORS["success"])
        self.log_box.pack(fill="x", pady=(0, 20))

        self.proceed_btn = ctk.CTkButton(self.main_view, text="PROCEED TO OCR ENGINE", command=self.run_ocr, width=350, height=60, corner_radius=30, fg_color=COLORS["success"], font=("Inter", 16, "bold"))
        self.proceed_btn.pack()

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
            selected_name = self.scanner_menu.get()
            target_uid = self.uids[self.scanner_menu._values.index(selected_name)]
            connected_device = connect_to_device_by_uid(device_uid=target_uid)
            img = scan_side(device=connected_device)
            save_path = os.path.join(os.path.dirname(__file__), "scanned_output.jpg")
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

    def ocr_thread_logic(self):
        try:
            data_list = get_ocr_result(self.selected_file)
            output = []
            output.append("="*50)
            output.append(f"TOTAL RECORDS FOUND: {len(data_list)}")
            output.append("="*50 + "\n")

            for student in data_list:
                symbol = student.get('SYMBOL', 'N/A')
                name = student.get('NAME OF THE STUDENT', 'N/A')
                total = student.get('TOTAL', 'N/A')
                rem = student.get('REM', 'N/A')
                dob = student.get('DOB', 'N/A')
                reg = student.get('REG.NO.', 'N/A')
                
                record = f"SYMBOL: {symbol} | REG.NO.: {reg}\n | NAME: {name}\n | DOB: {dob}\n" 
                record += f"REMARK: {rem} | TOTAL: {total}\n"
                record += "-"*30 + "\n"
                
                for i in range(1, 8):
                    code = student.get(f'CODE{i}')
                    if code: # Only print if the subject exists
                        th = student.get(f'TH{i}', '-')
                        pr = student.get(f'PR{i}', '-')
                        tot = student.get(f'TOT{i}', '-')
                        record += f"  > {code}: TH({th}) PR({pr}) | TOT: {tot}\n"
                
                record += "\n"
                output.append(record)

            final_text = "".join(output)
            self.after(0, lambda: self.finalize_ocr_button())
            self.after(0, lambda: self.log_message(final_text))
            self.after(0, lambda: self.log_message("OCR Sync: [DATA_DUMP_COMPLETE]"))

        except Exception as e:
            err = f"OCR Parse Error: {str(e)}"
            self.after(0, lambda: self.log_message(err))
            self.after(0, lambda: self.finalize_ocr_button(error=True))

    def finalize_ocr_button(self, results=None, error=None):
        if hasattr(self, 'btn_loader'):
            self.btn_loader.stop()
            self.btn_loader.destroy()
        self.proceed_btn.configure(state="normal", text="PROCEED TO OCR ENGINE", fg_color=COLORS["success"])
        if results:
            self.log_message(results)
            self.log_message("OCR Sequence: [SUCCESS]")
        if error:
            self.log_message(f"OCR Error: {error}")
            
    def log_message(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Digital Imaging", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                messagebox.showerror("Security", "Unsupported file format. Please select an image.")
                self.log_message(f"Security Alert: Blocked unauthorized file type ({ext})")
                return
            
            self.selected_file = path
            self.rotation = 0
            self.display_image()
            self.log_message(f"Secure Input: {os.path.basename(path)} loaded.")

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
