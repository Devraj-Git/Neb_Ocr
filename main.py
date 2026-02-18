import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import twain
import os
import datetime


class FileScannerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scanner UI")
        self.root.geometry("900x750")
        
        self.selected_file = None
        self.scanner_source = None
        self.photo_image = None
        self.scanners = self.get_scanners()
        
        self.setup_ui()
    
    def get_scanners(self):
        """Get available scanners using twain."""
        try:
            sm = twain.SourceManager(0)
            sources = [sm.GetSourceName(i) for i in range(sm.GetCount())]
            sm.CloseSource()
            return sources or ["Scanner Demo 1"]
        except:
            return ["Demo Scanner 1"]
    
    def setup_ui(self):
    # Set root background (visible in gaps)
        self.root.configure(bg="#E8EAED")
        
        # Footer frame (full width, white)
        footer_frame = tk.Frame(self.root, bg="#FFFFFF", height=35)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(0, 13))
        footer_frame.pack_propagate(False)  # prevent shrinking
        
        footer_label = tk.Label(
            footer_frame,
            text="© 2025 Radiant Info Tech Nepal",
            bg="#FFFFFF",
            fg="#202124",
            font=("Arial", 10)
        )
        footer_label.pack(expand=True)
        
        # Main content frame for left + right panes
        content_frame = tk.Frame(self.root, bg="#E8EAED")
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------------- Left Panel ----------------
        left_frame = tk.Frame(content_frame, width=300, bg="#FFFFFF")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=(0, 5))
        left_frame.pack_propagate(False)

        # Label for Scanners
        tk.Label(
            left_frame,
            text="Scanners",
            font=("Arial", 14, "bold"),
            fg="#202124",
            bg="#FFFFFF"
        ).pack(pady=10)

        # Scanner dropdown
        self.scanner_var = tk.StringVar()
        self.scanner_dropdown = ttk.Combobox(
            left_frame,
            textvariable=self.scanner_var,
            values=self.scanners,
            state="readonly"
        )
        self.scanner_dropdown.pack(pady=10, padx=10, fill=tk.X)
        self.scanner_dropdown.set(self.scanners[0] if self.scanners else "")
        
        # Year dropdown (2056 to 2100)
        # Year dropdown (2056 to current year dynamically)
        tk.Label(left_frame, text="Year", bg="#FFFFFF", fg="#202124").pack(pady=(10, 0), padx=10, anchor="w")
        self.year_var = tk.StringVar()

        # Calculate current Nepali year approximately
        current_ad_year = datetime.datetime.now().year
        current_bs_year = current_ad_year + 57  # Rough estimate; adjust if needed
        years = [str(y) for y in range(2056, current_bs_year + 1)]

        self.year_dropdown = ttk.Combobox(left_frame, textvariable=self.year_var, values=years, state="readonly")
        self.year_dropdown.pack(pady=5, padx=10, fill=tk.X)
        self.year_dropdown.set(years[0])
        
        #exam type 
        tk.Label(left_frame, text="Exam Type", bg="#FFFFFF", fg="#202124").pack(pady=(10, 0), padx=10, anchor="w")
        self.exam_var = tk.StringVar()
        exam_types = ["Regular Exam", "Partial Exam", "Supplementary Exam"]
        self.exam_dropdown = ttk.Combobox(left_frame, textvariable=self.exam_var, values=exam_types, state="readonly")
        self.exam_dropdown.pack(pady=5, padx=10, fill=tk.X)
        self.exam_dropdown.set(exam_types[0])

        # Grade dropdown
        tk.Label(left_frame, text="Grade", bg="#FFFFFF", fg="#202124").pack(pady=(10, 0), padx=10, anchor="w")
        self.grade_var = tk.StringVar()
        grades = ["11", "12"]
        self.grade_dropdown = ttk.Combobox(left_frame, textvariable=self.grade_var, values=grades, state="readonly")
        self.grade_dropdown.pack(pady=5, padx=10, fill=tk.X)
        self.grade_dropdown.set(grades[0])

        # Bottom buttons frame
        bottom_frame = tk.Frame(left_frame, bg="#FFFFFF")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Scan Button
        scan_btn = tk.Button(
            bottom_frame,
            text="Scan",
            command=self.scan_image,
            bg="#FFFFFF",
            fg="#202124",
            relief="solid",
            borderwidth=1,
            font=("Arial", 12)
        )
        scan_btn.pack(pady=5, padx=10, fill=tk.X)

        # Load File Button
        file_btn = tk.Button(
            bottom_frame,
            text="Load File",
            command=self.select_file,
            bg="#FFFFFF",
            fg="#202124",
            relief="solid",
            borderwidth=1,
            font=("Arial", 12)
        )
        file_btn.pack(pady=5, padx=10, fill=tk.X)

        # ---------------- Right Panel ----------------
        # Right panel
        self.right_frame = tk.Frame(content_frame, bg="#FFFFFF")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0, 5))
        self.right_frame.pack_propagate(False)

        # Fixed-size container for image display
        self.image_container = tk.Frame(self.right_frame, bg="#FFFFFF", width=500, height=500)
        self.image_container.pack(side=tk.TOP, pady=20, expand=True, fill=tk.BOTH)
        self.image_container.pack_propagate(False)

        self.display_label = tk.Label(
            self.image_container,
            text="Scanned Images",
            font=("Arial", 18),
            bg="#FFFFFF",
            fg="#202124"
        )
        self.display_label.pack(expand=True)

        # Proceed button at bottom of right pane
        self.proceed_btn = tk.Button(
            self.right_frame,
            text="Proceed",
            command=self.proceed,
            bg="#FFFFFF",
            fg="#202124",
            relief="solid",
            borderwidth=1,
            font=("Arial", 14)
        )
        self.proceed_btn.pack(side=tk.BOTTOM, pady=20)



    def scan_image(self):
        """Scan from selected scanner."""
        scanner_name = self.scanner_var.get()
        if not scanner_name:
            messagebox.showwarning("Warning", "Select a scanner available")
            return
        
        try:
            sm = twain.SourceManager(self.root.winfo_id())
            self.scanner_source = sm.OpenSource(scanner_name)
            if self.scanner_source:
                self.scanner_source.RequestAcquire(False, False)
                rv = self.scanner_source.XferImageNatively()
                if rv:
                    handle, count = rv
                    bmp_path = "scanned.bmp"
                    twain.DIBToBMFile(handle, bmp_path)
                    self.selected_file = bmp_path
                    self.display_image()
                self.scanner_source.CloseSource()
        except Exception as e:
            # Demo fallback
            self.selected_file = "demo_scan.png"
            messagebox.showinfo("Demo", f"Simulated scan from {scanner_name}. Using demo image.")
            self.display_image()
    
    def select_file(self):
        """Fallback file select for macOS."""
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("GIF files", "*.gif"),
            ("BMP files", "*.bmp")
        ]
        file_path = None
        try:
            try:
                self.root.update_idletasks()
                self.root.lift()
                self.root.attributes("-topmost", True)
            except Exception:
                pass

            file_path = filedialog.askopenfilename(
                parent=self.root,
                title="Select an image",
                filetypes=filetypes,
                initialdir=os.path.expanduser("~")
            )
        finally:
            try:
                self.root.attributes("-topmost", False)
            except Exception:
                pass

        if file_path:
            self.selected_file = file_path
            self.display_image()
   
    def display_image(self):
        if not self.selected_file:
            return
        try:
            image = Image.open(self.selected_file)
            image.thumbnail((500, 500), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(image)

            # Remove previous image label
            if hasattr(self, 'image_label'):
                self.image_label.destroy()

            self.image_label = tk.Label(self.image_container, image=self.photo_image, bg="white")
            self.image_label.pack(expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {str(e)}")
    
    def proceed(self):
        if self.selected_file:
            messagebox.showinfo("Proceed", f"Proceeding with scanned image: {self.selected_file}")
        else:
            messagebox.showwarning("Warning", "Scan or load a file first")


if __name__ == "__main__":
    root = tk.Tk()
    app = FileScannerUI(root)
    root.mainloop()
