import tkinter as tk
from tkinter import messagebox
import os

# Import the specific functions from your wia_scan installation
from wia_scan import (
    get_device_manager, 
    connect_to_device_by_uid, 
    scan_side
)

from ocr_front import get_ocr_result

class CanonScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Canon LiDE 110 - Auto Save")
        self.root.geometry("450x350")
        
        self.uids = []
        
        tk.Label(root, text="Canon LiDE 110 Scanner", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.listbox = tk.Listbox(root, width=55, height=8)
        self.listbox.pack(padx=20, pady=5)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="🔄 Refresh", command=self.load_devices, width=12).pack(side=tk.LEFT, padx=5)
        self.scan_btn = tk.Button(btn_frame, text="📸 Scan Now", command=self.run_scan, 
                                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=15)
        self.scan_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Select scanner and click Scan.")
        self.status_label = tk.Label(root, textvariable=self.status_var, fg="gray")
        self.status_label.pack(pady=10)
        self.image_path = None
        self.load_devices()

    def load_devices(self):
        try:
            manager = get_device_manager()
            self.listbox.delete(0, tk.END)
            self.uids = []
            
            if manager.DeviceInfos.Count == 0:
                self.status_var.set("No scanner detected.")
                return

            for i in range(1, manager.DeviceInfos.Count + 1):
                info = manager.DeviceInfos(i)
                self.listbox.insert(tk.END, info.Properties("Name").Value)
                self.uids.append(info.DeviceID)
            
            self.status_var.set(f"Ready: {len(self.uids)} device(s) found.")
        except Exception as e:
            self.status_var.set(f"Error: {e}")

    def run_scan(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a scanner first!")
            return

        target_uid = self.uids[selection[0]]
        
        # --- STATIC SAVE PATH ---
        # Saves in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "scanned_output.jpg")

        try:
            self.status_var.set("Connecting...")
            self.root.update()

            # Connect using your provided logic
            connected_device = connect_to_device_by_uid(device_uid=target_uid)

            self.status_var.set("Scanning... (Wait for motor)")
            self.root.update()

            # Perform the scan
            img = scan_side(device=connected_device)
            
            # Save the image automatically
            img.save(save_path, "JPEG", quality=95)
            self.image_path = save_path
            get_ocr_result('D:/Neb_Ocr_Final/402.jpg')
            
            self.status_var.set(f"Success! Saved to {os.path.basename(save_path)}")
            messagebox.showinfo("Scan Complete", f"Image saved successfully to:\n{save_path}")
            
        except Exception as e:
            self.status_var.set("Scan Failed.")
            messagebox.showerror("Error", f"Could not scan: {e}")
            print(e)

if __name__ == "__main__":
    root = tk.Tk()
    app = CanonScannerApp(root)
    root.mainloop()
