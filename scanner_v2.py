import tkinter as tk
from tkinter import messagebox, filedialog
import win32com.client
from wia_scan import list_devices, scan_side

class ScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Canon LiDE 110 Scanner")
        self.root.geometry("500x400")
        
        try:
            self.wia_manager = win32com.client.Dispatch("WIA.DeviceManager")
        except Exception as e:
            messagebox.showerror("Error", f"WIA Service Error: {e}")
            self.root.destroy()
            return

        tk.Label(root, text="Select Your Scanner:", font=("Arial", 11, "bold")).pack(pady=10)
        self.device_listbox = tk.Listbox(root, width=60, height=8)
        self.device_listbox.pack(padx=20, pady=5)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="Refresh List", command=self.refresh_devices).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Start Scan", command=self.perform_scan, bg="#2196F3", fg="white", width=15).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Searching...")
        tk.Label(root, textvariable=self.status_var, fg="gray").pack()

        self.refresh_devices()

    def refresh_devices(self):
        """Directly queries WIA DeviceInfos to ensure names show in UI."""
        try:
            # Re-sync with hardware
            self.devices = list_devices(self.wia_manager) 
            self.device_listbox.delete(0, tk.END)
            
            # Using WIA's native DeviceInfos to get the correct Display Name
            if self.wia_manager.DeviceInfos.Count == 0:
                self.status_var.set("No scanners found. Check USB and Unlock switch.")
                return

            # Loop through native Windows device info
            for i in range(1, self.wia_manager.DeviceInfos.Count + 1):
                device_info = self.wia_manager.DeviceInfos(i)
                # Property 3 is 'Name', Property 4 is 'Description'
                name = device_info.Properties("Name").Value
                self.device_listbox.insert(tk.END, name)
            
            self.status_var.set(f"Found {self.wia_manager.DeviceInfos.Count} scanner(s).")
        except Exception as e:
            self.status_var.set(f"Error: {e}")

    def perform_scan(self):
        selection = self.device_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection", "Select a scanner from the list!")
            return

        # Map UI selection back to wia_scan device object
        selected_device = self.devices[selection[0]]
        save_path = filedialog.asksaveAsFilename(defaultextension=".jpg", initialfile="scan.jpg")
        
        if save_path:
            try:
                self.status_var.set("Scanning... please wait.")
                self.root.update()
                # Execute scan using the wia_scan object
                img = scan_side(device=selected_device, dpi=300)
                img.save(save_path)
                messagebox.showinfo("Success", f"Scan saved to {save_path}")
                self.status_var.set("Scan complete.")
            except Exception as e:
                messagebox.showerror("Scan Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerApp(root)
    root.mainloop()
