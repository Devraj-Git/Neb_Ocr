from wia_scan import prompt_choose_device_and_connect, scan_side
import os

def scan_simple():
    try:
        # 1. This will open a dialog if multiple scanners are found 
        # or connect automatically to the Canon LiDE 110
        device = prompt_choose_device_and_connect()
        
        if not device:
            print("No scanner detected.")
            return

        print("Scanning...")
        # 2. Perform the scan (returns a Pillow Image object)
        # Use DPI 200 or 300 for optimal OCR speed
        pillow_image = scan_side(device=device)
        
        # 3. Save to file
        output_path = "scanned_doc.jpg"
        pillow_image.save(output_path, quality=95)
        print(f"Scan complete: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"Scanning failed: {e}")

if __name__ == "__main__":
    scan_simple()
