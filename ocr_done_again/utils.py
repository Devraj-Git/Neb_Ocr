"""
utils.py — Shared utility functions for the NEB OCR application.

Only includes functions actively used by main.py.
Old pipeline utilities have been moved to archive/.
"""

import os
import shutil
from datetime import datetime


def to_roman(grade: str) -> str:
    """Convert numeric grade to Roman numeral. E.g. '11' -> 'XI', '12' -> 'XII'."""
    mapping = {
        "11": "XI",
        "12": "XII",
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
        "5": "V",
        "6": "VI",
        "7": "VII",
        "8": "VIII",
        "9": "IX",
        "10": "X",
    }
    return mapping.get(str(grade).strip(), str(grade))


def to_english(grade: str) -> str:
    """Convert Roman numeral grade to English word. E.g. 'XI' -> 'Eleven', 'XII' -> 'Twelve'."""
    mapping = {
        "11": "Eleven",
        "12": "Twelve",
        "XI": "Eleven",
        "XII": "Twelve",
    }
    return mapping.get(str(grade).strip().upper(), str(grade))


def get_next_filename(folder_path: str) -> str:
    """
    Get the next available filename in a folder by incrementing a counter.
    Scans existing .jpg files named with integers and returns next_int.jpg.
    Falls back to timestamp-based name if no integer files exist.
    """
    os.makedirs(folder_path, exist_ok=True)

    existing = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".jpg"):
            base = os.path.splitext(fname)[0]
            if base.isdigit():
                existing.append(int(base))

    if existing:
        next_num = max(existing) + 1
    else:
        next_num = int(datetime.now().timestamp())

    return os.path.join(folder_path, f"{next_num}.jpg")


def save_image_smart(source_path: str, final_folder: str) -> str:
    """
    Copy an image to the target folder under a sequentially-numbered .jpg filename.
    If the source is not already JPEG, converts it using PIL.
    Returns the full path to the saved file.
    """
    from PIL import Image
    save_path = get_next_filename(final_folder)
    ext = os.path.splitext(source_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        shutil.copy2(source_path, save_path)
    else:
        img = Image.open(source_path)
        img.convert("RGB").save(save_path, "JPEG", quality=95)
    return save_path
