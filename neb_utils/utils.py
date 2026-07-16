"""
utils.py — Shared utility functions for the NEB OCR application.

Only includes functions actively used by main.py.
Old pipeline utilities have been moved to archive/.
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Optional


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


# ─────────────────────────────────────────────
# CHECKPOINT / RESUME HELPERS
# ─────────────────────────────────────────────

CHECKPOINT_DIR = "temp"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "batch_checkpoint.json")


def sanitize_filename_component(text: str, max_len: int = 40) -> str:
    """Clean a string for use as part of a filename."""
    # Keep only alphanumeric, underscore, hyphen
    cleaned = re.sub(r'[^A-Za-z0-9_\-]', '_', str(text))
    # Collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Strip leading/trailing underscores
    cleaned = cleaned.strip("_")
    return cleaned[:max_len]


def build_image_filename(data: dict) -> str:
    """
    Build a meaningful image filename from OCR-extracted data.
    Format: {school_code}_{page_number}_{first_regno}.jpg
    Example: 2738_77_682738012.jpg

    If school_code or page_number are missing / look like defaults,
    falls back to a timestamp-based name to avoid "0000_0_.jpg".
    """
    school_code = str(data.get("school_code", "")).strip()
    page = str(data.get("page_number", "")).strip()
    students = data.get("students", [])

    first_regno = ""
    if students:
        s = students[0]
        first_regno = str(s.get("registration_number", "") if isinstance(s, dict) else s.registration_number).strip()

    # If school_code is missing/placeholder AND no regno, use timestamp fallback
    has_school = bool(school_code) and school_code not in ("0", "0000", "?", "")
    has_page = bool(page) and page not in ("0", "?", "")
    has_regno = bool(first_regno)

    if has_school and has_page and has_regno:
        return f"{school_code}_{page}_{first_regno}.jpg"
    elif has_school and has_page:
        return f"{school_code}_{page}_noregnum.jpg"
    else:
        # Worst case — use timestamp
        ts = int(datetime.now().timestamp())
        parts = [school_code if has_school else "x", page if has_page else "x"]
        if first_regno:
            parts.append(first_regno)
        else:
            parts.append(str(ts))
        return f"{'_'.join(parts)}.jpg"


def load_checkpoint() -> Optional[dict]:
    """Load the batch checkpoint file, or return None if not found/corrupt."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "source_folder" not in data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None


def save_checkpoint(source_folder: str, total: int, file_map: dict,
                    completed: list, remaining: list):
    """Write the batch checkpoint to ./temp/batch_checkpoint.json."""
    import json
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    data = {
        "source_folder": os.path.abspath(source_folder),
        "started_at": datetime.utcnow().isoformat(),
        "total": total,
        "file_map": file_map,
        "completed": completed,
        "remaining": remaining,
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def delete_checkpoint():
    """Remove the checkpoint file after successful batch completion."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except OSError:
            pass


def normalize_path_for_db(file_path: str) -> str:
    """Normalize a file path to forward slashes for consistent DB matching."""
    return file_path.replace("\\", "/")
