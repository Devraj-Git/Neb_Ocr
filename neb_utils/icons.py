"""
PIL-generated button icons for the NEB OCR application.

Each `_draw_*` function draws a small icon into a PIL `ImageDraw` surface.
The module-level `*_ICON` constants are ready-to-use `CTkImage` instances.
"""

from PIL import Image, ImageDraw
import customtkinter as ctk


def _make_icon(draw_func, size=20, color="white"):
    """Create a small CTkImage icon from a PIL drawing function."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_func(draw, size, color)
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))


def _draw_scan_icon(draw, size, color):
    """Draw a small OCR/scan icon (document with lines)."""
    c = color
    m = 3  # margin
    w, h = size - 2 * m, size - 2 * m
    # Document body
    draw.rectangle([m, m, m + w, m + h], outline=c, width=1)
    # Content lines
    for y in range(m + 4, m + h - 2, 4):
        draw.line([m + 2, y, m + w - 2, y], fill=c, width=1)


def _draw_save_icon(draw, size, color):
    """Draw a small database/save icon (cylinder)."""
    c = color
    m = 3
    w, h = size - 2 * m, size - 2 * m
    # Cylinder top ellipse
    draw.ellipse([m, m, m + w, m + h // 3], outline=c, width=1)
    # Cylinder body
    draw.line([m, m + h // 3, m, m + h - h // 3], fill=c, width=1)
    draw.line([m + w, m + h // 3, m + w, m + h - h // 3], fill=c, width=1)
    # Cylinder bottom curve
    draw.ellipse([m, m + h - h // 3, m + w, m + h], outline=c, width=1)


def _draw_folder_icon(draw, size, color):
    """Draw a small folder icon."""
    c = color
    m = 3
    w, h = size - 2 * m, size - 2 * m
    # Folder tab
    draw.polygon(
        [m + 2, m, m + w // 3, m, m + w // 3 + 2, m + 3, m + w - 2, m + 3],
        outline=c,
        width=1,
    )
    # Folder body
    draw.rectangle([m, m + 3, m + w - 2, m + h - 2], outline=c, width=1)


def _draw_image_icon(draw, size, color):
    """Draw a small image/picture icon (landscape with mountain)."""
    c = color
    m = 3
    w, h = size - 2 * m, size - 2 * m
    # Frame (landscape rectangle)
    draw.rectangle([m, m + 2, m + w, m + h - 2], outline=c, width=1)
    # Sun (small circle top-right)
    cx_sun = m + int(w * 0.75)
    cy_sun = m + 6
    draw.ellipse([cx_sun - 2, cy_sun - 2, cx_sun + 2, cy_sun + 2], outline=c, width=1)
    # Mountain (triangle)
    draw.polygon(
        [
            m + 2, m + h - 2,
            m + w // 2 - 2, m + 6,
            m + w - 2, m + h - 2,
        ],
        outline=c,
        width=1,
    )
    # Base line (ground)
    draw.line([m, m + h - 2, m + w, m + h - 2], fill=c, width=1)


def _draw_globe_icon(draw, size, color):
    """Draw a small globe icon (circle with latitude/longitude lines)."""
    c = color
    m = 3
    w, h = size - 2 * m, size - 2 * m
    cx = size // 2
    # Outer circle
    draw.ellipse([m, m, m + w, m + h], outline=c, width=1)
    # Vertical longitude line
    draw.line([cx, m, cx, m + h], fill=c, width=1)
    # Horizontal latitude lines
    lat_y1 = m + h // 3
    lat_y2 = m + 2 * h // 3
    draw.line([m, lat_y1, m + w, lat_y1], fill=c, width=1)
    draw.line([m, lat_y2, m + w, lat_y2], fill=c, width=1)
    # Diagonal curve (continent suggestion)
    draw.arc(
        [m + w // 4, m + h // 4, m + 3 * w // 4, m + 3 * h // 4],
        start=-45,
        end=90,
        fill=c,
        width=1,
    )


# ── Ready-to-use icon constants ──────────────────────────────

OCR_ICON = _make_icon(_draw_scan_icon, 20, "white")
DB_ICON = _make_icon(_draw_save_icon, 20, "white")
IMAGE_ICON = _make_icon(_draw_image_icon, 18, "#94A3B8")
FOLDER_ICON = _make_icon(_draw_folder_icon, 18, "#94A3B8")
GLOBE_ICON = _make_icon(_draw_globe_icon, 16, "#94A3B8")
