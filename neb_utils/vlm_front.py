"""
vlm_front.py — Bridge between main.py UI and the VLM-based OCR pipeline.

Replaces the old ocr_front.py which used the complex OpenCV + TrOCR pipeline.
Now uses Ollama Qwen3-VL for end-to-end structured extraction.
"""

import json
import os
from typing import Optional

from .ollama_pipeline import process_image, save_to_database


def get_vlm_result(img_path: str,
                   model: str = "qwen3-vl:8b-instruct") -> tuple:
    """
    Run VLM OCR on the given image and return structured data + display summary.

    Args:
        img_path: Path to the image file.
        model: Ollama model name.

    Returns:
        (data_dict, summary_text) where:
            data_dict: The full NEBGradingSheet structure (dict).
            summary_text: Formatted string for display in the console log.
    """
    result = process_image(img_path, model=model)

    # Build display summary
    lines = []
    lines.append("=" * 60)
    school_info = f"SCHOOL: {result.get('school_name', '?')}  |  CODE: {result.get('school_code', '?')}"
    lines.append(school_info)
    lines.append(f"GRADE: {result.get('grade', '?')}  |  TYPE: {result.get('exame_Type', '?')}  |  YEAR: {result.get('examination_year', '?')}")
    lines.append(f"PAGE: {result.get('page_number', '?')}")
    lines.append("=" * 60)
    lines.append(f"TOTAL STUDENTS: {len(result.get('students', []))}")
    lines.append("=" * 60)

    for idx, student in enumerate(result.get("students", []), 1):
        symbol = student.get("symbol_number", "N/A")
        name = student.get("student_name", "N/A")
        total = student.get("grand_total", "N/A")
        rem = student.get("remark", "N/A")
        reg = student.get("registration_number", "N/A")
        dob = student.get("date_of_birth", "N/A")

        record = f"\n[{idx}] SYMBOL: {symbol} | REG: {reg}\n"
        record += f"     NAME: {name}\n"
        record += f"     DOB: {dob} | TOTAL: {total} | REMARK: {rem}\n"

        for subj in student.get("subjects", []):
            subj_name = subj.get("subject_name", "?")
            th = subj.get("theory") or "-"
            pr = subj.get("practical") or "-"
            tot = subj.get("total") or "-"
            extra = " [extra]" if subj.get("extra") else ""
            record += f"     > {subj_name}: TH({th}) PR({pr}) TOT({tot}){extra}\n"

        lines.append(record)

    lines.append("\n" + "=" * 60)
    lines.append("OCR EXTRACTION COMPLETE — Ready for Database Commit")
    lines.append("=" * 60)

    return result, "\n".join(lines)


def save_vlm_to_database(data: dict, image_path: str, meta: Optional[dict] = None):
    """
    Save VLM OCR results to the normalized 6-table database schema.

    Args:
        data: The NEBGradingSheet dict from get_vlm_result().
        image_path: The original image file path.
        meta: Dict with grade, exam_type, book_name, etc.
    """
    save_to_database(data, image_path, meta=meta)
