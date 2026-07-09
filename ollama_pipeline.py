"""
ollama_pipeline.py — NEW FLOW
─────────────────────────────
Image → Ollama (Qwen3-VL) → Structured JSON → Database

This replaces the old preprocessing-heavy pipeline (row/col detection + TrOCR).
"""

import os
import json
import re
from datetime import datetime
from typing import Optional

import mysql.connector
import ollama

# Shared schemas — update ocr_done_again/schemas.py when the format changes
from ocr_done_again.schemas import NEBGradingSheet


# ─────────────────────────────────────────────
# 2. CORE OLLAMA FUNCTION
#    Sends image → Qwen3-VL → parsed JSON
# ─────────────────────────────────────────────

def process_image(image_path: str, model: str = "qwen3-vl:8b-instruct") -> dict:
    """
    Send an image to Ollama Qwen3-VL and get back structured JSON.
    
    Args:
        image_path: Absolute or relative path to the image file.
        model: Ollama model name (default: qwen3-vl:8b-instruct)
    
    Returns:
        A dictionary parsed from the LLM JSON output (matches NEBGradingSheet schema).
    
    Raises:
        FileNotFoundError: if image_path doesn't exist.
        RuntimeError: if Ollama call fails or response is unparseable.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[Ollama] Sending {os.path.basename(image_path)} to {model} ...")

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Perform precise OCR data extraction on this academic grading ledger sheet. "
                "Carefully inspect each row. Note that some students have MATHS and other subjects "
                "in the second row as well — if the row does not contain the registration number or "
                "student name and only contains the DOB, also check for other subjects in that row "
                "aligned with any subject header. Capture all marks exactly as printed, preserving "
                "any asterisks (*) or handwritten corrections. Also, for the first subject of each "
                "student row, the PR and TOT columns may be empty — confirm that and make them null."
            ),
            "images": [image_path]
        }],
        format=NEBGradingSheet.model_json_schema(),
        options={
            "num_ctx": 16384,
            "num_predict": 8192,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0.1,
            "seed": 42,
            "repeat_penalty": 1.05,
        }
    )

    raw = response["message"]["content"]
    
    # --- Try to parse JSON from the response ---
    # The LLM sometimes wraps JSON in ```json ... ``` blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse LLM response as JSON:\n{str(e)}\n\nRaw response:\n{raw[:500]}"
        )

    return parsed


def validate_response(data: dict) -> NEBGradingSheet:
    """
    Validate the parsed JSON against the NEBGradingSheet schema.
    Raises a ValidationError if something is wrong.
    """
    return NEBGradingSheet.model_validate(data)


# ─────────────────────────────────────────────
# 3. DATABASE SAVE — Normalized 6-table schema
#    Tables: students_school, students_exam, students_subject,
#            students_student, students_result, students_mark
# ─────────────────────────────────────────────

def _get_db():
    """Create and return a fresh NEBDB-like connection using env vars."""
    conn = mysql.connector.connect(
        host=os.getenv("HOST"),
        port=int(os.getenv("PORT", 3306)),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        database=os.getenv("DATABASE"),
    )
    return conn


def _now():
    """Current UTC datetime for created_on / updated_on."""
    return datetime.utcnow()


def _upsert_school(conn, code: str, name: str) -> int:
    """
    Find school by code, or create it. Return its ID.
    Table: students_school (columns: code, name)
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM students_school WHERE code = %s", (code,))
    row = cursor.fetchone()
    if row:
        # Update name in case it changed
        cursor.execute(
            "UPDATE students_school SET name = %s, updated_on = %s WHERE id = %s",
            (name, _now(), row["id"])
        )
        conn.commit()
        return row["id"]
    else:
        cursor.execute(
            "INSERT INTO students_school (code, name, created_on, updated_on) VALUES (%s, %s, %s, %s)",
            (code, name, _now(), _now())
        )
        conn.commit()
        return cursor.lastrowid


def _upsert_exam(conn, exam_year: str, grade: str, exam_type: str) -> int:
    """
    Find exam by (grade, exam_type, exam_year), or create it. Return its ID.
    Table: students_exam (columns: grade, exam_type, exam_year)
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id FROM students_exam WHERE grade = %s AND exam_type = %s AND exam_year = %s",
        (grade, exam_type, exam_year)
    )
    row = cursor.fetchone()
    if row:
        return row["id"]
    else:
        cursor.execute(
            "INSERT INTO students_exam (grade, exam_type, exam_year, created_on, updated_on) VALUES (%s, %s, %s, %s, %s)",
            (grade, exam_type, exam_year, _now(), _now())
        )
        conn.commit()
        return cursor.lastrowid


def _upsert_subject(conn, code: str) -> int:
    """
    Find subject by code, or create it. Return its ID.
    Table: students_subject (columns: code)
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM students_subject WHERE code = %s", (code,))
    row = cursor.fetchone()
    if row:
        return row["id"]
    else:
        cursor.execute(
            "INSERT INTO students_subject (code, created_on, updated_on) VALUES (%s, %s, %s)",
            (code, _now(), _now())
        )
        conn.commit()
        return cursor.lastrowid


def _upsert_student(conn, symbol_no: str, reg_no: str, name: str, dob: str) -> int:
    """
    Find student by symbol_no, or create. Return ID.
    Table: students_student (columns: symbol_no, reg_no, name, sex, dob)
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM students_student WHERE symbol_no = %s", (symbol_no,))
    row = cursor.fetchone()
    if row:
        # Update details (name, reg, dob could change between scans)
        cursor.execute(
            "UPDATE students_student SET reg_no = %s, name = %s, dob = %s, updated_on = %s WHERE id = %s",
            (reg_no, name, dob, _now(), row["id"])
        )
        conn.commit()
        return row["id"]
    else:
        cursor.execute(
            "INSERT INTO students_student (symbol_no, reg_no, name, dob, created_on, updated_on) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (symbol_no, reg_no, name, dob, _now(), _now())
        )
        conn.commit()
        return cursor.lastrowid


def _insert_result(conn, *, total: str, result: str, remarks: Optional[str],
                   marginal_notes: Optional[str],
                   book_name: Optional[str], image_name: Optional[str],
                   file_path: Optional[str], qc_check: Optional[str],
                   qc_remarks: Optional[str], cluster_id: Optional[str],
                   ui: Optional[str],
                   exam_id: int, school_id: int, student_id: int,
                   is_legacy_image: bool = False) -> int:
    """
    Insert a new row into students_result. Return the new ID.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """INSERT INTO students_result
           (created_on, updated_on, total, result, remarks,
            book_name, image_name, file_path, qc_check, qc_remarks,
            cluster_id, ui, exam_id, school_id, student_id, is_legacy_image)
           VALUES (%s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s)""",
        (_now(), _now(), _safe_int(total), result,
         _build_remarks(remarks, marginal_notes),
         book_name, image_name, file_path, qc_check, qc_remarks,
         cluster_id, ui, exam_id, school_id, student_id, is_legacy_image)
    )
    conn.commit()
    return cursor.lastrowid


def _insert_mark(conn, *, theory: Optional[str], practical: Optional[str],
                 total: Optional[str], result_id: int, subject_id: int):
    """
    Insert a new row into students_mark.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """INSERT INTO students_mark
           (created_on, updated_on, theory, practical, total, result_id, subject_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (_now(), _now(), theory, practical, total, result_id, subject_id)
    )
    conn.commit()


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Safely convert a string to int, returning None if it fails."""
    if value is None:
        return None
    # Strip leading zeros (but keep "0")
    cleaned = value.lstrip("0") or "0"
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def _build_remarks(remarks: Optional[str], marginal_notes: Optional[str]) -> Optional[str]:
    """Combine general remarks and marginal notes into one string."""
    parts = []
    if remarks:
        parts.append(remarks)
    if marginal_notes:
        parts.append(f"[Marginal notes: {marginal_notes}]")
    return " | ".join(parts) if parts else None


def save_to_database(data: dict, image_path: str, meta: Optional[dict] = None):
    """
    Save the structured OCR result to the normalized database.

    Tables involved:
      1. students_school   — upsert by code
      2. students_exam     — upsert by (grade, exam_type, exam_year)
      3. students_subject  — upsert by code (subject_name)
      4. students_student  — upsert by symbol_no
      5. students_result   — insert new (links exam, school, student)
      6. students_mark     — insert new per subject (links result, subject)

    Args:
        data: Parsed JSON dict from process_image()
        image_path: Original image path
        meta: Dict with at least:
              - grade (str): "11" or "12"
              - exam_type (str): "Regular", "Partial", "Supplementary"
              Optional:
              - exam_year: overrides the year from JSON
              - book_name
              - qc_check, qc_remarks
              - cluster_id, ui
              - remarks
              - is_legacy_image (bool)
    """
    validated = validate_response(data)
    meta = meta or {}

    grade = meta.get("grade", "")
    exam_type = meta.get("exam_type", "")
    exam_year = meta.get("exam_year", validated.examination_year)
    book_name = meta.get("book_name")
    qc_check = meta.get("qc_check", "0")
    qc_remarks = meta.get("qc_remarks")
    cluster_id = meta.get("cluster_id")
    ui = meta.get("ui")
    remarks = meta.get("remarks")
    is_legacy_image = meta.get("is_legacy_image", False)

    # Derive image_name from the file path
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*60}")
    print(f"📄 School: {validated.school_name} ({validated.school_code})")
    print(f"📅 Year: {exam_year} | Grade: {grade} | Type: {exam_type}")
    print(f"👥 Students: {len(validated.students)}")
    print(f"{'='*60}\n")

    # Connect to DB
    try:
        conn = _get_db()
        print("✅ Connected to database.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    try:
        # 1. School
        school_id = _upsert_school(conn, validated.school_code, validated.school_name)
        print(f"🏫 School ID: {school_id} ({validated.school_code})")

        # 2. Exam
        exam_id = _upsert_exam(conn, exam_year, grade, exam_type)
        print(f"📅 Exam ID: {exam_id} ({exam_year} / {grade} / {exam_type})")

        # Cache subject IDs to avoid repeated DB hits for the same subject
        subject_cache: dict[str, int] = {}

        for student_idx, student in enumerate(validated.students, 1):
            # 3. Student
            student_id = _upsert_student(
                conn,
                symbol_no=student.symbol_number,
                reg_no=student.registration_number,
                name=student.student_name,
                dob=student.date_of_birth
            )

            # Join handwritten_marginal_notes from the sheet-level
            sheet_marginal = " | ".join(validated.handwritten_marginal_notes) if validated.handwritten_marginal_notes else None

            # 4. Result
            result_id = _insert_result(
                conn,
                total=student.grand_total,
                result=student.remark,
                remarks=remarks,
                marginal_notes=sheet_marginal,
                book_name=book_name,
                image_name=image_name,
                file_path=image_path,
                qc_check=qc_check,
                qc_remarks=qc_remarks,
                cluster_id=cluster_id,
                ui=ui,
                exam_id=exam_id,
                school_id=school_id,
                student_id=student_id,
                is_legacy_image=is_legacy_image
            )

            # 5. Marks (one row per subject)
            for subj in student.subjects:
                if subj.subject_name not in subject_cache:
                    subject_cache[subj.subject_name] = _upsert_subject(conn, subj.subject_name)
                subject_id = subject_cache[subj.subject_name]

                _insert_mark(
                    conn,
                    theory=subj.theory,
                    practical=subj.practical,
                    total=subj.total,
                    result_id=result_id,
                    subject_id=subject_id
                )

            print(f"  ✅ {student_idx}. {student.student_name}"
                  f"  | Symbol: {student.symbol_number}"
                  f"  | Total: {student.grand_total}"
                  f"  | Result: {student.remark}"
                  f"  | {len(student.subjects)} subjects")

        print(f"\n🎉 Successfully saved {len(validated.students)} student records to database!")

    except Exception as e:
        print(f"❌ Error during save: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# 4. CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NEB OCR via Ollama Qwen3-VL")
    parser.add_argument("image", help="Path to the mark sheet image")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct", help="Ollama model name")
    parser.add_argument("--save", action="store_true", help="Save results to database")
    parser.add_argument("--output", "-o", help="Save JSON output to a file")
    parser.add_argument("--grade", default="12", help="Student grade (11 or 12)")
    parser.add_argument("--exam-type", default="Regular",
                        choices=["Regular", "Partial", "Supplementary"],
                        help="Type of examination")
    parser.add_argument("--book-name", help="Book name for the result record")
    
    args = parser.parse_args()

    print(f"🚀 NEB Ollama OCR Pipeline")
    print(f"   Image: {args.image}")
    print(f"   Model: {args.model}\n")

    try:
        result = process_image(args.image, model=args.model)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if args.save:
            meta = {
                "grade": args.grade,
                "exam_type": args.exam_type,
                "book_name": args.book_name,
            }
            save_to_database(result, args.image, meta=meta)
        else:
            print("\n💡 Use --save to commit to database.")
            print("   Example: python ollama_pipeline.py image.jpg --save --grade 12 --exam-type Regular")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
