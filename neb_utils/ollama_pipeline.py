"""
ollama_pipeline.py — VLM-based OCR Pipeline
─────────────────────────────────────────────
Image → Ollama (Qwen3-VL) → Structured JSON → Normalized Database (6 tables)

This replaces the old preprocessing-heavy pipeline (row/col detection + TrOCR).
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Optional, List

import mysql.connector
import ollama
from PIL import Image

from .schemas import NEBGradingSheet


# ─────────────────────────────────────────────
# 1. IMAGE PREPROCESSING
# ─────────────────────────────────────────────

def prepare_image(image_path: str, max_dim: int = 2000) -> str:
    """
    Resize the image for VLM processing while maintaining aspect ratio.
    Returns the path to the prepared image.
    
    For OCR, higher resolution is better (within VLM context limits).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    
    # Only resize if larger than max_dim
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    
    # Save as PNG (lossless) to a temp file
    base, ext = os.path.splitext(image_path)
    prepared_path = f"{base}_vlm_prepared.png"
    img.save(prepared_path, "PNG")
    return prepared_path


# ─────────────────────────────────────────────
# 2. CORE OLLAMA FUNCTION
#    Sends image → Qwen3-VL → parsed JSON
# ─────────────────────────────────────────────

OCR_PROMPT = """
Perform accurate OCR extraction from this academic grading ledger sheet.

Instructions:

1. Detect the complete table structure and align every row and column according to the table headers. Headers may include:
   - SYMBOL
   - REG. NO.
   - NAME OF THE STUDENT
   - DOB
   - SUBJ
   - TH
   - PR
   - TOT
   - TOTAL
   - REMARKS
   - and any other visible columns.

2. Extract every student record exactly as printed. Preserve the original table alignment and reading order from top to bottom.

3. Some students span multiple rows. If a row does NOT contain the registration number or student name but contains the DOB and additional subject information, treat it as a continuation of the previous student. Associate those subjects with the same student.

4. For each subject row:
   - Read the subject code or name from the SUBJ column.
   - Extract TH, PR, and TOT values only from their corresponding columns.
   - Never shift values into neighboring columns.
   - Preserve every character exactly as printed, including:
       * asterisks (*)
       * handwritten corrections
       * overwritten values
       * special symbols

5. The first subject row for a student often has blank PR and TOT columns. If those cells are genuinely blank, return null for those fields. Do not infer or calculate missing values.

6. If any table cell is blank, return null.

7. Do not guess, correct, or normalize any extracted values. Return the OCR text exactly as it appears.

8. Follow the provided JSON schema exactly. Every extracted value must be mapped to the correct field according to the detected table layout.

9. IMPORTANT: Extract the GRADE (Eleven or Twelve) and EXAM_TYPE (Regular, Partial, or Supplementary) from the document header.
"""


def process_image(image_path: str, model: str = "qwen3-vl:8b-instruct",
                  max_retries: int = 2) -> dict:
    """
    Send an image to Ollama Qwen3-VL and get back structured JSON.

    Args:
        image_path: Absolute or relative path to the image file.
        model: Ollama model name (default: qwen3-vl:8b-instruct)
        max_retries: Number of retries if JSON parsing fails.

    Returns:
        A dictionary parsed from the LLM JSON output (validated against NEBGradingSheet schema).

    Raises:
        FileNotFoundError: if image_path doesn't exist.
        RuntimeError: if Ollama call fails or response is unparseable after retries.
    """
    # Prepare image
    prepared_path = prepare_image(image_path)
    
    print(f"[Ollama] Sending {os.path.basename(prepared_path)} to {model} ...")

    last_error = None
    for attempt in range(1 + max_retries):
        if attempt > 0:
            print(f"[Ollama] Retry {attempt}/{max_retries} ...")
            time.sleep(1)

        try:
            t0 = time.time()
            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": OCR_PROMPT,
                    "images": [prepared_path]
                }],
                format=NEBGradingSheet.model_json_schema(),
                options={
                    "num_ctx": 10288,
                    # "num_predict": 8192,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 0.1,
                    "seed": 42,
                    "repeat_penalty": 1.05,
                }
            )
            elapsed = time.time() - t0
            print(f"[Ollama] Response received in {elapsed:.2f}s")

            raw = response["message"]["content"]

            # --- Attempt to extract JSON (LLM sometimes wraps in ```json blocks) ---
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1)

            parsed = json.loads(raw)

            # --- Validate against the Pydantic schema ---
            try:
                NEBGradingSheet.model_validate(parsed)
                print(f"[Ollama] ✅ Validated against NEBGradingSheet schema")
            except Exception as ve:
                print(f"[Ollama] ⚠️ Schema validation warning (non-fatal): {ve}")
                # Keep going — the data might still be usable

            # Clean up temp file
            try:
                if prepared_path != image_path:
                    os.remove(prepared_path)
            except OSError:
                pass

            return parsed

        except json.JSONDecodeError as e:
            last_error = f"Failed to parse LLM response as JSON: {e}"
            print(f"[Ollama] ❌ {last_error}")
            print(f"[Ollama] Raw response (first 300 chars): {raw[:300]}")
            continue
        except Exception as e:
            last_error = str(e)
            print(f"[Ollama] ❌ Error: {e}")
            continue

    # Clean up temp file on final failure
    try:
        if prepared_path != image_path:
            os.remove(prepared_path)
    except OSError:
        pass

    raise RuntimeError(f"Ollama processing failed after {max_retries + 1} attempts. Last error: {last_error}")


# ─────────────────────────────────────────────
# 3. DATABASE CONNECTION
# ─────────────────────────────────────────────

def _get_db():
    """Create and return a fresh DB connection using env vars."""
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


# ─────────────────────────────────────────────
# 4. DATABASE SAVE — Normalized 6-table schema
#    Tables: students_school, students_exam, students_subject,
#            students_student, students_result, students_mark
# ─────────────────────────────────────────────

def _upsert_school(conn, code: str, name: str) -> int:
    """
    Find school by code (unique), or create it. Return its ID.
    Table: students_school (columns: code[unique], name)
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
    Find exam by unique (grade, exam_type, exam_year), or create it. Return its ID.
    Table: students_exam (unique on grade, exam_type, exam_year)
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
            "INSERT INTO students_exam (grade, exam_type, exam_year, created_on, updated_on) "
            "VALUES (%s, %s, %s, %s, %s)",
            (grade, exam_type, exam_year, _now(), _now())
        )
        conn.commit()
        return cursor.lastrowid


def _upsert_subject(conn, code: str) -> int:
    """
    Find subject by code (unique), or create it. Return its ID.
    Table: students_subject (columns: code[unique])
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


def _upsert_student(conn, symbol_no: str, reg_no: str, name: str, dob: str,
                    sex: Optional[str] = None) -> int:
    """
    Find student by symbol_no+reg_no approximate match, or create. Return ID.
    Table: students_student (columns: symbol_no, reg_no, name, sex, dob)
    
    Note: symbol_no is NOT unique, so we match on symbol_no + reg_no combo.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id FROM students_student WHERE symbol_no = %s AND reg_no = %s",
        (symbol_no, reg_no)
    )
    row = cursor.fetchone()
    if row:
        # Update details in case they changed
        cursor.execute(
            "UPDATE students_student SET name = %s, dob = %s, sex = %s, updated_on = %s WHERE id = %s",
            (name, dob, sex, _now(), row["id"])
        )
        conn.commit()
        return row["id"]
    else:
        cursor.execute(
            "INSERT INTO students_student (symbol_no, reg_no, name, sex, dob, created_on, updated_on) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (symbol_no, reg_no, name, sex, dob, _now(), _now())
        )
        conn.commit()
        return cursor.lastrowid


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Safely convert a string to int, returning None if it fails."""
    if value is None:
        return None
    # Strip leading zeros (but keep "0")
    cleaned = value.lstrip("0") or "0"
    # Remove non-digit characters like '*'
    cleaned = re.sub(r'[^0-9]', '', cleaned)
    if not cleaned:
        return None
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
        notes_str = " | ".join(marginal_notes) if isinstance(marginal_notes, list) else str(marginal_notes)
        parts.append(f"[Marginal notes: {notes_str}]")
    return " | ".join(parts) if parts else None


def save_to_database(data: dict, image_path: str, meta: Optional[dict] = None):
    """
    Save the structured OCR result to the normalized database.

    Tables involved:
      1. students_school   — upsert by code
      2. students_exam     — upsert by (grade, exam_type, exam_year)
      3. students_subject  — upsert by code (subject_name)
      4. students_student  — upsert by symbol_no + reg_no
      5. students_result   — insert new (links exam, school, student)
      6. students_mark     — insert new per subject (links result, subject)

    Args:
        data: Parsed JSON dict from process_image()
        image_path: Original image path
        meta: Dict with:
              - grade (str, optional): "11" or "12" — overrides grade from image
              - exam_type (str, optional): "Regular", "Partial", "Supplementary"
              - exam_year (str, optional): overrides year from image
              - book_name (str, optional)
              - qc_check, qc_remarks (str, optional)
              - cluster_id, ui (str, optional)
              - remarks (str, optional)
              - is_legacy_image (bool, optional)
    """
    # Validate
    try:
        validated = NEBGradingSheet.model_validate(data)
    except Exception as ve:
        print(f"❌ Schema validation failed: {ve}")
        print("   Attempting to save anyway with raw data...")
        validated = None

    meta = meta or {}

    # Use provided meta values, falling back to VLM-extracted values
    if validated:
        school_code = validated.school_code
        school_name = validated.school_name
        exam_year_from_img = validated.examination_year
        grade_from_img = validated.grade
        exam_type_from_img = validated.exame_Type
        students_list = validated.students
        sheet_marginal = " | ".join(validated.handwritten_marginal_notes) if validated.handwritten_marginal_notes else None
    else:
        # Fallback to raw data
        school_code = str(data.get("school_code", ""))
        school_name = data.get("school_name", "")
        exam_year_from_img = str(data.get("examination_year", ""))
        grade_from_img = data.get("grade", "")
        exam_type_from_img = data.get("exame_Type", "")
        raw_students = data.get("students", [])
        # Convert raw dicts to StudentRecord-like objects
        from pydantic import BaseModel
        class StudentLike:
            def __init__(self, d):
                self.symbol_number = d.get("symbol_number", "")
                self.registration_number = d.get("registration_number", "")
                self.student_name = d.get("student_name", "")
                self.date_of_birth = d.get("date_of_birth", "")
                self.grand_total = d.get("grand_total", "0")
                self.remark = d.get("remark", "PASS")
                self.subjects = d.get("subjects", [])
        students_list = [StudentLike(s) for s in raw_students]
        sheet_marginal = None

    # Override from meta if provided
    grade = meta.get("grade", grade_from_img)
    exam_type = meta.get("exam_type", exam_type_from_img)
    exam_year = meta.get("exam_year", exam_year_from_img)
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
    print(f"📄 School: {school_name} ({school_code})")
    print(f"📅 Year: {exam_year} | Grade: {grade} | Type: {exam_type}")
    print(f"👥 Students: {len(students_list)}")
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
        school_id = _upsert_school(conn, str(school_code), school_name)
        print(f"🏫 School ID: {school_id} ({school_code})")

        # 2. Exam
        exam_id = _upsert_exam(conn, str(exam_year), str(grade), str(exam_type))
        print(f"📅 Exam ID: {exam_id} ({exam_year} / {grade} / {exam_type})")

        # Cache subject IDs to avoid repeated DB hits for the same subject
        subject_cache: dict[str, int] = {}

        for student_idx, student in enumerate(students_list, 1):
            # 3. Student
            student_id = _upsert_student(
                conn,
                symbol_no=str(student.symbol_number),
                reg_no=str(student.registration_number),
                name=str(student.student_name),
                dob=str(student.date_of_birth),
            )

            # 4. Result
            result_id = _insert_result(
                conn,
                total=str(student.grand_total) if hasattr(student, 'grand_total') else "0",
                result=str(student.remark) if hasattr(student, 'remark') else "PASS",
                remarks=remarks,
                marginal_notes=sheet_marginal,
                book_name=book_name,
                image_name=image_name,
                file_path=image_path,
                qc_check=str(qc_check) if qc_check else None,
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
                subj_name = subj.get("subject_name") if isinstance(subj, dict) else subj.subject_name
                subj_theory = subj.get("theory") if isinstance(subj, dict) else subj.theory
                subj_practical = subj.get("practical") if isinstance(subj, dict) else subj.practical
                subj_total = subj.get("total") if isinstance(subj, dict) else subj.total

                if subj_name not in subject_cache:
                    subject_cache[subj_name] = _upsert_subject(conn, subj_name)
                subject_id = subject_cache[subj_name]

                _insert_mark(
                    conn,
                    theory=str(subj_theory) if subj_theory is not None else None,
                    practical=str(subj_practical) if subj_practical is not None else None,
                    total=str(subj_total) if subj_total is not None else None,
                    result_id=result_id,
                    subject_id=subject_id
                )

            print(f"  ✅ {student_idx}. {student.student_name}"
                  f"  | Symbol: {student.symbol_number}"
                  f"  | Total: {student.grand_total}"
                  f"  | Result: {student.remark}"
                  f"  | {len(student.subjects)} subjects")

        print(f"\n🎉 Successfully saved {len(students_list)} student records to database!")

    except Exception as e:
        print(f"❌ Error during save: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        raise
    finally:
        conn.close()


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


# ─────────────────────────────────────────────
# 5. CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NEB OCR via Ollama Qwen3-VL")
    parser.add_argument("image", help="Path to the mark sheet image")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct", help="Ollama model name")
    parser.add_argument("--save", action="store_true", help="Save results to database")
    parser.add_argument("--output", "-o", help="Save JSON output to a file")
    parser.add_argument("--grade", default=None, help="Override grade (11 or 12)")
    parser.add_argument("--exam-type", default=None,
                        choices=["Regular", "Partial", "Supplementary"],
                        help="Override examination type")
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
            # Print summary
            print(f"\n📋 School: {result.get('school_name', '?')} ({result.get('school_code', '?')})")
            print(f"   Grade: {result.get('grade', '?')} | Type: {result.get('exame_Type', '?')} | Year: {result.get('examination_year', '?')}")
            print(f"   Students: {len(result.get('students', []))}")
            print()

        if args.save:
            meta = {}
            if args.grade:
                meta["grade"] = args.grade
            if args.exam_type:
                meta["exam_type"] = args.exam_type
            if args.book_name:
                meta["book_name"] = args.book_name
            save_to_database(result, args.image, meta=meta)
        else:
            print("💡 Use --save to commit to database.")
            print("   Example: python ollama_pipeline.py image.jpg --save --grade 12 --exam-type Regular")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
