"""
Batch folder OCR processing – extracted from main.py for a lighter main module.

``BatchProcessor`` owns the folder-scanning, checkpoint-resume, per-image
processing pipeline, and UI-rollback logic.  It receives the parent
``RadiantUltraScanner`` instance so it can read sidebar values, update
widgets, and push log messages.
"""

import os
import threading
import warnings
from tkinter import filedialog, messagebox

from neb_utils.icons import FOLDER_ICON
from neb_utils.utils import (
    build_image_filename,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_image_smart,
    to_roman,
)
from neb_utils.vlm_front import get_vlm_result, save_vlm_to_database
from neb_utils.ollama_pipeline import _get_db, get_exam_id, is_student_already_in_db


warnings.filterwarnings("ignore", category=SyntaxWarning)


class BatchProcessor:
    """Handles folder-based batch OCR with checkpointing and resume."""

    def __init__(self, app):
        self.app = app

    # ── Public entry point (wired to the FOLDER button) ──────────

    def select_folder(self):
        """Open folder dialog and start batch OCR processing with resume support."""
        folder_path = filedialog.askdirectory(title="Select Folder with Mark Sheet Images")
        if not folder_path:
            return

        folder_abs = os.path.abspath(folder_path)

        # Recursively scan all subfolders for images
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = []
        for root, _dirs, files in os.walk(folder_abs):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    rel_path = os.path.relpath(os.path.join(root, fname), folder_abs)
                    all_files.append(rel_path)
        all_files.sort()

        if not all_files:
            messagebox.showwarning(
                "Empty Folder",
                "No image files found in the selected folder or its subfolders.",
            )
            return

        # --- Check for existing checkpoint (resume support) ---
        checkpoint = load_checkpoint()
        resume_remaining = None
        completed_originals = set()
        file_map = {}

        if checkpoint and checkpoint.get("source_folder") == folder_abs:
            resume_remaining = checkpoint.get("remaining", [])
            completed_originals = set(checkpoint.get("completed", []))
            file_map = checkpoint.get("file_map", {})

            done_count = len(completed_originals)
            total = checkpoint.get("total", len(all_files))
            if resume_remaining:
                answer = messagebox.askyesno(
                    "Resume Batch",
                    f"Found existing checkpoint for this folder.\n"
                    f"{done_count} of {total} images already completed.\n\n"
                    f"Resume processing the remaining {len(resume_remaining)}?",
                    icon="question",
                )
                if not answer:
                    delete_checkpoint()
                    resume_remaining = None
                    completed_originals = set()
                    file_map = {}

        # Build the work queue
        if resume_remaining is not None:
            work_queue = list(resume_remaining)
            skip_count = len(completed_originals)
        else:
            work_queue = list(all_files)
            completed_originals = set()
            file_map = {}
            skip_count = 0

        if not work_queue:
            messagebox.showinfo(
                "All Done",
                "All images in this folder have already been processed!",
            )
            return

        self.app.batch_files = work_queue
        self.app.batch_total = len(all_files)
        self.app.batch_index = 0
        self.app.batch_success = skip_count
        self.app.batch_fail = 0
        self.app.batch_abort = False
        self.app.is_batch_mode = True
        self.app.batch_folder = folder_abs
        self.app.batch_file_map = file_map
        self.app.batch_completed_originals = completed_originals

        self.app.log_message(f"\n{'=' * 60}")
        self.app.log_message(f"📂 BATCH MODE: '{os.path.basename(folder_abs)}'")
        self.app.log_message(
            f"   Total in folder: {self.app.batch_total}"
            f"  |  Already completed: {skip_count}"
            f"  |  Queue: {len(work_queue)}"
        )
        self.app.log_message(f"{'=' * 60}")

        # Disable UI during batch
        self.app.import_btn.configure(state="disabled")
        self.app.import_folder_btn.configure(state="disabled", text="PROCESSING...", image=None)
        self.app.proceed_btn.configure(state="disabled", text="BATCH ACTIVE")
        self.app.scan_btn.configure(state="disabled")

        threading.Thread(target=self.process_folder_thread, daemon=True).start()

    # ── Shared per-image pipeline ───────────────────────────────

    def process_and_save_image(self, img_path, source_filename="", log_prefix=""):
        """
        Copy image → run OCR → rename → save to DB with dedup.

        Returns (success, renamed_path, data_dict).
        """
        renamed_path = None
        data_dict = None

        try:
            selected_year = self.app.year_box.get()
            selected_exam = self.app.exam_btn.get()
            selected_grade = self.app.grade_btn.get()
            roman_grade = to_roman(selected_grade)
            year_folder = os.path.join(self.app.BASE_SAVE_PATH, selected_year)
            exam_folder = os.path.join(
                year_folder, f"{selected_year} {roman_grade} {selected_exam}"
            )
            os.makedirs(exam_folder, exist_ok=True)

            saved_path = save_image_smart(img_path, exam_folder)
            data_dict, _ = get_vlm_result(saved_path)

            new_name = build_image_filename(data_dict)
            renamed_path = os.path.join(exam_folder, new_name)
            if os.path.normpath(saved_path) != os.path.normpath(renamed_path):
                if os.path.exists(renamed_path):
                    os.remove(renamed_path)
                os.rename(saved_path, renamed_path)
            else:
                renamed_path = saved_path

            self.app.current_vlm_raw = data_dict
            self.app.current_ocr_data = data_dict.get("students", [])
            student_count = len(self.app.current_ocr_data)

            meta = {
                "grade": selected_grade,
                "exam_type": selected_exam,
                "exam_year": selected_year,
                "book_name": None,
                "qc_check": "0",
                "qc_remarks": None,
                "cluster_id": None,
                "ui": "True",
                "remarks": None,
                "is_legacy_image": False,
            }

            # Save to DB with dedup
            saved_count = 0
            skipped_count = 0
            try:
                conn = _get_db()
                exam_id = get_exam_id(conn, meta["exam_year"], meta["grade"], meta["exam_type"])
                if exam_id:
                    new_students = []
                    for s in data_dict.get("students", []):
                        reg = s.get("registration_number", "") if isinstance(s, dict) else s.registration_number
                        if not is_student_already_in_db(conn, reg, exam_id):
                            new_students.append(s)
                        else:
                            skipped_count += 1
                    conn.close()
                    if new_students:
                        filtered_data = dict(data_dict)
                        filtered_data["students"] = new_students
                        save_vlm_to_database(filtered_data, renamed_path, meta=meta)
                        saved_count = len(new_students)
                else:
                    conn.close()
                    save_vlm_to_database(data_dict, renamed_path, meta=meta)
                    saved_count = student_count
            except Exception as db_err:
                self.app.after(0, lambda e=db_err: self.app.log_message(f"⚠️ DB check failed (saving anyway): {e}"))
                save_vlm_to_database(data_dict, renamed_path, meta=meta)
                saved_count = student_count

            log_msg = f"💾 {log_prefix}{new_name}: {saved_count} saved"
            if skipped_count:
                log_msg += f", {skipped_count} already in DB (skipped)"
            self.app.after(0, lambda m=log_msg: self.app.log_message(m))

            return True, renamed_path, data_dict

        except Exception as e:
            log_name = source_filename or os.path.basename(img_path)
            self.app.after(
                0,
                lambda f=log_name, err=str(e): self.app.log_message(f"❌ {f} FAILED: {err}"),
            )
            return False, renamed_path, data_dict

    # ── Batch processing thread ─────────────────────────────────

    def process_folder_thread(self):
        """Process all images in batch mode sequentially, with checkpoint updates."""
        app = self.app

        while app.batch_index < len(app.batch_files) and not app.batch_abort:
            orig_filename = app.batch_files[app.batch_index]
            app.batch_index += 1

            img_path = os.path.join(app.batch_folder, orig_filename)
            idx = app.batch_index
            total = app.batch_total

            app.after(
                0,
                lambda i=idx, t=total, f=orig_filename: app.log_message(f"\n[{i}/{t}] Processing: {f}"),
            )

            success, renamed_path, data_dict = self.process_and_save_image(
                img_path, source_filename=orig_filename, log_prefix=f"[{idx}/{total}] "
            )

            if success:
                new_name = os.path.basename(renamed_path)
                app.batch_file_map[orig_filename] = new_name
                app.batch_completed_originals.add(orig_filename)
                remaining = [f for f in app.batch_files if f not in app.batch_completed_originals]

                app.after(0, lambda p=renamed_path: self._set_preview(p))
                app.after(
                    0,
                    lambda sf=app.batch_folder, tt=app.batch_total,
                    fm=app.batch_file_map, co=app.batch_completed_originals,
                    rem=remaining: save_checkpoint(sf, tt, dict(fm), list(co), rem),
                )
                app.batch_success += 1
            else:
                app.batch_fail += 1

        app.after(0, self.finish_batch)

    # ── Batch cleanup ───────────────────────────────────────────

    def finish_batch(self):
        """Re-enable UI after batch processing completes and clean up checkpoint."""
        app = self.app
        app.is_batch_mode = False
        app.import_btn.configure(state="normal")
        app.import_folder_btn.configure(state="normal", text="FOLDER", image=FOLDER_ICON)
        app.proceed_btn.configure(state="normal", text="PROCEED TO OCR ENGINE", fg_color=app.COLORS["success"])
        app.scan_btn.configure(state="normal")

        total = app.batch_total
        success = app.batch_success
        fail = app.batch_fail

        if fail == 0 and success >= total:
            delete_checkpoint()
            app.log_message("🗑️ Checkpoint cleared — batch fully complete.")
        else:
            app.log_message("📝 Checkpoint preserved — resume available on restart.")

        app.log_message(f"\n{'=' * 60}")
        app.log_message(f"📊 BATCH RESULT: {success}/{total} succeeded")
        if fail > 0:
            app.log_message(f"   ❌ {fail} failed")
        app.log_message(f"{'=' * 60}")
        app.log_message("System: Batch processing finished.")

    # ── Preview helper (called from batch thread) ───────────────

    def _set_preview(self, img_path):
        """Update the preview display with a specific image (thread-safe)."""
        app = self.app
        app.selected_file = img_path
        if img_path and os.path.exists(img_path):
            app.rotation = 0
            app.zoom_level = 1.0
            app.display_image()
