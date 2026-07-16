"""
Database commit and OCR processing – extracted from main.py.

``DbAndOcrProcessor`` owns the OCR pipeline (run → thread → finalize),
the manual DB save flow (confirmation dialog → thread), and the
auto-commit mode.  It receives the parent ``RadiantUltraScanner``
instance to access sidebar values, UI widgets, and shared state.
"""

import os
import threading
from tkinter import messagebox

import customtkinter as ctk

from neb_utils.vlm_front import get_vlm_result, save_vlm_to_database


class DbAndOcrProcessor:
    """Handles OCR execution and database commit operations."""

    def __init__(self, app):
        self.app = app

    # ── OCR pipeline ────────────────────────────────────────────

    def run_ocr(self):
        """Start the OCR engine on the currently selected image."""
        if self.app.selected_file:
            self.app.proceed_btn.configure(state="disabled", text="", fg_color=self.app.COLORS["sidebar"])
            self.app.btn_loader = ctk.CTkProgressBar(
                self.app.proceed_btn, width=280, height=8,
                mode="indeterminate", progress_color=self.app.COLORS["accent"],
            )
            self.app.btn_loader.place(relx=0.5, rely=0.5, anchor="center")
            self.app.btn_loader.start()
            self.app.log_message("OCR Engine: Extracting high-precision data...")
            threading.Thread(target=self.ocr_thread_logic, daemon=True).start()
        else:
            messagebox.showwarning(
                "System", "Input buffer empty. Please scan a document first."
            )

    def ocr_thread_logic(self, img_path=None):
        """Run VLM OCR in a background thread."""
        target = img_path or self.app.selected_file
        try:
            data_dict, summary_text = get_vlm_result(target)
            self.app.after(0, lambda: self.finalize_ocr_button(data_dict, img_path=target))
            self.app.after(0, lambda: self.app.log_message(summary_text))
        except Exception as e:
            err = f"VLM OCR Error on {os.path.basename(target)}: {str(e)}"
            self.app.after(0, lambda: self.app.log_message(err))
            self.app.after(0, lambda: self.finalize_ocr_button(error=True, img_path=target))

    def finalize_ocr_button(self, results=None, error=None, img_path=None):
        """Handle OCR completion — update UI, sync VLM data, auto-commit if enabled."""
        app = self.app

        if hasattr(app, "btn_loader") and app.btn_loader.winfo_exists():
            app.btn_loader.stop()
            app.btn_loader.destroy()

        if results:
            app.current_vlm_raw = results
            app.current_ocr_data = results.get("students", [])

            # Auto-update sidebar inputs from VLM extraction
            self._sync_ui_from_vlm(results)

            student_count = len(app.current_ocr_data)
            img_name = os.path.basename(img_path) if img_path else app.selected_file
            img_name = os.path.basename(img_name) if img_name else "?"

            app.log_message(f"✅ {img_name}: {student_count} students extracted.")

            if app.auto_commit.get():
                app.log_message("⏳ Auto-committing to database...")
                img_target = img_path or app.selected_file
                threading.Thread(
                    target=self._auto_save_to_db, args=(img_target,), daemon=True
                ).start()
            else:
                app.proceed_btn.configure(
                    state="normal", text="PROCEED TO OCR ENGINE",
                    fg_color=app.COLORS["success"],
                )
                app.db_save_btn.configure(state="normal", fg_color=app.COLORS["success"])
                app.log_message("System: Records verified. Ready for Database Commit.")

        if error:
            app.proceed_btn.configure(
                state="normal", text="PROCEED TO OCR ENGINE",
                fg_color=app.COLORS["success"],
            )
            app.log_message(f"VLM OCR Error: {error}")

    def _sync_ui_from_vlm(self, data: dict):
        """Sync sidebar input widgets with values extracted by the VLM."""
        app = self.app

        # --- Year ---
        extracted_year = str(data.get("examination_year", "")).strip()
        if extracted_year.isdigit():
            try:
                current_years = app.year_box.cget("values")
                if extracted_year in current_years:
                    app.year_box.set(extracted_year)
                else:
                    updated = list(current_years) + [extracted_year]
                    updated = sorted(set(updated), key=lambda x: int(x), reverse=True)
                    app.year_box.configure(values=updated)
                    app.year_box.set(extracted_year)
            except Exception:
                pass

        # --- Grade ---
        extracted_grade = str(data.get("grade", "")).strip().lower()
        grade_map = {"eleven": "11", "twelve": "12", "11": "11", "12": "12"}
        if extracted_grade in grade_map:
            try:
                app.grade_btn.set(grade_map[extracted_grade])
            except Exception:
                pass

        # --- Exam Type ---
        extracted_type = str(data.get("exame_Type", "")).strip()
        exam_type_map = {
            "regular": "Regular",
            "partial": "Partial",
            "supplementary": "Supplementary",
        }
        mapped_type = exam_type_map.get(extracted_type.lower())
        if mapped_type:
            try:
                app.exam_btn.set(mapped_type)
            except Exception:
                pass

    # ── Manual DB commit ────────────────────────────────────────

    def save_to_database(self):
        """Start the database commit flow (with confirmation dialog)."""
        if not self.app.current_ocr_data:
            self.app.log_message("Database Error: No data buffer found to commit.")
            return

        if not self._confirm_db_commit():
            self.app.log_message("Database Commit: Cancelled by user.")
            return

        self.app.log_message("Database: Initializing secure handshake...")
        self.app.db_save_btn.configure(state="disabled", text="⌛ COMMITTING...")
        threading.Thread(target=self.db_thread_logic, daemon=True).start()

    def _confirm_db_commit(self) -> bool:
        """Show a polished confirmation dialog with extracted details."""
        app = self.app
        raw = app.current_vlm_raw
        if not raw:
            return False

        school = raw.get("school_name", "?")
        school_code = raw.get("school_code", "?")
        grade = raw.get("grade", "?")
        exam_type = raw.get("exame_Type", "?")
        year = raw.get("examination_year", "?")
        page = raw.get("page_number", "?")
        students = raw.get("students", [])

        detail_lines = [
            f"School : {school} ({school_code})",
            f"Grade  : {grade}",
            f"Type   : {exam_type}",
            f"Year   : {year}",
            f"Page   : {page}",
            f"Students: {len(students)}",
            "",
            "─" * 40,
        ]

        preview_count = min(5, len(students))
        for idx in range(preview_count):
            s = students[idx]
            name = s.get("student_name", "?")
            symbol = s.get("symbol_number", "?")
            total = s.get("grand_total", "?")
            remark = s.get("remark", "?")
            detail_lines.append(f"{idx+1}. {name}  [{symbol}]  Total: {total}  {remark}")

        if len(students) > preview_count:
            detail_lines.append(f"   ... and {len(students) - preview_count} more")

        detail_lines.append("")
        detail_lines.append("Override values from sidebar will be used for:")
        detail_lines.append(f"  Grade (UI): {app.grade_btn.get()}")
        detail_lines.append(f"  Type  (UI): {app.exam_btn.get()}")
        detail_lines.append(f"  Year  (UI): {app.year_box.get()}")

        details = "\n".join(detail_lines)

        # Build the dialog
        C = app.COLORS
        dialog = ctk.CTkToplevel(app)
        dialog.title("Confirm Database Commit")
        dialog.geometry("520x520")
        dialog.configure(fg_color=C["bg"])
        dialog.transient(app)
        dialog.grab_set()

        header = ctk.CTkLabel(
            dialog,
            text="📋 Confirm OCR Data Before Saving",
            font=("Inter", 16, "bold"),
            text_color=C["accent"],
        )
        header.pack(pady=(20, 10))

        sep = ctk.CTkFrame(dialog, height=2, fg_color=C["border"])
        sep.pack(fill="x", padx=30, pady=(0, 10))

        text_box = ctk.CTkTextbox(
            dialog,
            height=280,
            fg_color=C["card"],
            border_width=1,
            border_color=C["border"],
            font=("JetBrains Mono", 12),
            text_color=C["text_main"],
        )
        text_box.pack(fill="both", expand=True, padx=30, pady=(0, 15))
        text_box.insert("1.0", details)
        text_box.configure(state="disabled")

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(0, 20))

        result = [False]

        def on_confirm():
            result[0] = True
            dialog.destroy()

        def on_cancel():
            result[0] = False
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="CANCEL",
            command=on_cancel,
            width=180,
            height=45,
            corner_radius=12,
            fg_color="transparent",
            border_width=1,
            border_color=C["border"],
            font=("Inter", 13, "bold"),
        ).pack(side="left", expand=True, padx=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="✅ CONFIRM & SAVE",
            command=on_confirm,
            width=180,
            height=45,
            corner_radius=12,
            fg_color=C["success"],
            font=("Inter", 13, "bold"),
        ).pack(side="right", expand=True, padx=(10, 0))

        app.wait_window(dialog)
        return result[0]

    def db_thread_logic(self):
        """Save OCR data to the database in a background thread."""
        app = self.app
        try:
            if not app.current_vlm_raw:
                raise ValueError("No VLM OCR data available. Run OCR first.")

            meta = {
                "grade": app.grade_btn.get(),
                "exam_type": app.exam_btn.get(),
                "exam_year": app.year_box.get(),
                "book_name": None,
                "qc_check": "0",
                "qc_remarks": None,
                "cluster_id": None,
                "ui": "True",
                "remarks": None,
                "is_legacy_image": False,
            }

            save_vlm_to_database(app.current_vlm_raw, app.selected_file, meta=meta)

            student_count = len(app.current_ocr_data)

            app.after(
                0,
                lambda: app.log_message(
                    f"Database: [COMMIT_SUCCESS] {student_count} records saved."
                ),
            )
            app.after(
                0,
                lambda: app.db_save_btn.configure(
                    text="COMMIT TO DATABASE", fg_color=app.COLORS["border"]
                ),
            )
            app.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Successfully saved {student_count} student records.",
                ),
            )

        except Exception as e:
            error_msg = f"Database Critical Failure: {str(e)}"
            app.after(0, lambda: app.log_message(error_msg))
            app.after(
                0,
                lambda: app.db_save_btn.configure(
                    state="normal",
                    text="📥 RETRY COMMIT",
                    fg_color=app.COLORS["danger"],
                ),
            )

    # ── Auto-commit mode ────────────────────────────────────────

    def toggle_auto_commit(self):
        """Enable/disable the DB save button based on the auto-commit checkbox."""
        app = self.app
        if app.auto_commit.get():
            app.db_save_btn.configure(
                state="disabled", fg_color=app.COLORS["border"], text="AUTO-COMMIT ON"
            )
        else:
            if app.current_ocr_data:
                app.db_save_btn.configure(
                    state="normal", fg_color=app.COLORS["success"],
                    text="COMMIT TO DATABASE",
                )
            else:
                app.db_save_btn.configure(
                    state="disabled", fg_color=app.COLORS["border"],
                    text="COMMIT TO DATABASE",
                )

    def _auto_save_to_db(self, img_path: str):
        """Save OCR result to DB without confirmation (auto-commit mode)."""
        app = self.app
        if not app.current_vlm_raw:
            return

        meta = {
            "grade": app.grade_btn.get(),
            "exam_type": app.exam_btn.get(),
            "exam_year": app.year_box.get(),
            "book_name": None,
            "qc_check": "0",
            "qc_remarks": None,
            "cluster_id": None,
            "ui": "True",
            "remarks": None,
            "is_legacy_image": False,
        }

        try:
            save_vlm_to_database(app.current_vlm_raw, img_path, meta=meta)
            student_count = len(app.current_ocr_data)
            app.log_message(f"💾 Auto-committed {student_count} records to DB.")
        except Exception as e:
            app.log_message(f"❌ Auto-commit failed: {e}")

        if not app.is_batch_mode:
            app.proceed_btn.configure(
                state="normal",
                text="PROCEED TO OCR ENGINE",
                fg_color=app.COLORS["success"],
            )
