import os
import json
from ocr_done_again.helper_using import NEB_OCR
from ocr_done_again.utils  import update_row_from_ocr, set_remarks
from ocr_done_again.database import NEBDB
import copy
import traceback
import datetime
import concurrent.futures
import threading
from tqdm import tqdm

lock = threading.Lock()

def process_item(item, neb, prefix, unmatched_ids, UNMATCHED_FILE, done_ids, DONE_FILE):
    try:
        rows = json.loads(item['rows_data'])
        row_ids = [row['id'] for row in rows]
        if any(rid in unmatched_ids for rid in row_ids) or any(rid in done_ids for rid in row_ids):
            return
        with lock:
            done_ids.add(row_ids[0])
            with open(DONE_FILE, "w") as f:
                json.dump(list(done_ids), f)
        if all(row['QC_REMARKS'] == '[0]' for row in rows):
            return
        recomputed_pass = True
        for row in rows:
            if row['QC_REMARKS'] != '[0]':
                # Run your QC check again
                if not set_remarks(dict(row)):   # convert to dict in case it's not
                    recomputed_pass = False
                    break
        if recomputed_pass:
            return  # skip this group because at least one failed QC again
        full_path = os.path.join(prefix, item['FILE_PATH'])
        # print(full_path)
        with lock:
            ocr_results = NEB_OCR(full_path)
        # Collect SYMBOLs
        ocr_symbols = {item["SYMBOL"] for item in ocr_results if item.get("SYMBOL")}
        row_symbols = {item["SYMBOL"] for item in rows if item.get("SYMBOL")}
        ocr_regs = {item["REG.NO."] for item in ocr_results if item.get("REG.NO.")}
        row_regs = {item["REG_NO"] for item in rows if item.get("REG_NO")}

        # Matches
        matched_symbols = ocr_symbols.intersection(row_symbols)
        matched_regs = ocr_regs.intersection(row_regs)
        unmatched_ocr_symbols = ocr_symbols - row_symbols
        unmatched_row_symbols = row_symbols - ocr_symbols

        # Separate rows
        matched_rows = [row for row in rows if row.get("SYMBOL") in matched_symbols]
        unmatched_rows = [row for row in rows if row.get("SYMBOL") in unmatched_row_symbols]

        unmatched_row_details = [
            row["id"]
            for row in rows
            if row.get("SYMBOL") in unmatched_row_symbols and not set_remarks(dict(row))
        ]
        if unmatched_rows:
            with lock:
                unmatched_ids.update(unmatched_row_details)
                with open(UNMATCHED_FILE, "w") as f:
                    json.dump(list(unmatched_ids), f)

        # Separate OCR
        matched_ocr = [ocr for ocr in ocr_results if ocr.get("SYMBOL") in matched_symbols]
        unmatched_ocr = [ocr for ocr in ocr_results if ocr.get("SYMBOL") in unmatched_ocr_symbols]

        # --- NEW STEP: check QC_REMARKS ---
        # bad_qc_rows = [row for row in matched_rows if str(row.get("QC_REMARKS")) != "[0]"]
        bad_qc_rows = []
        for row in matched_rows:
            ocr_candidate = next((ocr for ocr in matched_ocr if ocr["SYMBOL"] == row["SYMBOL"]), None)
            if not ocr_candidate:
                continue

            # Case 1: QC_REMARKS not zero
            if str(row.get("QC_REMARKS")) != "[0]":
                bad_qc_rows.append(row)
                continue

            # Case 2: SYMBOL matches but REG_NO does not
            if ocr_candidate.get("REG.NO.") != row.get("REG_NO"):
                bad_qc_rows.append(row)
                continue
        # Case 3: REG_NO matches but SYMBOL does not
        for row in rows:
            ocr_candidate = next((ocr for ocr in ocr_results if ocr.get("REG.NO.") == row.get("REG_NO")), None)
            if ocr_candidate and ocr_candidate.get("SYMBOL") != row.get("SYMBOL"):
                bad_qc_rows.append(row)

        # Remove duplicates if any
        bad_qc_rows = list({row['id']: row for row in bad_qc_rows}.values())

        # Print results
        # print(f"Matched SYMBOLs count: {len(matched_symbols)}")
        # print(f"Unmatched OCR SYMBOLs: {unmatched_ocr_symbols}")
        # print(f"Unmatched Row SYMBOLs: {unmatched_row_symbols}")
        # print(f"Matched rows with bad QC_REMARKS: {len(bad_qc_rows)}")

        # Updating bad QC
        fields_to_copy = [
            'BookName', 'Exam_Type', 'Exam_Year', 'FILE_PATH',
            'ImageName', 'School_Code', 'School_Name', 'Grade'
        ]
        matched_row_beta = next(
                        (r for r in rows if all(r.get(f) not in (None, "") for f in fields_to_copy)),
                        None
                    )
        for row in bad_qc_rows:
            row_copy = copy.deepcopy(row)
            bad_qc_ocr = next((ocr for ocr in matched_ocr if ocr["SYMBOL"] == row["SYMBOL"]), None)
            if bad_qc_ocr is None:
                continue
            if set_remarks(copy.deepcopy(bad_qc_ocr)):
                updated_row = update_row_from_ocr(row, copy.deepcopy(bad_qc_ocr))
                if updated_row:
                    if matched_row_beta:
                        for key in fields_to_copy:
                            if key not in updated_row or updated_row[key] in (None, ""):
                                updated_row[key] = matched_row_beta.get(key)
                    if set_remarks(updated_row):
                        with lock:
                            neb.update_with_log(row_copy, updated_row)
                    # print(f"Updated Row (from OCR Wrong): {updated_row}")
            else:
                pass
                # print("OCR did not pass QC, skipping update.")

        # Updating Dob of matched and good qc:
        for row in matched_rows:
            row_copy = copy.deepcopy(row)
            qc_ocr = next((ocr for ocr in matched_ocr if ocr["SYMBOL"] == row["SYMBOL"]), None)
            if qc_ocr is None:
                continue
            ocr_is_good = False
            if set_remarks(copy.deepcopy(qc_ocr)):
                ocr_is_good = True
                updated_row = update_row_from_ocr(row, copy.deepcopy(qc_ocr), dob=True)
                if updated_row:
                    if matched_row_beta:
                        for key in fields_to_copy:
                            if key not in updated_row or updated_row[key] in (None, ""):
                                updated_row[key] = matched_row_beta.get(key)
                    if set_remarks(updated_row):
                        with lock:
                            neb.update_with_log(row_copy, updated_row)
                    # print(f"Updated Row (from OCR DOB): {updated_row}")
            if ocr_is_good and  not set_remarks(copy.deepcopy(row)):
                updated_row = update_row_from_ocr(row, copy.deepcopy(qc_ocr))
                if updated_row:
                    if matched_row_beta:
                        for key in fields_to_copy:
                            if key not in updated_row or updated_row[key] in (None, ""):
                                updated_row[key] = matched_row_beta.get(key)
                    if set_remarks(updated_row):
                        with lock:
                            neb.update_with_log(row_copy, updated_row)
                    # print(f"Updated Row (from OCR Wrong): {updated_row}")
            else:
                pass
                # print("OCR did not pass QC, skipping update.")


        # Updating missed Rows
        key_map = {
                    "REG.NO.": "REG_NO",
                    "NAME OF THE STUDENT": "NAME_OF_THE_STUDENTS",
                    "REM": "RESULT",
                }
        for ocr in unmatched_ocr:
            if matched_row_beta:
                for key in fields_to_copy:
                    if key not in ocr or ocr[key] in (None, ""):
                        ocr[key] = matched_row_beta.get(key)
                r, pf = set_remarks(ocr, True)
                ocr['QC_REMARKS'] = str(r)
                ocr['QC_CHECK'] = pf
                ocr['REMARKS'] = None
                ocr['SEX'] = None
                for old_key, new_key in key_map.items():
                    if old_key in ocr:
                        ocr[new_key] = ocr.pop(old_key)
            if set_remarks(ocr):
                with lock:
                    neb.update_with_log(None, ocr)


    except Exception as e:
        with lock:
            with open(r"D:\OCR_CODE\ocr_done_again\error_log.txt", "a", encoding="utf-8") as f:
                f.write("\n" + "="*50 + "\n")
                f.write(f"Time: {datetime.datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Item: {str(item['FILE_PATH'])}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("="*50 + "\n")


if __name__ == "__main__":
    neb = NEBDB(table_name="neb_db_devraj_final_new", user_log="user_log")
    prefix = r"D:\xampp\htdocs\neb\img"
    exam__year = (2062, 2063)
    grouped_rows = neb.get_rows_grouped_by_path(exam__year)
    # grouped_rows = neb.get_rows_grouped_by_path(exam__year, "2069\\2069 XI Reg\\Book 1 2069 XI Reg 0101-0212\\121.jpg")

    UNMATCHED_FILE = r"D:\OCR_CODE\unmatched_ids.txt"
    if os.path.exists(UNMATCHED_FILE):
        with open(UNMATCHED_FILE, "r") as f:
            unmatched_ids = set(json.load(f))
    else:
        unmatched_ids = set()

    DONE_FILE = fr"D:\OCR_CODE\56_63_done_id.txt"
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r") as f:
            done_ids = set(json.load(f))
    else:
        done_ids = set()

    # 🔹 Use ThreadPoolExecutor or ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_item, item, neb, prefix, unmatched_ids, UNMATCHED_FILE, done_ids, DONE_FILE)
            for item in grouped_rows
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=len(futures), 
                           desc="Processing", 
                           unit="file"):
            future.result()  # re-raise exceptions if any

    print("Successful.")

    