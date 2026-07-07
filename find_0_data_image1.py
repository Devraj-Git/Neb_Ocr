import pandas as pd
import mysql.connector
import os, sys, re
from collections import defaultdict
from dotenv import load_dotenv

from ocr_done_again.database import NEBDB
from ocr_done_again.helper_using import NEB_OCR
load_dotenv()

def normalize_path(p):
    if not isinstance(p, str):
        return p
    return p.replace("\\", "/").strip()

def extract_meta_from_path(full_path, m):
    path = full_path.replace("\\", "/")
    parts = path.split("/")

    # --- Defaults ---
    exam_year = None
    grade = None
    exam_type = None
    book_name = None

    # --- 1. Exam Year ---
    # Example: /2070/
    for p in parts:
        if re.fullmatch(r"20\d{2}", p):
            exam_year = p
            break

    # --- 2. Find folder like: "2070 XI Reg"
    for p in parts:
        if "XI" in p or "XII" in p:
            # Grade
            if re.search(r"\bXII\b", p):
                grade = "Twelve"
            elif re.search(r"\bXI\b", p):
                grade = "Eleven"
            # Exam Type
            if "Reg" in p:
                exam_type = "Regular"
            elif "Partial" in p:
                exam_type = "Partial"
            elif "Supplementary" in p:
                exam_type = "Supplementary"

    # --- 3. Book Name ---
    # Example: "Book 1 2070 XI Reg 0101-0214"
    for p in parts:
        if "Book" in p:
            match = re.search(r"Book\s*\d+", p)
            if match:
                book_name = match.group()

    # --- 4. Image Name ---
    image_name = os.path.basename(full_path)

    return {
        "Grade": grade,
        "Exam_Type": exam_type,
        "Exam_Year": exam_year,
        "BookName": book_name,
        "ImageName": image_name,
        "FILE_PATH": m,
        "Cluster_ID": None
    }

def prepare_db_row(ocr_row, meta):
    """
    ocr_row: one dict from OCR
    meta: extra info (Grade, Year, file path etc.)
    """

    def clean(val):
        if val is None:
            return None
        val = str(val).strip()
        return val if val != "" else None

    return {
        "Grade": meta.get("Grade"),
        "Exam_Type": meta.get("Exam_Type"),
        "Exam_Year": meta.get("Exam_Year"),

        "School_Code": clean(ocr_row.get("School_Code")),
        "School_Name": clean(ocr_row.get("School_Name")),

        "SYMBOL": clean(ocr_row.get("SYMBOL")),
        "REG_NO": clean(ocr_row.get("REG.NO.")),
        "NAME_OF_THE_STUDENTS": clean(ocr_row.get("NAME OF THE STUDENT")),

        "SEX": None,
        "DOB": clean(ocr_row.get("DOB")),

        "CODE1": clean(ocr_row.get("CODE1")),
        "TH1": clean(ocr_row.get("TH1")),
        "PR1": clean(ocr_row.get("PR1")),
        "TOT1": clean(ocr_row.get("TOT1")),

        "CODE2": clean(ocr_row.get("CODE2")),
        "TH2": clean(ocr_row.get("TH2")),
        "PR2": clean(ocr_row.get("PR2")),
        "TOT2": clean(ocr_row.get("TOT2")),

        "CODE3": clean(ocr_row.get("CODE3")),
        "TH3": clean(ocr_row.get("TH3")),
        "PR3": clean(ocr_row.get("PR3")),
        "TOT3": clean(ocr_row.get("TOT3")),

        "CODE4": clean(ocr_row.get("CODE4")),
        "TH4": clean(ocr_row.get("TH4")),
        "PR4": clean(ocr_row.get("PR4")),
        "TOT4": clean(ocr_row.get("TOT4")),

        "CODE5": clean(ocr_row.get("CODE5")),
        "TH5": clean(ocr_row.get("TH5")),
        "PR5": clean(ocr_row.get("PR5")),
        "TOT5": clean(ocr_row.get("TOT5")),

        "CODE6": clean(ocr_row.get("CODE6")),
        "TH6": clean(ocr_row.get("TH6")),
        "PR6": clean(ocr_row.get("PR6")),
        "TOT6": clean(ocr_row.get("TOT6")),

        "CODE7": clean(ocr_row.get("CODE7")),
        "TH7": clean(ocr_row.get("TH7")),
        "PR7": clean(ocr_row.get("PR7")),
        "TOT7": clean(ocr_row.get("TOT7")),

        "TOTAL": clean(ocr_row.get("TOTAL")),
        "RESULT": clean(ocr_row.get("REM")),
        "REMARKS": None,

        "BookName": meta.get("BookName"),
        "ImageName": meta.get("ImageName"),
        "QC_CHECK": "PASS",
        "QC_REMARKS": None,
        "FILE_PATH": meta.get("FILE_PATH"),
        "Cluster_ID": meta.get("Cluster_ID"),
        "UI": 2,
    }

if __name__ == '__main__':
    db = mysql.connector.connect(
            host=os.getenv("HOST"),
            port=int(os.getenv("PORT", 3306)),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            database=os.getenv("DATABASE"),
        )
    
    which_year = 2069       
    DELETE_MISSING_PATHS = False # 2058 True 

    cursor = db.cursor()
    sql ="""
        SELECT 
            TRIM(REPLACE(FILE_PATH, '\\\\', '/')) AS normalized_path,
            COUNT(*) AS data,
            EXAM_YEAR
        FROM db_neb.neb_db_devraj_final_new
        GROUP BY normalized_path, EXAM_YEAR;
    """
    cursor.execute(sql)
    rows = cursor.fetchall()
    real_time_set = {
        row[0]: {  # already normalized_path from SQL
            "count": row[1],
            "exam_year": str(row[2])
        }
        for row in rows
    }
    real_count_main = 0
    database_rows_count=0
    counted_file_paths = set()
    imageToBeUpdated=[]
    # NEW: Excel→DB missing tracking
    excel_not_in_db = []
    excel_missing_total = 0

    source_path = f'D:/year/{which_year}/'
    excel_file_paths = set()
    file_path_to_excels  = defaultdict(set)
    database_path_counts = defaultdict(int)

    for file_name in os.listdir(source_path):
        # if file_name == "Book 6 2058 XI Partial  3601-4018.xlsx":
        #     print("pass")
        file_path = os.path.join(source_path, file_name)
        rows = pd.read_excel(file_path)    
        
        for row in rows.iterrows():
            try:
                FILE_PATH = normalize_path(row[1]['FILE_PATH'])
                file_path_to_excels[FILE_PATH].add(file_name)
            except Exception as e:
                print(file_path,e)
                sys.exit(0)

            excel_file_paths.add(FILE_PATH)
            original_count = row[1].iloc[1]
            real_count = row[1]['Real_Count']
            if pd.isna(original_count):
                original_count = 0  
            if pd.isna(real_count):
                real_count = "no"
            else:
                try:
                    real_count = int(real_count)
                except (ValueError, TypeError):
                    real_count = "no"
            
            if FILE_PATH in real_time_set:
                database_path_counts[FILE_PATH] += real_time_set[FILE_PATH]["count"]
                db_exam_year = real_time_set[FILE_PATH]["exam_year"]
                db_count = real_time_set[FILE_PATH]["count"]
                if db_exam_year != str(which_year):
                    print(f"Year mismatch for '{FILE_PATH}' in '{file_path}': DB={db_exam_year}, Expected={which_year}")
                    print("")
            else: # In excel but not in database
                # Missing total count
                if real_count != "no":
                    value = real_count
                else:
                    value = original_count

                if int(value) > 0:
                    if int(value) > 20:
                        print("Count is more then 20 in:- ",file_path)
                    excel_not_in_db.append(FILE_PATH)
                    excel_missing_total += value

                    
            if real_count != "no": # if manually updated
                real_count_main += int(real_count)
                if FILE_PATH in real_time_set:
                    database_rows_count += real_time_set[FILE_PATH]["count"]
                    if real_count != real_time_set[FILE_PATH]["count"]:
                        imageToBeUpdated.append(FILE_PATH)
            else:
                real_count_main += original_count
                if FILE_PATH in real_time_set:
                    database_rows_count += real_time_set[FILE_PATH]["count"]
                    if original_count != real_time_set[FILE_PATH]["count"]:
                        imageToBeUpdated.append(FILE_PATH)
    
    print(f"{which_year} :- ")
    print(f"Image To Be Updated:- {len(imageToBeUpdated)}")
    print(f"Image To Be add:- {len(excel_not_in_db)}")
    prefix = r"D:\xampp\htdocs\neb\img"
    for m in excel_not_in_db:
        full_path = os.path.join(prefix, m)
        print(full_path)
        try:
            ocr_results = NEB_OCR(full_path)
            # print(ocr_results)

            db = NEBDB()
            meta = extract_meta_from_path(full_path,m)

            for ocr_row in ocr_results:
                db_row = prepare_db_row(ocr_row, meta)
                # print(db_row)
                # --- OPTIONAL: check if already exists ---
                # db.cursor.execute(
                #     f"SELECT * FROM {db.table_name} WHERE REG_NO=%s and FILE_PATH=%s",
                #     (db_row["REG_NO"], db_row["FILE_PATH"])
                # )
                # existing = db.cursor.fetchone()
                # if existing :
                #     print(existing)
                # else:
                db.update_with_log(None, db_row, user_id=566, merged_or_not=0)
                # break

            # break 
        except Exception as e:
            print("Got error:",e)

    print("Successful !!")