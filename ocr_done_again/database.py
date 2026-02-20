import os
import mysql.connector
from datetime import datetime
import pytz

class NEBDB:
    def __init__(self):
        self.db = mysql.connector.connect(
            host=os.getenv("HOST"),
            port=int(os.getenv("PORT", 3306)),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            database=os.getenv("DATABASE"),
        )

        self.table_name = os.getenv("TABLE_NAME")
        self.user_log = os.getenv("USER_LOG")  # optional fallback

        self.cursor = self.db.cursor(dictionary=True)

    def get_all_columns(self):
        query = """
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """
        self.cursor.execute(query, (self.db.database, self.table_name))
        return [row['COLUMN_NAME'] for row in self.cursor.fetchall()]

    def get_rows_grouped_by_path(self, exam_year, file_path=None):
        columns = self.get_all_columns()
        json_columns = ", ".join([f"'{col}', {col}" for col in columns])
        placeholders = ", ".join(["%s"] * len(exam_year))
        query = f"""
            SELECT 
                FILE_PATH,
                JSON_ARRAYAGG(JSON_OBJECT({json_columns})) AS rows_data
            FROM {self.table_name}
            WHERE Exam_Year in ({placeholders})
        """
        
        params = list(exam_year)

        # Add filter if file_path is provided AND JSON_CONTAINS(QC_REMARKS, '7', '$') 
        if file_path:
            query += " AND FILE_PATH = %s"
            params.append(file_path)
        else:
            query += f"""
                    AND FILE_PATH IN (
                        SELECT FILE_PATH
                        FROM {self.table_name}
                        WHERE Exam_Year in ({placeholders})
                        AND QC_REMARKS <> '0'
                    )
                """
            params.extend(exam_year)

        query += " GROUP BY FILE_PATH"

        self.cursor.execute(query, tuple(params))
        return self.cursor.fetchall()
    
    def update_with_log(self, row, updated_row, user_id=566, merged_or_not=0):
        """
        Insert old + new row into user_log, then update main table.
        :param table_name: Main table name (e.g., neb_db_devraj_final_new)
        :param row: Old DB row (dict)
        :param updated_row: Updated row (dict from OCR)
        :param user_id: Logged-in user id
        :param merged_or_not: 0 or 1
        """
        if row is None:
            cols = [k for k in updated_row.keys() if k != "id"]
            placeholders = ", ".join(["%s"] * len(cols))
            cols_sql = ", ".join(cols)

            insert_query = f"INSERT INTO {self.table_name} ({cols_sql}) VALUES ({placeholders})"
            values = [updated_row[c] for c in cols]

            self.cursor.execute(insert_query, values)
            self.db.commit()

            row_id = self.cursor.lastrowid  # MySQL way to get new id
            row_obj = None  # no old row
        else:

            update_cols = [f"{k}=%s" for k in updated_row.keys() if k != "id"]
            update_query = f"UPDATE {self.table_name} SET {', '.join(update_cols)} WHERE id=%s"
            values = [updated_row[k] for k in updated_row.keys() if k != "id"] + [row["id"]]

            self.cursor.execute(update_query, values)
            self.db.commit()

            row_id = row["id"]
            row_obj = row
            
        if self.user_log:
            if row_obj is None:
                # If no old row, old_* = None
                log_data = {k: v for k, v in updated_row.items() if k != "id" and k!= "Cluster_ID"}
                for k in updated_row.keys():
                    if k != "id" and k!= "Cluster_ID":
                        log_data[f"old_{k}"] = None
            else:
                log_data = {**{k: v for k, v in updated_row.items() if k != "id" and k!= "Cluster_ID"}, **{f"old_{k}": row.get(k) for k in row if k != "id" and k!= "Cluster_ID"}}
            
            nepal_tz = pytz.timezone("Asia/Kathmandu")
            created_at = datetime.now(nepal_tz).strftime("%Y-%m-%d %H:%M:%S")

            log_data.update({
                "user_id": user_id,
                "updated_db_id": row_id,
                "mearged_or_not": merged_or_not,
                "created_at": created_at
            })

            self.cursor.execute(f"SELECT id FROM {self.user_log} WHERE updated_db_id=%s LIMIT 1", (row_id, ))
            existing = self.cursor.fetchone()
            if existing:
                # --- Update log ---
                update_cols = [f"{k}=%s" for k in log_data.keys()]
                update_query = f"UPDATE {self.user_log} SET {', '.join(update_cols)} WHERE updated_db_id=%s"
                values = list(log_data.values()) + [row_id]
                self.cursor.execute(update_query, values)
            else:
                log_query = f"""
                INSERT INTO {self.user_log} (
                    user_id, updated_db_id, mearged_or_not, created_at,
                    Grade, Exam_Type, Exam_Year, School_Code, School_Name, SYMBOL, REG_NO, NAME_OF_THE_STUDENTS,
                    SEX, DOB, CODE1, TH1, PR1, TOT1, CODE2, TH2, PR2, TOT2,
                    CODE3, TH3, PR3, TOT3, CODE4, TH4, PR4, TOT4, CODE5, TH5, PR5, TOT5,
                    CODE6, TH6, PR6, TOT6, CODE7, TH7, PR7, TOT7,
                    TOTAL, RESULT, REMARKS, BookName, ImageName, QC_CHECK, QC_REMARKS, FILE_PATH, UI,
                    old_Grade, old_Exam_Type, old_Exam_Year, old_School_Code, old_School_Name, old_SYMBOL, old_REG_NO,
                    old_NAME_OF_THE_STUDENTS, old_SEX, old_DOB, old_CODE1, old_TH1, old_PR1, old_TOT1,
                    old_CODE2, old_TH2, old_PR2, old_TOT2, old_CODE3, old_TH3, old_PR3, old_TOT3,
                    old_CODE4, old_TH4, old_PR4, old_TOT4, old_CODE5, old_TH5, old_PR5, old_TOT5,
                    old_CODE6, old_TH6, old_PR6, old_TOT6, old_CODE7, old_TH7, old_PR7, old_TOT7,
                    old_TOTAL, old_RESULT, old_REMARKS, old_BookName, old_ImageName, old_QC_CHECK, old_QC_REMARKS, old_FILE_PATH, old_UI
                ) VALUES (
                    %(user_id)s, %(updated_db_id)s, %(mearged_or_not)s, %(created_at)s,
                    %(Grade)s, %(Exam_Type)s, %(Exam_Year)s, %(School_Code)s, %(School_Name)s, %(SYMBOL)s, %(REG_NO)s, %(NAME_OF_THE_STUDENTS)s,
                    %(SEX)s, %(DOB)s, %(CODE1)s, %(TH1)s, %(PR1)s, %(TOT1)s, %(CODE2)s, %(TH2)s, %(PR2)s, %(TOT2)s,
                    %(CODE3)s, %(TH3)s, %(PR3)s, %(TOT3)s, %(CODE4)s, %(TH4)s, %(PR4)s, %(TOT4)s, %(CODE5)s, %(TH5)s, %(PR5)s, %(TOT5)s,
                    %(CODE6)s, %(TH6)s, %(PR6)s, %(TOT6)s, %(CODE7)s, %(TH7)s, %(PR7)s, %(TOT7)s,
                    %(TOTAL)s, %(RESULT)s, %(REMARKS)s, %(BookName)s, %(ImageName)s, %(QC_CHECK)s, %(QC_REMARKS)s, %(FILE_PATH)s, %(UI)s,
                    %(old_Grade)s, %(old_Exam_Type)s, %(old_Exam_Year)s, %(old_School_Code)s, %(old_School_Name)s, %(old_SYMBOL)s, %(old_REG_NO)s,
                    %(old_NAME_OF_THE_STUDENTS)s, %(old_SEX)s, %(old_DOB)s, %(old_CODE1)s, %(old_TH1)s, %(old_PR1)s, %(old_TOT1)s,
                    %(old_CODE2)s, %(old_TH2)s, %(old_PR2)s, %(old_TOT2)s, %(old_CODE3)s, %(old_TH3)s, %(old_PR3)s, %(old_TOT3)s,
                    %(old_CODE4)s, %(old_TH4)s, %(old_PR4)s, %(old_TOT4)s, %(old_CODE5)s, %(old_TH5)s, %(old_PR5)s, %(old_TOT5)s,
                    %(old_CODE6)s, %(old_TH6)s, %(old_PR6)s, %(old_TOT6)s, %(old_CODE7)s, %(old_TH7)s, %(old_PR7)s, %(old_TOT7)s,
                    %(old_TOTAL)s, %(old_RESULT)s, %(old_REMARKS)s, %(old_BookName)s, %(old_ImageName)s, %(old_QC_CHECK)s, %(old_QC_REMARKS)s, %(old_FILE_PATH)s, %(old_UI)s
                )
                """
                self.cursor.execute(log_query, log_data)
        self.db.commit()

        # --- Update main table ---

        return True