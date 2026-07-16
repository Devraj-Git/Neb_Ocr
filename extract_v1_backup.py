from enum import Enum
import time

from PIL import Image
import ollama
import os
from pydantic import BaseModel, Field
from typing import List, Optional

class Grade(str, Enum):
    ELEVEN = "Eleven"
    TWELVE = "Twelve"

class ExamType(str, Enum):
    REGULAR = "Regular"
    PARTIAL = "Partial"
    SUPPLEMENTARY = "Supplementary"

class Remark(str, Enum):
    PASS = "PASS"
    FAIL1 = "FAIL1"
    FAIL2 = "FAIL2"
    FAIL3 = "FAIL3"
    FAIL4 = "FAIL4"
    FAIL5 = "FAIL5"
    FAIL6 = "FAIL6"
    FAIL7 = "FAIL7"
    
# 1. Define the exact, repeatable structure for EVERY student row
class SubjectScores(BaseModel):
    subject_name: str
    theory: Optional[str]
    practical: Optional[str]
    total: Optional[str]
    extra: bool

class StudentRecord(BaseModel):
    symbol_number: str
    registration_number: str
    student_name: str
    date_of_birth: str
    subjects: List[SubjectScores]
    grand_total: str
    remark: Remark

class NEBGradingSheet(BaseModel):
    school_code: int = Field(
        description="School code is the 4 digit code that is on the left of the school name."
    )
    school_name: str
    page_number: int = Field(description="Only Digits/Numbers allowed.")
    examination_year: int = Field(description="Only Digits/Numbers of BS Years allowed.")
    grade: Grade
    exame_Type: ExamType
    students: List[StudentRecord]
    
# 2. File verification
image_filename = "77.jpg"  # Ensure this matches your file name exactly
image_new_filename = "resized12.png"
if not os.path.exists(image_filename):
    print(f"Error: Could not find '{image_filename}'")
    exit()

img = Image.open(image_filename)
img.thumbnail((1400, 1400))
img.save(image_new_filename)

print("Extracting structured schema using Qwen3-VL-8B-Instruct...")

try:
    t0 = time.time()
    # 3. Request data with enforced schema constraint
    response = ollama.chat(
        model="qwen3-vl:8b-instruct",
        messages=[{
            "role": "user",
            "content": """
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
                        """,
            "images": [image_new_filename]
        }],
        format=NEBGradingSheet.model_json_schema(),
        options={
            "num_ctx": 10288,
            # "num_predict": 8192,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0.1,
            "seed": 42,
            "repeat_penalty": 1.0,
        }
    )
    print(f"Elapsed: {time.time() - t0:.2f}s")
    # 4. Save and Output clean JSON
    json_output = response['message']['content']
    # print("\n--- Structured JSON Output ---")
    # print(json_output)
    
    # Optional: Automatically saves the result to a clean dataset file
    output_json_path = os.path.splitext(image_filename)[0] + "_extracted2.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"\nSuccessfully saved structured data to: {output_json_path}")

except Exception as e:
    print(f"An error occurred: {e}")
