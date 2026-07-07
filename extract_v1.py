import ollama
import os
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Define the exact, repeatable structure for EVERY student row
class SubjectScores(BaseModel):
    subject_name: str = Field(description="Name of the subject, e.g., C.ENG, C.NEP, PHY, CHEM, BIO, or MATHS")
    theory: Optional[str] = Field(None, description="Theory marks (TH), Preserve *, A, AB, etc.")
    practical: Optional[str] = Field(None, description="Practical marks (PR) if present.")
    total: Optional[str] = Field(None, description="Total marks for this subject (TOT), Total marks only if physically printed under this subject.")
    extra: bool = Field(
        description="True if this subject is a continuation subject appearing on the second row (e.g. MATHS, COMP, OPTIONAL). False otherwise."
    )

class StudentRecord(BaseModel):
    symbol_number: str
    registration_number: str
    student_name: str
    date_of_birth: str
    subjects: List[SubjectScores] = Field(description="List of all subjects and scores for this student")
    grand_total: str = Field(description="The TOT / TOTAL column score at the end")
    remark: str = Field(description="PASS, FAIL1, FAIL2, FAIL3, etc.")

class NEBGradingSheet(BaseModel):
    school_code: str = Field(
        description="School code is the 4 digit code that is on the left of the school name."
    )
    school_name: str
    page_number: str
    examination_year: str
    students: List[StudentRecord]
    handwritten_marginal_notes: List[str] = Field(description="Extract any handwritten text or correction notes found at the bottom or margins")

# 2. File verification
image_filename = "invoice.jpg"  # Ensure this matches your file name exactly
if not os.path.exists(image_filename):
    print(f"Error: Could not find '{image_filename}'")
    exit()

print("Extracting structured schema using Qwen3-VL-8B-Instruct...")

try:
    # 3. Request data with enforced schema constraint
    response = ollama.chat(
        model="qwen3-vl:8b-instruct",
        messages=[{
            "role": "user",
            "content": '''
                Perform precise OCR data extraction on this academic grading ledger sheet. 
                Carefully inspect each row. Note that some students have MATHS and other subjects in the second row as well if the row does not contain the registration number or student name and only contain the dob then also check for other subject in that row it is aligned with any of the subject header.
                Capture all marks exactly as printed, preserving any asterisks (*) or handwritten corrections.
                Also check out for header generally for the first subject of the student row it does not contain the pr or tot confirm that and make them null.
            ''',
            "images": [image_filename]
        }],
        format=NEBGradingSheet.model_json_schema(), # <-- Locks the output structure completely
        options={
            "num_ctx": 16384,       # Constrain context to exactly what this image needs
            "num_predict": 8192,   # Prevent early cutoff blocks
            "temperature": 0.0,    # Eliminate random text alterations
            "top_k": 1,            # Force model to pick only the absolute top tokens
            "top_p": 0.1,           # Restrict text variances completely
            "seed": 42,
            "repeat_penalty": 1.05,
        }
    )
    
    # 4. Save and Output clean JSON
    json_output = response['message']['content']
    print("\n--- Structured JSON Output ---")
    print(json_output)
    
    # Optional: Automatically saves the result to a clean dataset file
    output_json_path = os.path.splitext(image_filename)[0] + "_extracted1.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"\nSuccessfully saved structured data to: {output_json_path}")

except Exception as e:
    print(f"An error occurred: {e}")
