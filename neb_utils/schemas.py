"""
schemas.py — Shared Pydantic models for NEB OCR output.
Both the old pipeline and the new Ollama pipeline import from here.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SubjectScores(BaseModel):
    subject_name: str = Field(description="Subject code, e.g. C.ENG, C.NEP, PHY, CHEM, BIO, MATHS")
    theory: Optional[str] = Field(None, description="Theory marks (TH), preserve *, A, AB etc.")
    practical: Optional[str] = Field(None, description="Practical marks (PR)")
    total: Optional[str] = Field(None, description="Total marks for this subject (TOT)")
    extra: bool = Field(
        default=False,
        description="True if this subject is a 2nd-row continuation subject (e.g. MATHS, COMP, OPTIONAL)"
    )


class StudentRecord(BaseModel):
    symbol_number: str
    registration_number: str
    student_name: str
    date_of_birth: str
    subjects: List[SubjectScores] = Field(description="All subjects and scores for this student")
    grand_total: str = Field(description="TOTAL / TOT column score at the end")
    remark: str = Field(description="PASS, FAIL1, FAIL2, ..., FAIL7")


class NEBGradingSheet(BaseModel):
    school_code: str = Field(description="4-digit school code")
    school_name: str
    page_number: str
    examination_year: str
    grade: str = Field(description="Grade: Eleven or Twelve")
    exame_Type: str = Field(description="Exam type: Regular, Partial, or Supplementary")
    students: List[StudentRecord]
    handwritten_marginal_notes: List[str] = Field(
        default_factory=list,
        description="Extract any handwritten text or correction notes found at bottom or margins"
    )
