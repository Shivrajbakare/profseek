import re
import pandas as pd
from config import DATA_DIR
import os

# Load all known course codes once (from your CSV)


def load_all_course_codes():
    grades_path = os.path.join(DATA_DIR, "course_grades_rows.csv")
    if os.path.exists(grades_path):
        df = pd.read_csv(grades_path)
        return set(df['course_code'].dropna().str.upper().str.replace(" ", "").unique())
    return set()


ALL_COURSE_CODES = load_all_course_codes()


def extract_course_code(text):
    """
    Extracts a valid IITK course code from the user input.
    Works for any course like:
    SEE624, SEE 624, CS201A, MSE 303, ESO207, ME601A, etc.
    """
    print("Running this fn!")
    # Regex: 2–4 letters + optional space + 2–3 digits + optional letter
    pattern = r'\b([A-Z]{2,4}\s?\d{2,3}[A-Z]?)\b'
    matches = re.findall(pattern, text.upper())

    # Clean and match against known codes
    for m in matches:
        code = m.replace(" ", "").upper()
        if code in ALL_COURSE_CODES:
            return code

    return None



# src/planner.py

import re

# dimport re

import re

def detect_intent(user_input: str):
    text = user_input.lower().strip()

    # 1️⃣ Specific course info
    match = re.search(r'\b([A-Z]{2,3}\d{3})\b', user_input)
    if match:
        return {"intent": "get_course_info", "course_code": match.group(1).upper()}

    # 2️⃣ Top courses query like "average grade ≥ 8", "avg > 7", "good grading courses"
    if "average grade" in text or "avg grade" in text or "grading" in text:
        num_match = re.search(r'(\d+(\.\d+)?)', text)
        if num_match:
            return {"intent": "get_top_courses", "min_grade": float(num_match.group(1))}
        else:
            return {"intent": "get_top_courses", "min_grade": 8.0}  # default

    # 3️⃣ Subject-based request (AI, ML, Data, etc.)
    if any(k in text for k in ["ai", "machine learning", "ml", "data", "deep learning"]):
        return {"intent": "get_subject_courses", "subject": "AI"}

    # 4️⃣ Fallback
    return {"intent": "general"}



# def detect_intent(user_input: str):
#     user_input = user_input.strip().lower()
#     course_code = extract_course_code(user_input)

#     if course_code:
#         return {"intent": "get_course_info", "course_code": course_code}
#     else:
#         return {"intent": "unknown", "course_code": None}
