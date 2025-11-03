import os
import pandas as pd
import streamlit as st
from config import DATA_DIR
from preprocess import load_data, plot_grade_distribution


def compute_course_summary(course_code, grades_df, reviews_df):
    """
    Combines grading + review info for a specific course code.
    Returns a structured dict with course name, instructor, avg grade, ratings, and sample reviews.
    """

    # Normalize course codes
    grades_df['course_code'] = grades_df['course_code'].astype(
        str).str.replace(" ", "").str.upper()
    reviews_df['interaction_type'] = reviews_df['interaction_type'].astype(
        str).str.replace(" ", "").str.upper()
    code = course_code.replace(" ", "").upper()

    # Filter data
    course_grades = grades_df[grades_df['course_code'] == code]
    course_reviews = reviews_df[reviews_df['interaction_type'] == code]

    if course_grades.empty and course_reviews.empty:
        return {"error": "No data"}

    # Pick first available entry (latest or representative)
    top_row = course_grades.head(1).to_dict(orient="records")[
        0] if not course_grades.empty else {}

    # Average grade (use course_stats.csv if available)
    avg_grade = "N/A"
    if 'avg_grade' in course_grades.columns and not course_grades.empty:
        try:
            avg_grade = round(
                course_grades['avg_grade'].astype(float).mean(), 2)
        except:
            pass
    else:
        # Try loading from preprocessed stats file if missing
        stats_path = os.path.join(DATA_DIR, "course_stats.csv")
        if os.path.exists(stats_path):
            stats_df = pd.read_csv(stats_path)
            stats_df['course_code'] = stats_df['course_code'].astype(
                str).str.replace(" ", "").str.upper()
            row = stats_df[stats_df['course_code'] == code]
            if not row.empty:
                avg_grade = round(float(row['avg_grade'].iloc[0]), 2)

    # Average rating from reviews
    avg_star = "N/A"
    if 'overall_star' in course_reviews.columns and not course_reviews.empty:
        try:
            avg_star = round(
                course_reviews['overall_star'].astype(float).mean(), 2)
        except:
            pass

    # Collect sample reviews
    reviews_texts = []
    if 'review' in course_reviews.columns:
        reviews_texts = [
            r for r in course_reviews['review'].dropna().tolist() if len(r.strip()) > 0]

    # Prepare structured summary
    return {
        "course": f"{top_row.get('course_code', code)} - {top_row.get('course_title', 'N/A')}",
        "prof": top_row.get('prof_name', 'Unknown'),
        "semester": f"{top_row.get('semester', 'N/A')} ({top_row.get('year', 'N/A')})",
        "avg_grade": avg_grade,
        "avg_star": avg_star,
        "reviews": reviews_texts[:5],
    }


def get_course_info(course_code):
    """
    Streamlit-compatible handler that:
    - Loads course data
    - Summarizes grades & reviews
    - Displays text summary + grade distribution plot
    """
    grades_csv = os.path.join(DATA_DIR, "course_grades_rows.csv")
    reviews_csv = os.path.join(DATA_DIR, "Reviews_rows.csv")

    # Safety check
    if not (os.path.exists(grades_csv) and os.path.exists(reviews_csv)):
        return "⚠️ Data files missing. Please check that 'data/course_grades_rows.csv' and 'data/Reviews_rows.csv' exist."

    # Load both data sources
    grades_df, reviews_df = load_data(grades_csv, reviews_csv)
    summary = compute_course_summary(course_code, grades_df, reviews_df)

    # Handle missing data
    if "error" in summary:
        return f"⚠️ No data found for {course_code}. Please check the code."

    # Display course info
    st.markdown(f"""
### 📘 {summary['course']}
🧑‍🏫 **Instructor:** {summary['prof']}
🗓️ **Semester:** {summary['semester']}
🎓 **Average Grade (AGP):** {summary['avg_grade']}
⭐ **Average Rating:** {summary['avg_star']}/5

#### 💬 Top Reviews:
""")

    if summary['reviews']:
        for r in summary['reviews']:
            st.markdown(f"- {r}")
    else:
        st.info("No reviews available for this course.")

    # Display grade distribution chart (if available)
    image_b64 = plot_grade_distribution(course_code, grades_df)
    if image_b64:
        st.image(f"data:image/png;base64,{image_b64}",
                 caption=f"Grade Distribution for {course_code}",
                 use_column_width=True)

    return ""  # Everything rendered in Streamlit
