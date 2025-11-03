# src/preprocess.py
import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, MODELS_DIR, REVIEWS_PICKLE

# ------------------------------------------------------------
# 🎓 GRADE POINTS MAPPING
# ------------------------------------------------------------
GRADE_POINTS = {
    "A*": 10, "A": 10, "B+": 9, "B": 8,
    "C+": 7, "C": 6, "D+": 5, "D": 4,
    "E": 2, "F": 0
}

# ------------------------------------------------------------
# 📂 LOAD DATA
# ------------------------------------------------------------
def load_data(grades_csv=None, reviews_csv=None):
    if grades_csv is None:
        grades_csv = os.path.join(DATA_DIR, "course_grades_rows.csv")
    if reviews_csv is None:
        reviews_csv = os.path.join(DATA_DIR, "Reviews_rows.csv")

    grades = pd.read_csv(grades_csv, dtype=str).fillna("")
    reviews = pd.read_csv(reviews_csv, dtype=str).fillna("")

    # Normalize column names
    grades.columns = [c.strip() for c in grades.columns]
    reviews.columns = [c.strip() for c in reviews.columns]

    return grades, reviews

# ------------------------------------------------------------
# 🧮 COMPUTE COURSE STATISTICS (AVG GRADES)
# ------------------------------------------------------------
def compute_course_stats(grades_df):
    def avg_grade_for_row(row):
        tot, cnt = 0.0, 0.0
        for g, pts in GRADE_POINTS.items():
            if g in row and row[g] != "":
                try:
                    n = float(row[g])
                except ValueError:
                    n = 0.0
                tot += pts * n
                cnt += n
        return round(tot / cnt, 2) if cnt > 0 else None

    df = grades_df.copy()
    df['avg_grade'] = df.apply(avg_grade_for_row, axis=1)

    # Aggregate across same course/prof
    agg = df.groupby(['course_code', 'prof_name'], dropna=False).agg(
        avg_grade=('avg_grade', 'mean'),
        samples=('id', 'count')
    ).reset_index()

    return agg

# ------------------------------------------------------------
# 🧹 CLEAN REVIEWS
# ------------------------------------------------------------
def prepare_reviews(reviews_df):
    df = reviews_df.copy()

    # Normalize course code
    if 'gradeortopic' in df.columns:
        df['course_code'] = df['gradeortopic'].str.strip().str.upper()
    else:
        df['course_code'] = ""

    # Normalize professor field
    if 'prof_email' in df.columns:
        df['prof_name'] = df['prof_email'].astype(str)
    elif 'prof_name' not in df.columns:
        df['prof_name'] = ""

    # Keep relevant columns
    cols = ['id', 'created_at', 'prof_name', 'username', 'review',
            'overall_star', 'course_code', 'year', 'upvotes']
    df = df[[c for c in cols if c in df.columns]].fillna("")
    return df

# ------------------------------------------------------------
# 📊 PLOT GRADE DISTRIBUTION (BAR + PIE)
# ------------------------------------------------------------
def plot_grade_distribution(course_code, grades_df):
    df = grades_df.copy()
    df['course_code'] = df['course_code'].str.upper()

    course_data = df[df['course_code'] == course_code.upper()]
    if course_data.empty:
        return None

    # Define grade order
    grade_columns = ['A*', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'E', 'F', 'S', 'S^', 'X']
    grade_counts = {}
    for g in grade_columns:
        if g in course_data.columns:
            try:
                grade_counts[g] = course_data[g].astype(float).sum()
            except:
                grade_counts[g] = 0

    grade_counts = {g: v for g, v in grade_counts.items() if v > 0}
    if not grade_counts:
        return None

    plot_df = pd.DataFrame({
        'Grade': list(grade_counts.keys()),
        'Count': list(grade_counts.values())
    })

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar plot
    sns.barplot(x='Grade', y='Count', data=plot_df, ax=axes[0], palette='Spectral')
    for i, val in enumerate(plot_df['Count']):
        axes[0].text(i, val + 0.3, f"{int(val)}", ha='center', va='bottom', fontsize=8)
    axes[0].set_title(f"Grade Distribution: {course_code}")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("No. of Students")

    # Pie chart
    axes[1].pie(
        plot_df['Count'],
        labels=plot_df['Grade'],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('Spectral', len(plot_df))
    )
    axes[1].set_title("Percentage Share")

    plt.tight_layout()

    # Convert to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return image_b64

# ------------------------------------------------------------
# 🚀 CLI ENTRY POINT (OPTIONAL)
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", required=True)
    parser.add_argument("--reviews", required=True)
    args = parser.parse_args()

    grades_df, reviews_df = load_data(args.grades, args.reviews)
    stats = compute_course_stats(grades_df)
    reviews_clean = prepare_reviews(reviews_df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    reviews_clean.to_pickle(REVIEWS_PICKLE)
    stats.to_csv(os.path.join(MODELS_DIR, "course_stats.csv"), index=False)

    print("✅ Preprocessing complete — saved cleaned reviews & stats.")
