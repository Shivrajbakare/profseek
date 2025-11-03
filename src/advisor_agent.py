# src/advisor_agent.py
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from transformers import pipeline

# === 1️⃣ Load data safely ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_STATS = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "course_stats.csv"))
DATA_PATH_GRADES = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "course_grades_rows.csv"))

print(f"🔍 Loading course stats from: {DATA_PATH_STATS}")
print(f"🔍 Loading course grades from: {DATA_PATH_GRADES}")

if not os.path.exists(DATA_PATH_STATS):
    raise FileNotFoundError(f"course_stats.csv not found at {DATA_PATH_STATS}")
if not os.path.exists(DATA_PATH_GRADES):
    raise FileNotFoundError(f"course_grades_row.csv not found at {DATA_PATH_GRADES}")

course_stats = pd.read_csv(DATA_PATH_STATS)
course_grades = pd.read_csv(DATA_PATH_GRADES)

# Normalize columns
course_stats.columns = course_stats.columns.str.strip().str.lower()
course_grades.columns = course_grades.columns.str.strip().str.lower()

course_stats.rename(
    columns={
        'course_code': 'course',
        'prof_name': 'professor',
        'avg_grade': 'average_grade',
        'samples': 'num_samples'
    },
    inplace=True
)

course_stats['average_grade'] = pd.to_numeric(course_stats['average_grade'], errors='coerce')

# === 2️⃣ Load NLP model for intent detection ===
intent_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === 3️⃣ Utility: find course info ===
def find_course(course_code):
    mask = course_stats['course'].astype(str).str.contains(course_code, case=False, na=False)
    return course_stats[mask]

def find_grade_distribution(course_code):
    mask = course_grades['course_code'].astype(str).str.contains(course_code, case=False, na=False)
    return course_grades[mask]

# === 4️⃣ Create bar graph for grade distribution ===
def generate_grade_bar_chart(course_code, df):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from io import BytesIO
    import base64

    grade_cols = ['a', 'a*', 'b', 'b+', 'c', 'c+', 'd', 'd+', 'e', 'f', 's', 's^', 'x']
    grade_counts = df[grade_cols].sum(numeric_only=True)
    grade_counts = grade_counts[grade_counts > 0]

    if grade_counts.empty:
        return None

    # --- Dark background style ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = cm.Pastel1.colors[:len(grade_counts)]

    # --- Bar chart (left) ---
    axes[0].bar(grade_counts.index.str.upper(), grade_counts.values, color=colors, edgecolor='white')
    axes[0].set_title(f"Grade Distribution: {course_code}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("No. of Students")
    axes[0].grid(False)
    # Annotate counts on bars
    for i, v in enumerate(grade_counts.values):
        axes[0].text(i, v + 0.5, str(int(v)), ha='center', fontsize=9)

    # --- Pie chart (right) ---
    axes[1].pie(
        grade_counts.values,
        labels=grade_counts.index.str.upper(),
        autopct='%1.1f%%',
        colors=colors,
        textprops={'color': 'white', 'fontsize': 8}
    )
    axes[1].set_title("Percentage Share", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# === 5️⃣ Advisor agent core ===
def advisor_agent(query: str):
    try:
        query = query.strip().lower()

        # --- 1️⃣ Detect course-specific query ---
        course_code_match = re.findall(r"[A-Za-z]{2,3}\d{3}[A-Za-z]?", query.upper())
        if course_code_match:
            code = course_code_match[0]
            data = find_course(code)
            if not data.empty:
                row = data.iloc[0]
                avg = row.get('average_grade', 'N/A')
                prof = row.get('professor', 'Unknown')
                n = row.get('num_samples', 'N/A')
                response = f"📘 *{code}*\n👨‍🏫 Professor: {prof}\n📊 Average Grade: {avg}\n🧮 Samples: {n}"

                # Add grade distribution
                grade_df = find_grade_distribution(code)
                if not grade_df.empty:
                    chart = generate_grade_bar_chart(code, grade_df)
                    if chart:
                        response += f"\n\n🧾 **Grade Distribution:**\n![Grade Distribution]({chart})"
                    else:
                        response += "\n\n(No detailed grade distribution available.)"
                else:
                    response += "\n\n(No detailed grade distribution available.)"

                return response

            return f"❌ No data found for course {code}."

        # --- 2️⃣ Use AI to detect intent ---
        intents = {
            "high_avg": "find courses with highest average grades",
            "easy_courses": "find easy scoring courses or lenient grading",
            "professors_high_grades": "find professors who give high grades",
            "ai_courses": "find AI or machine learning courses",
        }

        candidate_labels = list(intents.values())
        pred = intent_model(query, candidate_labels=candidate_labels)
        best_label = pred['labels'][0]
        best_intent = [k for k, v in intents.items() if v == best_label][0]

        # --- 3️⃣ Respond based on intent ---
        if best_intent == "high_avg":
            top = course_stats.sort_values(by='average_grade', ascending=False).head(10)
            res = "\n".join([f"{r.course} — {r.average_grade:.2f}" for r in top.itertuples()])
            return f"🏆 Top courses by average grade:\n{res}"

        elif best_intent == "easy_courses":
            easy = course_stats[course_stats['average_grade'] >= 8].sort_values(by='average_grade', ascending=False)
            if easy.empty:
                return "❌ No courses found with average grade ≥ 8."
            res = "\n".join([f"{r.course} — {r.average_grade:.2f}" for r in easy.head(10).itertuples()])
            return f"😌 Easiest scoring courses:\n{res}"

        elif best_intent == "professors_high_grades":
            if 'professor' in course_stats.columns:
                prof_avg = (
                    course_stats.groupby('professor')['average_grade']
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                res = "\n".join([f"{p}: {g:.2f}" for p, g in prof_avg.items()])
                return f"👨‍🏫 Professors giving highest average grades:\n{res}"
            else:
                return "❌ Professor data not available."

        elif best_intent == "ai_courses":
            mask = course_stats['course'].astype(str).str.contains("AI|ML|CS|EE", case=False, na=False)
            ai_courses = course_stats[mask].sort_values(by='average_grade', ascending=False).head(10)
            if ai_courses.empty:
                return "❌ No AI-related courses found."
            res = "\n".join([f"{r.course} — {r.average_grade:.2f}" for r in ai_courses.itertuples()])
            return f"🤖 AI/ML related courses:\n{res}"

        # --- 4️⃣ Fallback ---
        return "🤔 I couldn’t understand your query. Try:\n- 'average grade in AE201A'\n- 'professors who give high grades'\n- 'easy scoring AI courses'"

    except Exception as e:
        return f"⚠️ Error processing query: {str(e)}"

# === 6️⃣ For Streamlit / Local Testing ===
def get_course_advice(query: str):
    return advisor_agent(query)

if __name__ == "__main__":
    while True:
        q = input("🧑‍🎓 You: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("🎓 Advisor:", advisor_agent(q))
