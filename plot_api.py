from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import traceback
import os

app = FastAPI()

@app.get("/plot/{course_code}")
def get_plot(course_code: str):
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "course_grades_row.csv")
        csv_path = os.path.abspath(csv_path)
        print(f"📂 Trying to load: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV with shape: {df.shape}")

        if "course_code" not in df.columns:
            raise KeyError(f"'course_code' column not found. Columns: {list(df.columns)}")

        course_df = df[df["course_code"].astype(str).str.strip().str.lower() == course_code.strip().lower()]
        if course_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {course_code}")

        grade_cols = ['A', 'A*', 'B', 'B+', 'C', 'C+', 'D', 'D+', 'E', 'F', 'S', 'S^', 'X']
        missing = [c for c in grade_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing grade columns: {missing}")

        grade_sums = course_df[grade_cols].sum(numeric_only=True)
        print("📊 Grade distribution:\n", grade_sums)

        fig, ax = plt.subplots(figsize=(8, 4))
        grade_sums.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f"Grade Distribution for {course_code}")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Number of Students")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        print("✅ Successfully generated plot.")
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("❌ ERROR OCCURRED:")
        traceback.print_exc()  # shows exact error trace in terminal
        raise HTTPException(status_code=500, detail=str(e))
