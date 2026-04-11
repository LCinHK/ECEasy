import pandas as pd
from pathlib import Path

def generate_ustspace_reviews(csv_path: str = "ustspace_reviews.csv"):
    output_dir = Path("ECEknowledge/ustspace_reviews")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        course_code = str(row["course_code"]).strip().upper()
        filename = f"ustspace_review_{course_code}.txt"
        filepath = output_dir / filename

        content = f"""# UST.space Course Review - {course_code} ({row['semester']})

**Course Code:** {course_code}
**Course Title:** {row.get('course_title', 'N/A')}
**Instructor:** {row['instructor']}
**Semester:** {row['semester']}

## Overall Ratings
- **CONTENT:**   {row['content_rating']}
- **TEACHING:**  {row['teaching_rating']}
- **GRADING:**   {row['grading_rating']}
- **WORKLOAD:**  {row['workload_rating']}

## Assessment Components
{row.get('assessments', 'N/A')}

## CONTENT
{row.get('content_text', 'N/A')}

## TEACHING
{row.get('teaching_text', 'N/A')}

## GRADING
{row.get('grading_text', 'N/A')}

## WORKLOAD
{row.get('workload_text', 'N/A')}

**Source:** UST.space student reviews (anonymous)
**Last Updated:** {row.get('last_updated', 'April 2026')}
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

        print(f"✓ Created: {filename}")

    print(f"\n🎉 Done! Generated {len(df)} review files.")
    print("Now run: python ingest_FAISS.py")

if __name__ == "__main__":
    generate_ustspace_reviews()