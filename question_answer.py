import os
import pickle
import pandas as pd

# 1. Load DataFrame
with open("data.pkl", "rb") as f:
    df = pickle.load(f)

# 2. Drop duplicates based on review text
df = df.drop_duplicates(subset=["reviews"]).reset_index(drop=True)

# 3. Base folder to store reviews
base_dir = "reviews"
os.makedirs(base_dir, exist_ok=True)

# 4. Loop through institutions and their courses
for institution in sorted(df["institution"].dropna().unique()):
    sanitized_institution = institution.replace("/", "_").replace("\\", "_").strip()
    inst_dir = os.path.join(base_dir, sanitized_institution)
    os.makedirs(inst_dir, exist_ok=True)

    inst_df = df[df["institution"] == institution]
    courses = sorted(inst_df["name"].dropna().unique())

    for course in courses:
        sanitized_course = course.replace("/", "_").replace("\\", "_").strip()
        course_file = os.path.join(inst_dir, f"{sanitized_course}.txt")

        course_df = inst_df[inst_df["name"] == course]

        with open(course_file, "w", encoding="utf-8") as f:
            for review in course_df["reviews"]:
                f.write(f"{review.strip()}\n")

print("\n✅ All reviews have been exported to the 'reviews' folder, one per line per file.")
