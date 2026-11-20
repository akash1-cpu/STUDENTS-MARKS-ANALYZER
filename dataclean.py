import numpy as np
import pandas as pd

# Import uncleaned student examination dataset
marks_data = pd.read_csv('student.csv')
print("=== Uncleaned Input Dataset (First 10 Rows) ===")
print(marks_data.head(10))

# Drop duplicate Student_ID entries (retaining the first authentic student record)
marks_data = marks_data.drop_duplicates(subset=['Student_ID'], keep='first').copy()
print("\n=== Dataset After Removing Duplicate Student_IDs ===")
print(marks_data.head(10))

# Standardize Gender entries to 'M' and 'F'
marks_data['Gender'] = marks_data['Gender'].astype(str).str.strip().str.upper()
gender_mapping = {
    'FEMALE': 'F',
    'F': 'F',
    'MALE': 'M',
    'M': 'M'
}
marks_data['Gender'] = marks_data['Gender'].map(gender_mapping).fillna('Unknown')
print("\n=== Dataset After Standardizing Gender Labels ===")
print(marks_data.head(10))

# Map all Semester entries to first-year cohort ('1')
marks_data['Semester'] = marks_data['Semester'].astype(str).str.strip().str.upper()
semester_mapping = {
    '4': '1',
    '3': '1',
    '2': '1',
    '1': '1'
}
marks_data['Semester'] = marks_data['Semester'].map(semester_mapping).fillna('Unknown')

# Slice Student_ID numeric suffix, sort ascending, and restore admission prefix
marks_data['Student_ID'] = marks_data['Student_ID'].str[5:]
marks_data['Student_ID'] = pd.to_numeric(marks_data['Student_ID'])
marks_data = marks_data.sort_values(by='Student_ID')
marks_data['Student_ID'] = '2026B0' + marks_data['Student_ID'].astype(str)

# Replace 'Absent' text flags in Electrical_Marks with NaN
marks_data['Electrical_Marks'] = marks_data['Electrical_Marks'].astype(str).str.strip()
marks_data.loc[marks_data['Electrical_Marks'] == 'Absent', 'Electrical_Marks'] = np.nan

# Convert attendance and subject mark fields to numeric float values
float_columns = [
    'Attendance_Pct',
    'Maths_Marks',
    'Physics_Marks',
    'Programming_Marks',
    'Electrical_Marks',
    'AI_Marks'
]
for field in float_columns:
    marks_data[field] = pd.to_numeric(marks_data[field], errors='coerce')

# Clamp Attendance_Pct to [0, 100] and impute out-of-bounds/missing values using median
marks_data.loc[(marks_data['Attendance_Pct'] < 0) | (marks_data['Attendance_Pct'] > 100), 'Attendance_Pct'] = np.nan
marks_data['Attendance_Pct'] = marks_data['Attendance_Pct'].fillna(marks_data['Attendance_Pct'].median())

# Clamp all subject scores to [0, 100] and fill invalid/missing values with 0
course_columns = [
    'Maths_Marks',
    'Physics_Marks',
    'Programming_Marks',
    'Electrical_Marks',
    'AI_Marks'
]
for course in course_columns:
    marks_data.loc[(marks_data[course] < 0) | (marks_data[course] > 100), course] = np.nan
    marks_data[course] = marks_data[course].fillna(0)

print("\n=== Final Cleaned Dataset (First 10 Rows) ===")
print(marks_data.head(10))

# Write sanitized dataset to CSV
marks_data.to_csv('cleaned_student.csv', index=False)
print("\nProcessing finished: saved to 'cleaned_student.csv'.")