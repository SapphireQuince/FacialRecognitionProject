import pandas as pd

df = pd.read_csv(r"D:\FacialRecognition Project\landmarks.csv", header=0)

# Keep only the first 25 rows (your training images)
df = df.iloc[:25]

df.to_csv(r"D:\FacialRecognition Project\landmarks.csv", index=False)

print("Trimmed landmarks.csv to 25 rows")
