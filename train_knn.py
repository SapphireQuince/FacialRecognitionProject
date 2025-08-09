import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Load the facial landmarks extracted CSV
df = pd.read_csv(r"D:\FacialRecognition Project\landmarks.csv", header=0)

# Create a list of 25 person labels as strings "Person1" to "Person25"
names = [f"Person{i}" for i in range(1, 26)]

# Check if the number of rows matches number of names
if len(df) != len(names):
    raise ValueError(f"Number of rows in CSV ({len(df)}) does not match number of labels ({len(names)}).")

# Add the 'name' column
df["name"] = names

# Move 'name' column to be the first column
cols = df.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df = df[cols]

# Save the labeled CSV
labeled_csv_path = r"D:\FacialRecognition Project\landmarks_labeled.csv"
df.to_csv(labeled_csv_path, index=False)
print(f"Labeled CSV saved to {labeled_csv_path}")

# Train the KNN classifier
data = pd.read_csv(labeled_csv_path)
X = data.iloc[:, 1:].values  # Landmark features
y = data['name'].values       # Labels

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save the trained model
model_dir = r"D:\FacialRecognition Project\models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "knn_face_model.pkl")
joblib.dump(knn, model_path)
print(f"Model trained and saved to {model_path}")
