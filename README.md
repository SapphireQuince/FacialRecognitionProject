# Facial Recognition Project

A simple yet effective **facial recognition** system built for **local use on Windows**.  
It uses **dlib** for facial landmark detection and a **K-Nearest Neighbors (KNN)** classifier to identify faces — no mobile app, no complex setup, just Python scripts you can run locally.

---

## 📂 Project Structure

| File / Folder                              | Description |
|--------------------------------------------|-------------|
| `facial_landmark_extraction.py`            | Extracts 68 facial landmarks from each face image and saves them to a CSV file. |
| `get_landmarks.py`                         | Contains a function to extract landmarks from any given image. |
| `train_knn.py`                             | Labels landmark data with person IDs, trains a KNN model, and saves it. |
| `predict_face.py`                          | Loads the trained model and predicts the person in a new image. |
| `models/`                                  | Contains the trained KNN model file. |
| `shape_predictor_68_face_landmarks.dat`    | Pre-trained dlib model for landmark detection. |
| `landmarks.csv / landmarks_labeled.csv`    | Raw and labeled facial landmark data. |
| `How-to-Add-More-Images-in-the-model.docx` | Step-by-step guide for updating the model with new images. |

---

## 🛠 Setup Instructions

### 1. Install Requirements
Make sure you have **Python 3.7+** installed, then install dependencies:

pip install tensorflow opencv-python dlib numpy scikit-learn matplotlib joblib
text

### 2. Download Landmark Model
Download `shape_predictor_68_face_landmarks.dat` from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the **project root directory**.

---

## 🚀 Running the Project

### Step 1 — Add Training Images
Add your training images to the `images/` directory.  
Example naming: `1.jpg`, `2.jpg`, ..., `25.jpg`.

### Step 2 — Extract Facial Landmarks
Run:
python facial_landmark_extraction.py
text
This will generate `landmarks.csv`.

### Step 3 — Label Data & Train Model
Run:
python train_knn.py
text
This creates:
- `landmarks_labeled.csv` (labeled dataset)
- `models/knn_face_model.pkl` (trained model)

### Step 4 — Test Face Recognition
Place your test image in `images/` (e.g., `test_image.jpg`) and run:
python predict_face.py
text
The script will output the **predicted person** and **confidence score**.

---

## ➕ Adding More Images & Retraining
For a full guide, see the included `How-to-Add-More-Images-in-the-model.docx`.  
Quick summary:

1. Add new images to `images/`.
2. Run:
python facial_landmark_extraction.py
text
3. Update the `names` list in `train_knn.py` to match the new image count.
4. Retrain:
python train_knn.py
text

---

## 💡 Tips
- Use clear, front-facing images for best results.
- Adjust the confidence threshold in `predict_face.py` if you get too many "No reliable match" results.
- Consider adding variations of the same person (different lighting, angles) for improved accuracy.
- For advanced improvements, integrate deep learning models (CNNs) or add a graphical interface.

---

## 🤝 Contributing
Pull requests, feature suggestions, and improvements are welcome!  
Feel free to **fork** the repo and submit your ideas.

---

**Project developed by [Animesh Aundhekar (SapphireQuince)](https://github.com/SapphireQuince)**
