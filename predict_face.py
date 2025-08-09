import joblib
from get_landmarks import get_landmarks

# Load trained KNN model
knn = joblib.load(r"D:\FacialRecognition Project\models\knn_face_model.pkl")

# Specify test image path
image_path = r"D:\FacialRecognition Project\images\test_image.jpg"  # Ensure this image exists

# Extract landmarks from test image
landmarks = get_landmarks(image_path)

if landmarks is not None:
    landmarks = landmarks.reshape(1, -1)
    prediction = knn.predict(landmarks)
    probabilities = knn.predict_proba(landmarks)
    confidence = probabilities.max() * 100
    if confidence > 30:  # Lower threshold for testing
        print(f"Match found: {prediction[0]} ({confidence:.2f}%)")
    else:
        print("No reliable match found.")
else:
    print("No face detected in the test image.")
