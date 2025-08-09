import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Path to predictor .dat file

def get_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None
    landmarks = predictor(gray, faces[0])
    coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    return np.array(coords).reshape(-1)
