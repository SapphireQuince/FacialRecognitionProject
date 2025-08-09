import dlib
import cv2
import numpy as np
import os
import csv

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this file is in your project directory or provide full path

# Function to extract 68 facial landmarks as a flat array from an image
def get_landmarks(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    return np.array(coords).reshape(-1)

# Folder containing images (use raw string or double backslashes for Windows paths)
image_folder = r'D:\FacialRecognition Project\images'

# Output CSV file to save landmarks
output_csv = 'landmarks.csv'

# Process all images in the folder and save results to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header with landmark point labels
    header = []
    for i in range(68):
        header.append(f'x{i+1}')
        header.append(f'y{i+1}')
    writer.writerow(header)
    
    # Loop over images in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        landmarks = get_landmarks(image_path)
        if landmarks is not None:
            writer.writerow(landmarks)
        else:
            print(f"No face detected in image {image_name}.")
