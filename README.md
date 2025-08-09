# Facial Recognition Project

This project is a simple yet effective facial recognition system built for local use on Windows 11. It uses **dlib** to detect facial landmarks and a **K-Nearest Neighbors (KNN)** classifier to identify faces. There's no mobile app involved — just straightforward, easy-to-run Python scripts.

## What's Inside the Project

- **facial_landmark_extraction.py**: Extracts 68 key points (landmarks) from face images and saves them to a CSV file.
- **get_landmarks.py**: Contains a function to extract landmarks from any given image.
- **train_knn.py**: Takes the landmark data, labels it with person IDs, trains a KNN model, and saves it for later use.
- **predict_face.py**: Loads the trained model to test face recognition on new images.
- **models/**: Folder where the trained KNN model is saved.
- **shape_predictor_68_face_landmarks.dat**: A pre-trained model file for facial landmark detection.
- **landmarks.csv / landmarks_labeled.csv**: CSV files holding raw and labeled landmark data respectively.
- **How-to-Add-More-Images-in-the-model.docx**: Instructions on how to update your model with new images over time.

***

## How to Use This Project

### Get Your Environment Ready

- Make sure you have Python 3.7 or higher installed.
- Install all required dependencies by running:

  ```
  pip install tensorflow opencv-python dlib numpy scikit-learn matplotlib joblib
  ```

- Download the facial landmark predictor file (`shape_predictor_68_face_landmarks.dat`) from the official [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Put this file in your project’s main folder.

***

### Running the Project — Step by Step

1. **Add your training images:**
   Add or edit the files names and paths inside the python files according to your directory creation. 
   Place your images into the `images/` folder. Make sure they are named simply as `1.jpg`, `2.jpg`, ..., up to `25.jpg`.

3. **Extract facial landmarks:**  
   Run the landmark extraction script — this will process the images and save the landmark data in `landmarks.csv`.

   ```
   python facial_landmark_extraction.py
   ```

4. **Label the data and train the model:**  
   Run the training script to label each image with an ID and train the KNN classifier.

   ```
   python train_knn.py
   ```

5. **Test the face recognition:**  
   Put a test image inside the `images/` folder, for example `test_image.jpg`, and run the prediction script to see if the system recognizes it.

   ```
   python predict_face.py
   ```

***

### Adding More Images and Retraining

For details on expanding your dataset and retraining your model, check out the included `How-to-Add-More-Images-in-the-model.docx`. Here's a quick overview:

- Add any new images to your `images/` folder.
- Run the landmark extraction again:

  ```
  python facial_landmark_extraction.py
  ```

- Update the `names` list in `train_knn.py` to include new labels matching the total number of images.
- Rerun the training script:

  ```
  python train_knn.py
  ```

Your model will update to recognize the new faces.

***

## A Few Tips

- Use clear, front-facing photos for the best recognition results.
- You can tweak the confidence threshold for predictions inside `predict_face.py` if needed.
- This project is great for learning and local use. If you want to expand, consider exploring deep learning techniques or creating a user interface.

***

Feel free to open issues, suggest improvements, or contribute new features!

***

*Project developed by Animesh Aundhekar (SapphireQuince)*
