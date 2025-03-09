import cv2
import streamlit as st
from deepface import DeepFace
import numpy as np

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit UI
st.title("Real-Time Face Expression Detection")

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()  # Placeholder for displaying frames
stop_button = st.button("Stop Webcam")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region for analysis
        face_img = frame[y:y + h, x:x + w]

        # Analyze expression using DeepFace
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            expression = analysis[0]['dominant_emotion']

            # Display detected emotion
            cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            st.warning(f"DeepFace Error: {e}")

    # Convert the frame to RGB (Streamlit requires RGB format)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Stop the webcam if the button is pressed
    if stop_button:
        break

# Release the webcam
cap.release()
st.success("Webcam stopped successfully!")
