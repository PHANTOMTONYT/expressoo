import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# Set up the Streamlit page
st.set_page_config(page_title="Expression Detection", layout="wide")
st.title("Facial Expression Detection App")

# Initialize face detection models
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
        st.error("Error: Could not load cascade classifiers")
        return None, None, None
        
    return face_cascade, eye_cascade, smile_cascade

face_cascade, eye_cascade, smile_cascade = load_models()

def get_face_landmarks(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    landmarks_list = []
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        # Detect smile
        smile = smile_cascade.detectMultiScale(face_roi, 1.7, 20)
        
        landmarks_list.append({
            'face': (x, y, w, h),
            'eyes': eyes,
            'smile': smile
        })
    
    return landmarks_list if landmarks_list else None

def analyze_expression(landmarks):
    if landmarks is None:
        return "No face detected"
    
    eyes = landmarks['eyes']
    smile = landmarks['smile']
    
    # Basic emotion analysis
    if len(smile) > 0:
        if len(eyes) >= 2:
            return "Happy"
        return "Smiling"
    elif len(eyes) >= 2:
        return "Neutral"
    elif len(eyes) == 1:
        return "Winking"
    else:
        return "Unknown"

def process_image(image):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Get facial landmarks
    landmarks_list = get_face_landmarks(img_array)
    
    # Create a copy for drawing
    img_with_detections = img_array.copy()
    
    # Face counter and statistics
    face_count = 0
    emotion_stats = {"Happy": 0, "Smiling": 0, "Neutral": 0, "Winking": 0, "Unknown": 0}
    
    if landmarks_list:
        for landmarks in landmarks_list:
            face_count += 1
            x, y, w, h = landmarks['face']
            
            # Draw face rectangle
            cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw eyes
            for (ex, ey, ew, eh) in landmarks['eyes']:
                cv2.rectangle(img_with_detections, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                
            # Draw smile
            for (sx, sy, sw, sh) in landmarks['smile']:
                cv2.rectangle(img_with_detections, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
            
            # Get and display emotion
            emotion = analyze_expression(landmarks)
            cv2.putText(img_with_detections, f"Expression: {emotion}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Update emotion statistics
            if emotion in emotion_stats:
                emotion_stats[emotion] += 1
    
    return img_with_detections, face_count, emotion_stats

# Create sidebar for options
st.sidebar.title("Options")
detection_mode = st.sidebar.radio("Select Mode", ["Upload Image", "Webcam"])

if detection_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process the image and display results
        with col2:
            st.subheader("Processed Image")
            processed_img, face_count, emotion_stats = process_image(image)
            st.image(processed_img, use_column_width=True)
        
        # Display statistics
        st.subheader("Detection Results")
        st.write(f"Faces detected: {face_count}")
        
        if face_count > 0:
            # Create columns for each emotion
            cols = st.columns(5)
            for i, (emotion, count) in enumerate(emotion_stats.items()):
                cols[i].metric(emotion, count)

elif detection_mode == "Webcam":
    # Initialize webcam session state
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    # Webcam control buttons
    col1, col2 = st.sidebar.columns(2)
    start_button = col1.button("Start Webcam")
    stop_button = col2.button("Stop Webcam")
    
    if start_button:
        st.session_state.webcam_running = True
    
    if stop_button:
        st.session_state.webcam_running = False
    
    # Create placeholders for webcam feed and stats
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if st.session_state.webcam_running:
        try:
            # Create a video capture object
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please make sure your webcam is connected and not being used by another application.")
            else:
                st.sidebar.success("Webcam is running! Press 'Stop Webcam' to stop.")
                
                while st.session_state.webcam_running:
                    # Read frame from webcam
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture image from webcam.")
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # Process the frame
                    processed_img, face_count, emotion_stats = process_image(image)
                    
                    # Display the processed frame
                    frame_placeholder.image(processed_img, caption="Webcam Feed", use_column_width=True)
                    
                    # Display statistics
                    with stats_placeholder.container():
                        st.write(f"Faces detected: {face_count}")
                        if face_count > 0:
                            # Create columns for each emotion
                            cols = st.columns(5)
                            for i, (emotion, count) in enumerate(emotion_stats.items()):
                                cols[i].metric(emotion, count)
                    
                    # Add a small sleep to reduce CPU usage
                    time.sleep(0.1)
                    
                    # Check if the webcam should still be running
                    if not st.session_state.webcam_running:
                        break
                
                # Release the webcam
                cap.release()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.webcam_running = False
    else:
        st.info("Click 'Start Webcam' to begin face detection.")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app detects facial expressions using OpenCV and Streamlit.
    
    Features:
    - Face detection
    - Eye detection
    - Smile detection
    - Expression classification:
      - Happy
      - Smiling
      - Neutral
      - Winking
      - Unknown
    
    Detection colors:
    - Blue: Face
    - Green: Eyes
    - Red: Smile
    """
)

# Display instructions
if detection_mode == "Upload Image" and not uploaded_file:
    st.info("Please upload an image using the sidebar to begin face detection.")
