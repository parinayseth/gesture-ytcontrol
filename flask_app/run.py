import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

# Initialize HandDetector and Classifier objects
detector = HandDetector(maxHands=1)
classifier = Classifier("E:\Ak-gesture\sampleapp\gesture_based_youtube_control\model\keras_model.h5", "E:\Ak-gesture\sampleapp\gesture_based_youtube_control\model\labels.txt")
# Constants
offset = 20
imgSize = 300
labels = ["Play", "Pause", "ytvolup", "ytvoldown", "forward", "backword"]

# Function to perform predictions and overlay on image
def predict_and_overlay(img):
    # Perform hand detection
    hands, _ = detector.findHands(img)  
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        # Extract hand region
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Prepare image for prediction
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
        
        # Perform prediction on the prepared image
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
        # Overlay predictions on the image
        cv2.rectangle(img, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
    
    return img

# Streamlit app code
def main():
    st.title("Hand Gesture Recognition")

    # Create a video capture object
    cap = cv2.VideoCapture(0)

    # Infinite loop to continuously read frames and display them
    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            st.error("Failed to capture frame from webcam")
            break

        # Perform prediction and overlay on the frame
        frame_with_predictions = predict_and_overlay(frame)

        # Display the frame with predictions
        st.image(frame_with_predictions, channels="BGR", use_column_width=True)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    main()
