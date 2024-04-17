import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")

# Load the face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels for emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Skip frames to reduce computational load
frame_skip = 5
frame_count = 0

# Main loop for real-time emotion detection
while True:
    # Read a frame from the webcam
    i, im = webcam.read()

    # Increment frame count and skip frames if necessary
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (p, q, r, s) in faces:
        # Extract the face region
        face_image = gray[q:q+s, p:p+r]

        # Resize the face image
        face_image = cv2.resize(face_image, (48, 48))

        # Preprocess the face image
        img = extract_features(face_image)

        # Predict emotion using the pre-trained model
        pred = model.predict(img)

        # Check if the highest predicted emotion probability is above 75%
        if np.max(pred) > 0.75:
            prediction_label = labels[pred.argmax()]

            # Draw rectangle around the face
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

            # Annotate the frame with predicted emotion
            cv2.putText(im, '%s' % prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    # Display the annotated frame
    cv2.imshow("Output", im)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
