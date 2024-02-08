import cv2
import dlib

# Load face detection and facial landmark models
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_predictor = dlib.shape_predictor('dlib_files/shape_predictor_68_face_landmarks.dat')

# Load the image
image = cv2.imread('images/jeo.jpg')

# Convert the image to grayscale for face detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # Detect facial landmarks
    landmarks = landmark_predictor(gray_image, dlib.rectangle(x, y, x+w, y+h))
    
    # Define regions of interest
    right_cheek = image[y:y+h, x+w//2:x+w, :]
    left_cheek = image[y:y+h, x:x+w//2, :]
    forehead = image[y:y+h//2, x:x+w, :]
    neck = image[y+h//2:y+h, x:x+w, :]

    # Save the extracted regions
    cv2.imwrite('right_cheek.jpg', right_cheek)
    cv2.imwrite('left_cheek.jpg', left_cheek)
    cv2.imwrite('forehead.jpg', forehead)
    cv2.imwrite('neck.jpg', neck)
