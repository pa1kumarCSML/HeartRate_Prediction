import cv2
import dlib

# Load DLIB's pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")

# Open the video file
video_capture = cv2.VideoCapture('videos/TCS.mp4')

# Process the video frame by frame
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)

        # Iterate over the 68 facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle at each facial landmark
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the frame with facial landmarks
    cv2.imshow('Facial Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()