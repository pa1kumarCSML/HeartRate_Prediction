import cv2
import dlib

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")

# Open video file
cap = cv2.VideoCapture('videos/real/brad.mp4')

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Process each detected face
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        
        # Identify forehead region based on landmarks
        forehead_x1 = min(landmarks.part(17).x, landmarks.part(18).x)  # Left eyebrow
        forehead_x2 = max(landmarks.part(26).x, landmarks.part(25).x)  # Right eyebrow
        forehead_y1 = min(landmarks.part(19).y, landmarks.part(24).y)  # Hairline
        forehead_y2 = max(landmarks.part(19).y, landmarks.part(24).y)  # Baseline
        
        # Increase breadth of the rectangle by a factor (e.g., 1.5 times)
        # You can adjust this factor as needed
        width_factor = 1.5
        width_increase = int((forehead_x2 - forehead_x1) * (width_factor - 1) / 2)
        forehead_x1 -= width_increase
        forehead_x2 += width_increase
        
        # Draw rectangle with increased breadth
        cv2.rectangle(frame, (forehead_x1, forehead_y1), (forehead_x2, forehead_y2), (0, 255, 0), 2)
        
        # Draw facial landmarks with numbers
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # Display frame with highlighted forehead region and facial landmarks
    cv2.imshow('Forehead Detection with Landmarks', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
