import cv2
import dlib

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")

# Open video file
video_path= 'videos/real/jenny.mp4'
cap = cv2.VideoCapture(video_path)

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
        forehead_x1 = landmarks.part(19).x # Left eyebrow-x
        forehead_x2 = landmarks.part(24).x  # Right eyebrow-x
        forehead_y1 = landmarks.part(19).y # Left eyebrow-y
        forehead_y2 = landmarks.part(24).y # Right eyebrow-y
        
       
        height=int((forehead_x2 - forehead_x1)*0.7)
        # Draw rectangle with increased breadth
        cv2.rectangle(frame, (forehead_x1, forehead_y1-height), (forehead_x2, forehead_y2), (0, 255, 0), 2)
        
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