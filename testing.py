import dlib
import cv2

# Load the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")

# Define the cheek landmark ranges (modify as needed for your requirements)
left_cheek_indices = [0, 4, 29, 8]
right_cheek_indices = [16, 12, 29, 8]


# Open the video capture
cap = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if frame is read correctly
  if not ret:
      print("Failed to capture frame!")
      break

  # Convert frame to grayscale (optional, but often helps)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Face detection
  faces = detector(gray)

  for face in faces:
    # Facial landmark detection
    landmarks = predictor(gray, face)

    # Get the left and right cheek landmark points
    left_cheek_points = [landmarks.part(i) for i in left_cheek_indices]
    right_cheek_points = [landmarks.part(i) for i in right_cheek_indices]

    left_cheek_x1 = min(p.x for p in left_cheek_points)
    left_cheek_y1 = min(p.y for p in left_cheek_points)
    left_cheek_x2 = max(p.x for p in left_cheek_points)
    left_cheek_y2 = max(p.y for p in left_cheek_points)


    if(left_cheek_x2-left_cheek_x1 > 25):
       diff=left_cheek_x2-left_cheek_x1
       left_cheek_x1+=diff//4
       left_cheek_x2-=diff//3

    if(left_cheek_y2-left_cheek_y1 > 25):
       diff=left_cheek_y2-left_cheek_y1
       left_cheek_y1+=diff//5
       left_cheek_y2-=diff//2

    right_cheek_x1 = min(p.x for p in right_cheek_points)
    right_cheek_y1 = min(p.y for p in right_cheek_points)
    right_cheek_x2 = max(p.x for p in right_cheek_points)
    right_cheek_y2 = max(p.y for p in right_cheek_points)

    if(right_cheek_x2-right_cheek_x1 > 25):
       diff=right_cheek_x2-right_cheek_x1
       right_cheek_x1+=diff//3
       right_cheek_x2-=diff//4

    if(right_cheek_y2-right_cheek_y1 > 25):
       diff=right_cheek_y2-right_cheek_y1
       right_cheek_y1+=diff//5
       right_cheek_y2-=diff//2



    # Draw the boxes around the cheeks
    cv2.rectangle(frame, (left_cheek_x1, left_cheek_y1), (left_cheek_x2, left_cheek_y2), (0, 255, 0), 2)  # Green for left cheek
    cv2.rectangle(frame, (right_cheek_x1, right_cheek_y1), (right_cheek_x2, right_cheek_y2), (0, 0, 255), 2)  # Red for right cheek

  # Display the resulting frame
    left_cheek_frame = frame[left_cheek_y1:left_cheek_y2, left_cheek_x1:left_cheek_x2]
    right_cheek_frame = frame[right_cheek_y1:right_cheek_y2, right_cheek_x1:right_cheek_x2]

  cv2.imshow("Detecting Cheeks in Video", frame)
  cv2.imshow("left cheek", left_cheek_frame)
  cv2.imshow("right cheek", right_cheek_frame)

  # Exit if 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Release capture and close all windows
cap.release()
cv2.destroyAllWindows()
