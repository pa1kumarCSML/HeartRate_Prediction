import cv2
import numpy as np
from scipy.signal import butter, lfilter

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def calculate_refined_pulse_signal(Xs, Ys, sampling_rate, low_cutoff, high_cutoff):
    Xf = bandpass_filter(Xs, low_cutoff, high_cutoff, sampling_rate)
    Yf = bandpass_filter(Ys, low_cutoff, high_cutoff, sampling_rate)
    alpha = np.std(Xf) / np.std(Yf)
    S_refined = Xf - alpha * Yf
    return S_refined

def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def calculate_pulse_signal(R, G, B, sampling_rate):
    Rn = R / np.mean(R)
    Gn = G / np.mean(G)
    Bn = B / np.mean(B)

    Xs = 3 * Rn - 2 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn

    low_cutoff = 0.5  # Placeholder value, replace with actual value
    high_cutoff = 3.0  # Placeholder value, replace with actual value

    S_refined = calculate_refined_pulse_signal(Xs, Ys, sampling_rate, low_cutoff, high_cutoff)

    return S_refined

# Open a video file
video_path = 'videos/TCS.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is not read successfully, break the loop
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]

        # Check if the face region has 3 channels (R, G, B)
        if face_roi.shape[-1] != 3:
            raise ValueError("The face region should have 3 color channels (R, G, B)")

        # Extract R, G, B channels from the face region
        R, G, B = cv2.split(face_roi)

        sampling_rate = 2*cap.get(cv2.CAP_PROP_FPS)
        # Apply the algorithm
        pulse_signal = calculate_pulse_signal(R, G, B, sampling_rate)

        # Display the original frame with the face bounding box and the calculated pulse signal
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a blue rectangle around the face
        cv2.imshow('Original Frame with Face Detection', frame)
        cv2.imshow('Pulse Signal', pulse_signal)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
