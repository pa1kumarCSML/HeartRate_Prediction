import cv2
import numpy as np
from scipy.signal import butter, lfilter
import dlib

def calculate_refined_pulse_signal(Xs, Ys,sampling_rate):
    low_cutoff=0.5
    high_cutoff=4.1
    Xf = bandpass_filter(Xs, low_cutoff, high_cutoff, sampling_rate)
    Yf = bandpass_filter(Ys, low_cutoff, high_cutoff, sampling_rate)
    alpha = np.std(Xf) / np.std(Yf)
    S_refined = Xf - alpha * Yf
    return S_refined

def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    Xf = np.fft.fft(signal)
    
    # Create the frequency axis
    freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    
    # Create the bandpass filter mask
    bandpass_mask = (freq >= low_cutoff) & (freq <= high_cutoff)
    
    # Apply the filter mask to the Fourier Transform
    Xf_filtered = Xf * bandpass_mask
    
    # Perform Inverse Fourier Transform
    filtered_signal = np.fft.ifft(Xf_filtered)
    return filtered_signal

def calculate_pulse_signal(R, G, B):
    Rn = R / np.mean(R)
    Gn = G / np.mean(G)
    Bn = B / np.mean(B)
    Xs = 3 * Rn - 2 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn

    return Xs, Ys


# Open a video file
video_path = 'videos/real/jenny.mp4'
cap = cv2.VideoCapture(video_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")

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

    # Check if the frame has 3 channels (R, G, B)
    if frame.shape[-1] != 3:
        raise ValueError("The frame should have 3 color channels (R, G, B)")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cropped_frame = frame[y:y+h, x:x+w]
        # Extract R, G, B channels
        R, G, B = cv2.split(cropped_frame)
        sampling_rate = int(cap.get(cv2.CAP_PROP_FPS))
        # Calculate the pulse signal
        Xs, Ys = calculate_pulse_signal(R, G, B)
        # Apply bandpass filter to the pulse signal
        # Apply bandpass filter to the pulse signal
        pulse_signal = calculate_refined_pulse_signal(Xs, Ys, sampling_rate)

# Convert the pulse signal to the appropriate data type for visualization
        pulse_signal_display = np.uint8(pulse_signal)

# Display the refined pulse signal
        cv2.imshow('Refined Pulse Signal', pulse_signal)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
