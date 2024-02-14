import cv2
import numpy as np
from scipy.signal import butter, lfilter

def normalize_channel(channel):
    return channel / np.mean(channel)

def skin_tone_standardization(channel):
    standardized_channel = channel / np.mean(channel)
    return standardized_channel

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
    Rn = normalize_channel(R)
    Gn = normalize_channel(G)
    Bn = normalize_channel(B)

    Xs = 3 * Rn - 2 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn

    low_cutoff = 0.6 
    high_cutoff = 4.0

    S_refined = calculate_refined_pulse_signal(Xs, Ys, sampling_rate, low_cutoff, high_cutoff)

    return S_refined

# Read the video
video_path = 'videos/real/TCS.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sampling_rate = 2*int(cap.get(cv2.CAP_PROP_FPS))
# Process each frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Check if the frame has 3 channels (R, G, B)
    if frame.shape[-1] != 3:
        raise ValueError("The frame should have 3 color channels (R, G, B)")

    # Extract R, G, B channels
    R, G, B = cv2.split(frame)

    # Apply the algorithm
    pulse_signal = calculate_pulse_signal(R, G, B, sampling_rate)
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Pulse Signal', pulse_signal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
