import cv2
import numpy as np
from scipy.signal import find_peaks

def extract_chrominance(frame):
    # Convert the frame to the YCrCb color space
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Extract the Cr channel (chrominance)
    chrominance = ycrcb_frame[:, :, 1]
    
    return chrominance

def estimate_pulse_rate(chrominance, fps):
    # Calculate the average value along each column (vertical axis)
    avg_chrominance = np.mean(chrominance, axis=0)
    
    # Apply a bilateral filter to the original chrominance signal
    filtered_chrominance = cv2.bilateralFilter(avg_chrominance.astype(np.float32), 5, 75, 75)
    
    # Normalize the signal
    normalized_signal = (filtered_chrominance - np.mean(filtered_chrominance)) / np.std(filtered_chrominance)
    
    # Ensure the signal is a 1-D array
    normalized_signal = normalized_signal.ravel()
    
    # Find peaks in the signal
    peaks, _ = find_peaks(normalized_signal, distance=int(fps/1.5), height=0.5)
    
    # Calculate pulse rate in beats per minute (BPM)
    pulse_rate = len(peaks) / len(normalized_signal) * fps * 60
    
    return pulse_rate

# Video file or webcam input
video_path = "videos/TCS.mp4"
cap = cv2.VideoCapture(video_path)  # Use 0 for webcam

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract chrominance from the frame
    chrominance = extract_chrominance(frame)
    
    # Estimate pulse rate
    pulse_rate = estimate_pulse_rate(chrominance, fps)
    
    # Display the result
    cv2.putText(frame, f"Pulse Rate: {pulse_rate:.2f} BPM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pulse Rate Estimation", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
