import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Bandpass filter design
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Load video
video_path = 'path_to_your_video.mp4'
output_video_path = 'pulse_signal_video.mp4'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Parameters
standardized_skin_vector = np.array([0.7682, 0.5121, 0.3841])
lowcut = 0.75  # Low cut frequency for bandpass filter
highcut = 2.5  # High cut frequency for bandpass filter

# Storage for RGB channels
R_channel = []
G_channel = []
B_channel = []

# Read video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Average the RGB values across the frame
    R_channel.append(np.mean(frame_rgb[:, :, 0]))
    G_channel.append(np.mean(frame_rgb[:, :, 1]))
    B_channel.append(np.mean(frame_rgb[:, :, 2]))

cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays
R_channel = np.array(R_channel)
G_channel = np.array(G_channel)
B_channel = np.array(B_channel)

# Normalize the RGB channels
R_mean = np.mean(R_channel)
G_mean = np.mean(G_channel)
B_mean = np.mean(B_channel)

Rn = R_channel / R_mean
Gn = G_channel / G_mean
Bn = B_channel / B_mean

# Standardize the RGB channels
Rs = 0.7682 * Rn
Gs = 0.5121 * Gn
Bs = 0.3841 * Bn

# Create chrominance signals
Xs = 3 * Rn - 2 * Gn
Ys = 1.5 * Rn + Gn - 1.5 * Bn

# Apply bandpass filter
Xf = bandpass_filter(Xs, lowcut, highcut, fps)
Yf = bandpass_filter(Ys, lowcut, highcut, fps)

# Calculate the weighting factor alpha
alpha = np.std(Xf) / np.std(Yf)

# Compute the pulse signal
S = Xf - alpha * Yf

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Generate frames for the pulse signal video
for frame_idx in range(n_frames):
    # Create a blank frame
    pulse_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Draw the pulse signal on the frame
    time_axis = np.linspace(0, frame_width, len(S))
    signal_amplitude = np.interp(np.linspace(0, len(S), frame_width), np.arange(len(S)), S)
    signal_amplitude = (signal_amplitude - np.min(signal_amplitude)) / (np.max(signal_amplitude) - np.min(signal_amplitude))
    signal_amplitude = signal_amplitude * (frame_height // 2)  # Scale to fit the frame height
    
    for i in range(1, frame_width):
        x1 = int(time_axis[i - 1])
        y1 = int(frame_height // 2 - signal_amplitude[i - 1])
        x2 = int(time_axis[i])
        y2 = int(frame_height // 2 - signal_amplitude[i])
        cv2.line(pulse_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Write the frame to the output video
    out.write(pulse_frame)

out.release()

print("Pulse signal video has been saved to", output_video_path)
