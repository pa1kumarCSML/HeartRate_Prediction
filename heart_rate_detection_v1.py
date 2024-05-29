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
video_path = 'videos/real/jenny.mp4'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = n_frames / fps

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

# Plot the resulting pulse signal
time_axis = np.arange(len(S)) / fps
plt.figure(figsize=(12, 6))
plt.plot(time_axis, S, label='Pulse Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Extracted Pulse Signal')
plt.legend()

# Save the plot
plot_path = 'plots/pulse_signal_plot.png'
plt.savefig(plot_path)

# Display the plot
plt.show()
