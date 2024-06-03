import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# Window sizes for extracting heart rate
windowSizes = [45]
# Low and high cutoff frequencies
lowcut = 0.75  # Low cut frequency for bandpass filter
highcut = 2.5  # High cut frequency for bandpass filter

# Sampling rate
fs = 2 * highcut


def normalize_video(video, window_size):
    video = video.astype(np.float32)
    normalized_video = np.zeros_like(video)
    for frame_idx in range(video.shape[0]):
        # Get a window of frames around the current frame
        window_start = max(0, frame_idx - window_size // 2)
        window_end = min(video.shape[0], frame_idx + window_size // 2 + 1)
        window = video[window_start:window_end]

        # Calculate average color for each channel across the window
        average_color = np.mean(window, axis=0)

        # Normalize each channel of the current frame
        normalized_video[frame_idx] = video[frame_idx] / average_color

    return normalized_video


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.6 * fs  # Nyquist frequency
    lowcut_norm = lowcut / nyquist 
    highcut_norm = highcut / nyquist
    # print(lowcut,highcut,nyquist,lowcut_norm,highcut_norm)
    b, a = butter(order, [lowcut_norm, highcut_norm], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=0)
    return y


def bandpass_filter(frames, lowcut, highcut, fs, order=5):
    filtered_images = np.empty_like(frames)
    for i, frame in enumerate(frames):
        filtered_images[i] = apply_bandpass_filter(frame, lowcut, highcut, fs, order)
    return filtered_images


def hearrate_detected(signal, fps):
    # Collapse 3D signal array to 1D time series by averaging over spatial dimensions
    time_series = np.mean(signal, axis=(1, 2))

    # Apply FFT
    N = len(time_series)
    yf = fft(time_series)
    xf = fftfreq(N, 1 / fps)[:N // 2]

    # Find the dominant frequency in the expected heart rate range
    heart_rate_range = (lowcut, highcut)
    idx_range = np.where((xf >= heart_rate_range[0]) & (xf <= heart_rate_range[1]))
    dominant_freq = xf[idx_range][np.argmax(np.abs(yf[idx_range]))]

    # Convert frequency to beats per minute (BPM)
    bpm = dominant_freq * 60

    return bpm


# Read video from file
video_path = "videos/real/vid.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties (assuming known beforehand)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Allocate memory for storing video frames
all_frames = []

# Read frames from video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)


cap.release()
cv2.destroyAllWindows()

# Convert frames list to a numpy array
video = np.array(all_frames)

# Normalize the video
for windowSize in windowSizes:
    normalized_with_windowSize = normalize_video(video.copy(), windowSize)

    Rn = normalized_with_windowSize[:, :, :, 0]
    Gn = normalized_with_windowSize[:, :, :, 1]
    Bn = normalized_with_windowSize[:, :, :, 2]

    Xs = 3 * Rn - 2 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn

    # Apply bandpass filter
    Rf = bandpass_filter(Rn, lowcut, highcut, fs)
    Gf = bandpass_filter(Gn, lowcut, highcut, fs)
    Bf = bandpass_filter(Bn, lowcut, highcut, fs)
    Xf = bandpass_filter(Xs, lowcut, highcut, fs)
    Yf = bandpass_filter(Ys, lowcut, highcut, fs)

    alpha = np.std(Xf) / np.std(Yf)

    signal = np.empty_like(Rf)
    for i, frame in enumerate(signal):
        signal[i] = 3 * (1 - (alpha / 2)) * Rf[i] - 2 * (1 + (alpha / 2)) * Gf[i] + ((3 * alpha) / 2) * Bf[i]
    bpm = hearrate_detected(signal, fps)
    print(f"Estimated Heart Rate: {bpm} BPM")
    time_series = np.mean(signal, axis=(1, 2))
    plt.plot(time_series, label='Pulse Signal')
    plt.title('Pulse Signal over Time')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    # plt.xlim(0, len(signal)//fps)
    plt.legend()
    # Display the BPM on the plot
    plt.text(0.05, 0.95, f'Estimated Heart Rate: {bpm:.2f} BPM', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig('plots/pulse_signal.png')
