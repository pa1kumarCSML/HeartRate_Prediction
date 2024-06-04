import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Window sizes for extracting heart rate
windowSizes = [40]
# Low and high cutoff frequencies
lowcut = 0.75  # Low cut frequency for bandpass filter
highcut = 2.25  # High cut frequency for bandpass filter

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
    nyquist = 0.5 * fs  # Nyquist frequency
    lowcut_norm = lowcut / nyquist 
    highcut_norm = highcut / nyquist
    # print(lowcut,highcut,lowcut_norm,highcut_norm)
    b, a = butter(order, [lowcut_norm, highcut_norm-0.1], btype='band')
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
video_path = "videos/fake/vid.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Allocate memory for storing video frames
left_cheek_frames = []
right_cheek_frames = []
forehead_frames = []
full_face_frames = []

isWAndHObtained = False
cheek_w, cheek_h = 0, 0
forehead_h, forehead_w = 0, 0
face_h, face_w = 0, 0

# Read frames from video and detect face region
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Assume the first detected face is the region of interest
        x, y, w, h = faces[0]

        # Define regions for left cheek, right cheek, and forehead
        if not isWAndHObtained:
            cheek_w = w // 4
            cheek_h = h // 4
            forehead_h = h // 5
            forehead_w = w // 2
            face_h, face_w = h, w
            isWAndHObtained = True

        left_cheek = frame[y + h // 2:y + h // 2 + cheek_h, x:x + cheek_w]
        right_cheek = frame[y + h // 2:y + h // 2 + cheek_h, x + w - cheek_w:x + w]
        forehead = frame[y:y + forehead_h, x + w // 4:x + w // 4 + forehead_w]
        full_face = frame[y:y + face_h, x:x + face_w]

        left_cheek_frames.append(left_cheek)
        right_cheek_frames.append(right_cheek)
        forehead_frames.append(forehead)
        full_face_frames.append(full_face)

cap.release()
cv2.destroyAllWindows()

# Function to process each region and estimate heart rate
def process_region(region_frames, region_name):
    # Convert frames list to a numpy array
    video = np.array(region_frames)

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
        print(f"Estimated Heart Rate from {region_name}: {bpm} BPM")
        
        # Plot the signal
        time_series = np.mean(signal, axis=(1, 2))
        plt.plot(time_series, label=f'{region_name} Pulse Signal')
        plt.title(f'{region_name} Pulse Signal over Time')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.legend()
        plt.text(0.05, 0.95, f'Estimated Heart Rate: {bpm:.2f} BPM', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(f'plots/{region_name}_pulse_signal.png')
        plt.clf()
        
        return bpm, signal

# Process each region and get heart rate
left_cheek_bpm, left_cheek_signal = process_region(left_cheek_frames, 'Left Cheek')
right_cheek_bpm, right_cheek_signal = process_region(right_cheek_frames, 'Right Cheek')
forehead_bpm, forehead_signal = process_region(forehead_frames, 'Forehead')
full_face_bpm, full_face_signal = process_region(full_face_frames, 'Full Face')

# Superimpose all plots in a single plot
plt.figure(figsize=(10, 6))
plt.plot(np.mean(left_cheek_signal, axis=(1, 2)), label='Left Cheek Pulse Signal')
plt.plot(np.mean(right_cheek_signal, axis=(1, 2)), label='Right Cheek Pulse Signal')
plt.plot(np.mean(forehead_signal, axis=(1, 2)), label='Forehead Pulse Signal')
plt.plot(np.mean(full_face_signal, axis=(1, 2)), label='Full Face Pulse Signal')
plt.title('Superimposed Pulse Signals')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('plots/superimposed_pulse_signals.png')
