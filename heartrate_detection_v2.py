import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def normalize_video(video, window_size):  
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


def bandpass_filter(signal, lowcut, highcut, fps):
  N = len(signal)
  T = 1 / fps

  # Perform Fourier transform
  freq_signal = fft(signal)
  frequencies = fftfreq(N, T)

  # Apply bandpass filter
  mask = (frequencies > lowcut) & (frequencies < highcut)
  filtered_freq_signal = freq_signal * mask

  # Perform inverse Fourier transform
  filtered_signal = ifft(filtered_freq_signal)
  return np.real(filtered_signal)
  



#window sizes for extracting heartrate
# windowSizes=[10,20,30,40,50,60]
windowSizes=[10]

lowcut = 0.5  # Low cut frequency for bandpass filter
highcut = 4  # High cut frequency for bandpass filter
fs = 10
hr=[ 0.5,4]
# Read video from file
video_path="videos/real/vid.avi"
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

# Convert frames list to a numpy array
video = np.array(all_frames)

# Normalize the video
for windowSize in windowSizes:
    normalized_with_windowSize = normalize_video(video.copy(), windowSize)
    Rn = normalized_with_windowSize[:,:,:,0]
    Gn = normalized_with_windowSize[:,:,:,1]
    Bn = normalized_with_windowSize[:,:,:,2]   

    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn

    # Apply bandpass filter
    Xf = bandpass_filter(Xs, lowcut, highcut, fs)
    Yf = bandpass_filter(Ys, lowcut, highcut, fs)

    # Calculate the weighting factor alpha
    alpha = np.std(Xf) / np.std(Yf)

    # Compute the pulse signal
    S = Xf - alpha * Yf

cap.release()
cv2.destroyAllWindows()
print("frames:",fps,"width:",width,"height:",height)

