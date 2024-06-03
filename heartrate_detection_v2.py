import cv2
import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2


#window sizes for extracting heartrate
# windowSizes=[30,45,50,55,60,65]
windowSizes=[45]
#low and high cutoff frequencies
lowcut = 0.75  # Low cut frequency for bandpass filter
highcut = 2.5  # High cut frequency for bandpass filter

#sampling rate
fs = 2*highcut





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



def butter_bandpass(data, lowcut, highcut, fs, order=5):
  nyquist = 0.5 * fs  # Nyquist frequency
  lowcut_norm = lowcut / nyquist
  highcut_norm = highcut / nyquist
  b, a = butter(order, [lowcut_norm, highcut_norm], btype='bandpass')

  fft_data = fft2(data)
  filtered_fft = b[:, :, np.newaxis] * fft_data
  filtered_image = ifft2(filtered_fft).real

  return filtered_image



  
def bandpass_filter(frames, lowcut, highcut, fs):
  filtered_images = np.empty_like(frames)
  for i, frame in enumerate(frames):
    filtered_images[i] = butter_bandpass(frame, lowcut, highcut, fs)
  return filtered_images



def hearrate_detected(signal):
  #provide implementation for detecting heartrate
  return 

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
  Rf = bandpass_filter(Rn, lowcut, highcut, fs)
  Gf = bandpass_filter(Gn, lowcut, highcut, fs)
  Bf = bandpass_filter(Bn, lowcut, highcut, fs)
  Xf = bandpass_filter(Xs, lowcut, highcut, fs)
  Yf = bandpass_filter(Ys, lowcut, highcut, fs)

  alpha = np.std(Xf) / np.std(Yf)

  Signal = 3*(1-(alpha/2))*Rf - 2*(1 + (alpha/2))*Gf + ((3*alpha)/2)*Bf
  print(hearrate_detected(Signal))


cap.release()
cv2.destroyAllWindows()
print("frames:",fps,"width:",width,"height:",height)

