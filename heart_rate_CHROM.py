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

def calculate_pulse_signal(R, G, B):
    Rn = normalize_channel(R)
    Gn = normalize_channel(G)
    Bn = normalize_channel(B)

    # Rs =  Gs = 0.5121Gn Bs = 0.3841Bn .
    # Rs = 0.7682*Rn
    # Gs = 0.5121*Gn
    # Bs = 0.3841*Bn

    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn

    sampling_rate = 1000  # Placeholder value, replace with actual value
    low_cutoff = 0.5  # Placeholder value, replace with actual value
    high_cutoff = 3.0  # Placeholder value, replace with actual value

    # Xs = 3 * Rs - 2 * Gs
    # Ys = 1.5 * Rs + Gs - 1.5 * Bs

    S_refined = calculate_refined_pulse_signal(Xs, Ys, sampling_rate, low_cutoff, high_cutoff)

    return S_refined

# Read the image
image = cv2.imread('images/leo1.jpg')

# Check if the image has 3 channels (R, G, B)
if image.shape[-1] != 3:
    raise ValueError("The image should have 3 color channels (R, G, B)")

# Extract R, G, B channels
R, G, B = cv2.split(image)

# Apply the algorithm
pulse_signal = calculate_pulse_signal(R, G, B)

# Display the original image and the calculated pulse signal
cv2.imshow('Original Image', image)
cv2.imshow('Pulse Signal', pulse_signal)
cv2.waitKey(0)
cv2.destroyAllWindows()
