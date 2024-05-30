import cv2
import numpy as np
from scipy.fft import fft2, ifft2


def find_dominant_frequencies(image):
  """
  Detects dominant frequencies in an image using frequency domain analysis.

  Args:
      image: A 2D grayscale image array.

  Returns:
      A list containing the top N dominant frequencies (adjustable).
  """
  # Convert image to grayscale if needed (assuming RGB input)
  if len(image.shape) == 3:
    image = np.mean(image, axis=2)

  # Perform 2D FFT to get frequency domain representation
  fft_image = fft2(image)

  # Shift zero-frequency component to the center for visualization
  f_shift = np.fft.fftshift(fft_image)

  # Calculate magnitude spectrum (absolute values)
  magnitude_spectrum = np.abs(f_shift)

  # Find indices of top N largest values in magnitude spectrum
  top_n_indices = np.unravel_index(np.argpartition(magnitude_spectrum.flatten(), -10)[-10:], magnitude_spectrum.shape)

  # Convert indices back to frequencies (assuming equal spacing)
  rows, cols = image.shape
  nyquist_x = 0.5 / (cols / 2)  # Nyquist frequency in x-direction
  nyquist_y = 0.5 / (rows / 2)  # Nyquist frequency in y-direction
  dominant_frequencies = []
  for row, col in zip(*top_n_indices):
    # Normalize frequencies based on Nyquist limits
    normalized_freq_x = (col / (cols - 1)) * nyquist_x
    normalized_freq_y = (row / (rows - 1)) * nyquist_y
    dominant_frequencies.append((normalized_freq_x, normalized_freq_y))

  return dominant_frequencies


def analyze_video(video_path, num_dominant_freqs=5):
  """
  Analyzes a video and finds dominant frequencies over time.

  Args:
      video_path: Path to the video file.
      num_dominant_freqs: Number of dominant frequencies to track (default: 5).

  Returns:
      A list of lists, where each inner list contains the dominant frequencies for a frame.
  """
  cap = cv2.VideoCapture(video_path)

  # List to store dominant frequencies per frame
  all_dominant_frequencies = []

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find dominant frequencies in the frame
    dominant_frequencies = find_dominant_frequencies(gray_frame)[:num_dominant_freqs]
    all_dominant_frequencies.append(dominant_frequencies)

  cap.release()
  cv2.destroyAllWindows()

  return all_dominant_frequencies


# Example usage
video_path = "videos/real/vid.avi"
dominant_frequencies_over_time = analyze_video(video_path)

# Option 1: Calculate average dominant frequency across all frames
# Assuming dominant frequencies are represented as tuples (frequency_x, frequency_y)
average_freqs = np.mean(np.array(dominant_frequencies_over_time), axis=0)
print("Average dominant frequencies:", average_freqs)

# Option 2: Track changes in dominant frequencies over time
# This example focuses on the first dominant frequency for illustration
first_dominant_freqs = [df[0] for df in dominant_frequencies_over_time]
import matplotlib.pyplot as plt

plt.plot(first_dominant_freqs)
plt.xlabel("Frame Number")
plt.ylabel("Dominant Frequency (normalized)")
plt.title("Change in First Dominant Frequency Over Time")
plt.savefig('dominant_frequency_plot.png')
plt.show()

# Further analysis can be done here based on your needs

