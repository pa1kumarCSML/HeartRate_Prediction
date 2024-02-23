import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Generate sample data (replace this with your actual signal)
t = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * t)  # Example sinusoidal signal

# Apply peak detection
peaks, _ = find_peaks(signal, height=0)

# Plot original signal with detected peaks
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Signal')
plt.plot(t[peaks], signal[peaks], 'ro', label='Peaks')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal with Detected Peaks')
plt.legend()
plt.grid(True)
plt.show()
