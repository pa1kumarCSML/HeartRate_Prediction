import cv2
import numpy as np

# Define functions for pulse signal extraction process
def normalize_color_channel(channel):
    """
    Normalize color channel by dividing each sample by its mean.
    """
    return channel / np.mean(channel)

def roverg_method(green_channel, red_channel):
    """
    Calculate pulse signal using RoverG method.
    """
    return (green_channel / red_channel) - 1

def compute_chrominance_signals(red_channel, green_channel, blue_channel):
    """
    Compute chrominance signals X and Y.
    """
    x = red_channel - green_channel
    y = 0.5 * red_channel + 0.5 * green_channel - blue_channel
    return x, y

def skin_tone_standardization(red_channel, green_channel, blue_channel):
    """
    Perform skin-tone standardization on RGB channels.
    """
    # Standard skin-tone values
    Rs, Gs, Bs = 0.7682, 0.5121, 0.3841
    # Standardize RGB channels
    Rs_channel = Rs * red_channel
    Gs_channel = Gs * green_channel
    Bs_channel = Bs * blue_channel
    return Rs_channel, Gs_channel, Bs_channel

def fixed_algorithm(red_channel, green_channel, blue_channel):
    """
    Extract pulse signal using fixed algorithm.
    """
    # Coefficients for fixed algorithm
    c1, c2, c3 = 1.5, -3, 1.5
    return c1 * red_channel + c2 * green_channel + c3 * blue_channel

def adjust_pulse_signal(Xs, Ys):
    """
    Adjust pulse signal based on standard deviations of chrominance signals.
    """
    alpha = np.std(Xs) / np.std(Ys)
    return Xs - alpha * Ys

# Function to process each frame of the video
def process_frame(frame):
    # Extract RGB channels from the frame
    red_channel = frame[:,:,2]
    green_channel = frame[:,:,1]
    blue_channel = frame[:,:,0]
    
    # Normalize color channels
    red_normalized = normalize_color_channel(red_channel)
    green_normalized = normalize_color_channel(green_channel)
    blue_normalized = normalize_color_channel(blue_channel)
    
    # Compute pulse signal using RoverG method
    pulse_signal_roverg = roverg_method(green_normalized, red_normalized)
    
    # Compute chrominance signals X and Y
    X, Y = compute_chrominance_signals(red_normalized, green_normalized, blue_normalized)
    
    # Perform skin-tone standardization
    Rs_channel, Gs_channel, Bs_channel = skin_tone_standardization(red_normalized, green_normalized, blue_normalized)
    
    # Extract pulse signal using fixed algorithm
    pulse_signal_fixed = fixed_algorithm(Rs_channel, Gs_channel, Bs_channel)
    
    # Adjust pulse signal
    adjusted_pulse_signal = adjust_pulse_signal(X, Y)
    
    # Return the processed pulse signal
    return pulse_signal_roverg, pulse_signal_fixed, adjusted_pulse_signal

# Load video file
video_capture = cv2.VideoCapture('videos/TCS.mp4')

# Check if video file opened successfully
if not video_capture.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Read the first frame to get frame dimensions
ret, frame = video_capture.read()
if not ret:
    print("Error: Unable to read the first frame.")
    exit()

# Define the output video writer
output_video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 
                                      30, (frame.shape[1], frame.shape[0]))

# Process each frame of the video
while True:
    # Read a frame
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Process the frame to extract pulse signal
    pulse_signal_roverg, pulse_signal_fixed, adjusted_pulse_signal = process_frame(frame)
    
    # Visualize or store the pulse signal (e.g., plot on the frame)
    # You can use matplotlib to plot the pulse signal on the frame
    
    # Write the frame with pulse signal to the output video
    output_video_writer.write(frame)

    # Display the frame (optional)
    cv2.imshow('Pulse Signal Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
video_capture.release()
output_video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
