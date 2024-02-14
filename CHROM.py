import cv2
import numpy as np

def extract_heart_rate(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract region of interest (ROI) - here, we're assuming the entire frame is the ROI
        roi = gray

        # Apply temporal filtering to enhance color changes
        # (You would implement this based on your specific requirements)

        # Perform CHROM-based heart rate extraction
        # Here, let's calculate the average pixel intensity of the ROI over time as a simple measure
        average_intensity = np.mean(roi)

        # Convert average intensity to heart rate (this is just a placeholder for demonstration)
        # You would need to develop a more sophisticated method based on CHROM principles
        # For demonstration, let's assume a linear relationship between intensity and heart rate
        heart_rate = average_intensity * 2  # Adjust the scaling factor as needed

        # Display the estimated heart rate on the frame
        cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} BPM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the estimated heart rate
        cv2.imshow('Video', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'videos/real/jenny.mp4'
extract_heart_rate(video_path)
