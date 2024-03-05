import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import dlib


def calculate_refined_pulse_signal(Rn, Gn, Bn, Xs, Ys, sampling_rate, low_cutoff, high_cutoff):

    Rf = bandpass_filter(Rn, low_cutoff, high_cutoff, sampling_rate)
    Gf = bandpass_filter(Gn, low_cutoff, high_cutoff, sampling_rate)
    Bf = bandpass_filter(Bn, low_cutoff, high_cutoff, sampling_rate)
    Xf = bandpass_filter(Xs, low_cutoff, high_cutoff, sampling_rate)
    Yf = bandpass_filter(Ys, low_cutoff, high_cutoff, sampling_rate)

    alpha = np.std(Xf) / np.std(Yf)

    Signal = 3*(1-(alpha/2))*Rf - 2*(1 + (alpha/2))*Gf + ((3*alpha)/2)*Bf

    return Signal

def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def normalizedSignal(channel):
    mini = np.min(channel)
    normal_signal = (channel - mini)/(np.max(channel) - mini)
    return normal_signal

def calculate_pulse_signal(R, G, B,sampling_rate):
    #pure chrom based

    # Rn=normalizedSignal(R)
    # Gn=normalizedSignal(G)
    # Bn=normalizedSignal(B)

    # Xs=Rn-Gn
    # Ys=0.5*Rn + 0.5*Gn - Bn

    #based on paper

    Rn = R / np.mean(R)
    Gn = G / np.mean(G)
    Bn = B / np.mean(B)    

    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn

    low_cutoff = 0.5
    high_cutoff = 4.1

    S_refined = calculate_refined_pulse_signal(Rn, Gn, Bn,Xs, Ys, sampling_rate, low_cutoff, high_cutoff)

    return S_refined

# Open a video file
video_path = 'videos/real/brad.mp4'
cap = cv2.VideoCapture(video_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")
avg_pulse_signal=np.empty([])
left_cheek_indices = [0, 4, 29, 8]
right_cheek_indices = [16, 12, 29, 8]

# Check if the video is successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is not read successfully, break the loop
    if not ret:
        break

    # Check if the frame has 3 channels (R, G, B)
    if frame.shape[-1] != 3:
        raise ValueError("The frame should have 3 color channels (R, G, B)")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        cropped_frame = frame[y:y+h, x:x+w]
        # Extract R, G, B channels
        R, G, B = cv2.split(cropped_frame)
        # sampling_rate = 2*int(cap.get(cv2.CAP_PROP_FPS))
        sampling_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # Apply the algorithm
        pulse_signal = calculate_pulse_signal(R, G, B,sampling_rate)

        landmarks = predictor(gray, face)

        # Get the left and right cheek landmark points
        left_cheek_points = [landmarks.part(i) for i in left_cheek_indices]
        right_cheek_points = [landmarks.part(i) for i in right_cheek_indices]

        left_cheek_x1 = min(p.x for p in left_cheek_points)
        left_cheek_y1 = min(p.y for p in left_cheek_points)
        left_cheek_x2 = max(p.x for p in left_cheek_points)
        left_cheek_y2 = max(p.y for p in left_cheek_points)

        if(left_cheek_x2-left_cheek_x1 > 25):
            diff=left_cheek_x2-left_cheek_x1
            left_cheek_x1+=diff//4
            left_cheek_x2-=diff//3

        if(left_cheek_y2-left_cheek_y1 > 25):
            diff=left_cheek_y2-left_cheek_y1
            left_cheek_y1+=diff//5
            left_cheek_y2-=diff//2

        right_cheek_x1 = min(p.x for p in right_cheek_points)
        right_cheek_y1 = min(p.y for p in right_cheek_points)
        right_cheek_x2 = max(p.x for p in right_cheek_points)
        right_cheek_y2 = max(p.y for p in right_cheek_points)

        if(right_cheek_x2-right_cheek_x1 > 25):
            diff=right_cheek_x2-right_cheek_x1
            right_cheek_x1+=diff//3
            right_cheek_x2-=diff//4

        if(right_cheek_y2-right_cheek_y1 > 25):
            diff=right_cheek_y2-right_cheek_y1
            right_cheek_y1+=diff//5
            right_cheek_y2-=diff//2

        left_cheek_x1 -= x
        left_cheek_y1 -= y
        left_cheek_x2 -= x
        left_cheek_y2 -= y

        right_cheek_x1 -= x
        right_cheek_y1 -= y
        right_cheek_x2 -= x
        right_cheek_y2 -= y

        # Extract cheek ROIs from the cropped_frame 
        left_cheek_frame = pulse_signal[left_cheek_y1:left_cheek_y2, left_cheek_x1:left_cheek_x2]
        right_cheek_frame = pulse_signal[right_cheek_y1:right_cheek_y2, right_cheek_x1:right_cheek_x2]
        cv2.rectangle(cropped_frame, (left_cheek_x1, left_cheek_y1), (left_cheek_x2, left_cheek_y2), (0, 255, 0), 2)  
        cv2.rectangle(cropped_frame, (right_cheek_x1, right_cheek_y1), (right_cheek_x2, right_cheek_y2), (0, 0, 255), 2) 

        #-----------------------------heart_rate detection part----------------------------------------

        # avg_pulse_signal=np.mean(pulse_signal, axis=(0, 1))
        avg_pulse_signal=np.append(avg_pulse_signal, np.mean(pulse_signal, axis=(0, 1)))

        peaks, _ = find_peaks(avg_pulse_signal, prominence=0.4)  

        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sampling_rate * 60  
            heart_rate_peaks = np.mean(peak_intervals)
        else:
            heart_rate_peaks = np.nan  # No peaks detected

        # Frequency Analysis (FFT)
        fft_result = np.abs(np.fft.rfft(avg_pulse_signal))
        freqs = np.fft.rfftfreq(len(avg_pulse_signal), 1/sampling_rate)
        peak_freq = freqs[np.argmax(fft_result)]
        heart_rate_fft = peak_freq * 60

        # Display heart rate estimations
        cv2.putText(cropped_frame, f"HR(P):{heart_rate_peaks:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(cropped_frame, f"HR(F):{heart_rate_fft:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



        #------------------------------------------------------------------------------------


        cv2.imshow("Detecting Cheeks in Video", cropped_frame)
        cv2.imshow("Pulse Signal", pulse_signal)
        cv2.imshow("left cheek", left_cheek_frame)
        cv2.imshow("right cheek", right_cheek_frame)

        # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
