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
    low = float(low_cutoff) / float(nyquist)
    high = float(high_cutoff) / float(nyquist)
    b, a = butter(6.0, [low, high], btype='band')
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

    low_cutoff = 0.6
    high_cutoff = 4

    S_refined = calculate_refined_pulse_signal(Rn, Gn, Bn,Xs, Ys, sampling_rate, low_cutoff, high_cutoff)

    return S_refined

def calculate_heart_rate(peaks, fps):
    # Calculate time between consecutive peaks (in seconds)
    peak_times = np.array(peaks) / fps
    peak_intervals = np.diff(peak_times)

    # Calculate heart rate (beats per minute)
    heart_rate = 60 / np.mean(peak_intervals)
    return heart_rate

# Open a video file
video_path = 'videos/real/jenny.mp4'
cap = cv2.VideoCapture(video_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_files/shape_predictor_68_face_landmarks.dat")

left_cheek_indices = [0, 4, 29, 8]
right_cheek_indices = [16, 12, 29, 8]
left_cheek_pulses=np.empty([])

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
        cropped_frame = frame[y:y+h, x:x+w]  #face region cropped

        # Extract R, G, B channels
        B, G, R = cv2.split(cropped_frame)

        # sampling_rate = 2*int(cap.get(cv2.CAP_PROP_FPS))
        sampling_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # Apply the algorithm based on paper
        pulse_signal = calculate_pulse_signal(R, G, B,sampling_rate)

        #detecting roi--->left_cheek, right_cheek
        landmarks = predictor(gray, face)

        # Get the left and right cheek landmark points
        left_cheek_points = [landmarks.part(i) for i in left_cheek_indices]
        right_cheek_points = [landmarks.part(i) for i in right_cheek_indices]

        #Adjusting parameters for left_cheek and right_cheek

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

        # Extract ROI frames from the cropped_frame 
        left_cheek_frame = pulse_signal[left_cheek_y1:left_cheek_y2, left_cheek_x1:left_cheek_x2]
        right_cheek_frame = pulse_signal[right_cheek_y1:right_cheek_y2, right_cheek_x1:right_cheek_x2]

        left_cheek_pulses = np.append(left_cheek_pulses,np.mean(left_cheek_frame))
        peaks, _ = find_peaks(left_cheek_pulses)

        # Calculate heart rate from peaks
        heart_rate = calculate_heart_rate(peaks, sampling_rate)
        if heart_rate is not np.NaN:
            print(heart_rate)

        #Bounding Boxes for ROI
        cv2.rectangle(cropped_frame, (left_cheek_x1, left_cheek_y1), (left_cheek_x2, left_cheek_y2), (0, 255, 0), 2)  
        cv2.rectangle(cropped_frame, (right_cheek_x1, right_cheek_y1), (right_cheek_x2, right_cheek_y2), (0, 0, 255), 2) 


        #Displaying frames
        cv2.imshow("Detecting Cheeks in Video", cropped_frame)
        cv2.imshow("Pulse Signal", pulse_signal)
        cv2.imshow("left cheek", left_cheek_frame)
        cv2.imshow("right cheek", right_cheek_frame)
        
        # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
