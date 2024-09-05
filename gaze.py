import cv2
import mediapipe as mp
import numpy as np
from collections import deque

def eye_gaze_detection():
    # Initialize video capture and Mediapipe face mesh
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    # Define indices for eye landmarks and pupil
    right_eye_indices = [33, 133]  # Right eye corners
    left_eye_indices = [362, 263]  # Left eye corners
    pupil_indices = {"right": 468, "left": 473}  # Pupil indices

    # Initialize smoothing buffers
    buffer_size = 10
    right_eye_buffer = deque(maxlen=buffer_size)
    left_eye_buffer = deque(maxlen=buffer_size)

    while True:
        # Read and preprocess the frame
        _, frame = cam.read()
        
        frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Extract coordinates for both eyes and pupils
            right_eye_left, right_eye_right = landmarks[right_eye_indices[0]], landmarks[right_eye_indices[1]]
            left_eye_left, left_eye_right = landmarks[left_eye_indices[0]], landmarks[left_eye_indices[1]]
            right_pupil, left_pupil = landmarks[pupil_indices["right"]], landmarks[pupil_indices["left"]]

            # Normalize the pupil positions within the eye region
            right_eye_width = right_eye_right.x - right_eye_left.x
            left_eye_width = left_eye_right.x - left_eye_left.x
            right_pupil_position = (right_pupil.x - right_eye_left.x) / right_eye_width
            left_pupil_position = (left_pupil.x - left_eye_left.x) / left_eye_width

            # Add positions to buffer for smoothing
            right_eye_buffer.append(right_pupil_position)
            left_eye_buffer.append(left_pupil_position)

            # Smooth the positions using a moving average
            smoothed_right_position = sum(right_eye_buffer) / len(right_eye_buffer) if right_eye_buffer else 0.5
            smoothed_left_position = sum(left_eye_buffer) / len(left_eye_buffer) if left_eye_buffer else 0.5

            # Determine eye focus based on smoothed pupil position
            right_eye_focus = "Focused" if 0.4 <= smoothed_right_position <= 0.55 else "Focus on Your Screen !!!"
            left_eye_focus = "Focused" if 0.4 <= smoothed_left_position <= 0.55 else "Focus on Your Screen !!!"

            # Combine direction for both eyes
            eye_direction = "Focused" if "Focused" in right_eye_focus and "Focused" in left_eye_focus else "Focus on Your Screen !!!"

            # Draw eye focus direction on frame
            cv2.putText(frame, eye_direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Visualize the pupils and eye corners (optional for debugging)
            right_pupil_pos = (int(right_pupil.x * frame_w), int(right_pupil.y * frame_h))
            left_pupil_pos = (int(left_pupil.x * frame_w), int(left_pupil.y * frame_h))
            cv2.circle(frame, right_pupil_pos, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_pupil_pos, 5, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Eye Gaze Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

# Call the function to start eye gaze detection
eye_gaze_detection()
