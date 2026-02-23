import cv2
import mediapipe as mp
import time
import os

# 1. Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 2. Eye Landmark Indices (MediaPipe specific)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_ear(landmarks, eye_indices):
    # Vertical distances
    v1 = ((landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x)**2 + (landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y)**2)**0.5
    v2 = ((landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x)**2 + (landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y)**2)**0.5
    # Horizontal distance
    h = ((landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x)**2 + (landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y)**2)**0.5
    return (v1 + v2) / (2.0 * h)

# 3. Setup Video (0 for webcam, or "path/to/video.mp4")
cap = cv2.VideoCapture(0)

# Create video_recording directory if it doesn't exist
if not os.path.exists('video_recording'):
    os.makedirs('video_recording')

# Find the next recording number
recording_num = 1
while os.path.exists(f'video_recording/recording_{recording_num}.mp4'):
    recording_num += 1

# Get video properties for saving
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:  # Sometimes webcam FPS is 0, default to 30
    fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
output_path = f'video_recording/recording_{recording_num}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

blink_count = 0
eye_closed = False
start_time = time.time()

print(f"Recording for 60 seconds... Saving to {output_path}")
print("Press 'q' to stop early.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or (time.time() - start_time) > 60: break
    
    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        mesh_coords = results.multi_face_landmarks[0].landmark
        ear = (get_ear(mesh_coords, LEFT_EYE) + get_ear(mesh_coords, RIGHT_EYE)) / 2
        
        # 4. Blink Logic: If EAR drops below 0.2, eye is closed
        if ear < 0.20:
            if not eye_closed:
                blink_count += 1
                eye_closed = True
        else:
            eye_closed = False

    # Add text overlay
    cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Recording: {recording_num}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Write frame to video file
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Blink Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

elapsed_time = time.time() - start_time
print(f"\nVideo saved to: {output_path}")
print(f"Total Blinks: {blink_count}")
print(f"Recording Duration: {elapsed_time:.2f} seconds")
print(f"Blink Rate: {blink_count / elapsed_time:.2f} blinks/sec" if elapsed_time > 0 else "N/A")
