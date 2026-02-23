import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye Landmark Indices (MediaPipe specific)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    # Vertical distances
    v1 = ((landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x)**2 + 
          (landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y)**2)**0.5
    v2 = ((landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x)**2 + 
          (landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y)**2)**0.5
    # Horizontal distance
    h = ((landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x)**2 + 
         (landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y)**2)**0.5
    return (v1 + v2) / (2.0 * h)

def process_video(video_path, label):
    """Process a video and extract EAR values and blink count"""
    print(f"Processing {label}: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    ear_values = []
    frame_count = 0
    blink_count = 0
    eye_closed = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_coords = results.multi_face_landmarks[0].landmark
            ear = (get_ear(mesh_coords, LEFT_EYE) + get_ear(mesh_coords, RIGHT_EYE)) / 2
            ear_values.append(ear)
            
            # Count blinks
            if ear < 0.20:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False
        else:
            # If no face detected, use NaN
            ear_values.append(np.nan)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    
    # Calculate time axis
    duration = frame_count / fps
    time_axis = np.linspace(0, duration, frame_count)
    
    # Calculate blink rate
    blink_rate = blink_count / duration if duration > 0 else 0
    
    print(f"  Total frames: {frame_count}")
    print(f"  Total blinks: {blink_count}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Blink rate: {blink_rate:.2f} blinks/sec\n")
    
    return time_axis, ear_values, blink_count, duration, blink_rate

def create_visualization(youtube_data, reading_data):
    """Create and save the visualization"""
    # Unpack data
    youtube_time, youtube_ear, youtube_blinks, youtube_duration, youtube_rate = youtube_data
    reading_time, reading_ear, reading_blinks, reading_duration, reading_rate = reading_data
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Blink Rate Analysis: YouTube vs Reading', fontsize=16, fontweight='bold')
    
    # Top Plot: Time-Series EAR
    ax1.plot(youtube_time, youtube_ear, label='YouTube/Baseline', color='#FF6B6B', alpha=0.7, linewidth=1)
    ax1.plot(reading_time, reading_ear, label='Reading/Focus', color='#4ECDC4', alpha=0.7, linewidth=1)
    ax1.axhline(y=0.20, color='red', linestyle='--', alpha=0.5, label='Blink Threshold (0.20)')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=12)
    ax1.set_title('Eye Aspect Ratio Over Time', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.4])
    
    # Bottom Plot: Bar Chart
    activities = ['YouTube/Baseline', 'Reading/Focus']
    blink_rates = [youtube_rate, reading_rate]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(activities, blink_rates, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, rate in zip(bars, blink_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f} blinks/sec',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Blink Rate (blinks/second)', fontsize=12)
    ax2.set_title('Blink Rate Comparison', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, max(blink_rates) * 1.2])
    
    # Add statistics box
    stats_text = f"YouTube: {youtube_blinks} blinks in {youtube_duration:.1f}s\n"
    stats_text += f"Reading: {reading_blinks} blinks in {reading_duration:.1f}s\n"
    stats_text += f"Reduction: {(1 - reading_rate/youtube_rate)*100:.1f}%" if youtube_rate > 0 else ""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'blink_research_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {output_path}")
    
    return fig

def main():
    """Main analysis function"""
    print("=" * 50)
    print("BLINK RATE ANALYSIS DASHBOARD")
    print("=" * 50 + "\n")
    
    # Check if videos exist - handle different naming conventions
    video1_path = 'video_recording/recording_1.mp4'
    
    # Try to find the second video with different naming patterns
    possible_video2_paths = [
        'video_recording/recording_2.mp4',
        'video_recording/recording_3(60_sec_pdf).mp4',
        'video_recording/recording_3.mp4'
    ]
    
    video2_path = None
    for path in possible_video2_paths:
        if os.path.exists(path):
            video2_path = path
            break
    
    if not os.path.exists(video1_path):
        print(f"Error: {video1_path} not found!")
        return
    
    if video2_path is None:
        print("Error: Reading/Focus video not found!")
        print("Looking for: recording_2.mp4 or recording_3(60_sec_pdf).mp4")
        return
    
    print(f"Using videos:")
    print(f"  YouTube/Baseline: {video1_path}")
    print(f"  Reading/Focus: {video2_path}\n")
    
    # Process videos
    youtube_data = process_video(video1_path, "YouTube/Baseline")
    reading_data = process_video(video2_path, "Reading/Focus")
    
    # Create visualization
    fig = create_visualization(youtube_data, reading_data)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    
    # Close matplotlib to free memory
    plt.close(fig)

if __name__ == "__main__":
    main()