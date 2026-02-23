# Blink Rate Analysis Using Computer Vision

## Project Overview
This project uses computer vision and the Eye Aspect Ratio (EAR) algorithm to detect and analyze blink patterns during different cognitive tasks. The study compares blink rates between passive viewing (watching YouTube) and active focus (reading PDF documents).

## Technologies Used
- **OpenCV**: Video capture and processing
- **MediaPipe**: Face mesh detection and landmark tracking
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Python 3.12**

## Methodology

### Eye Aspect Ratio (EAR) Algorithm
The EAR algorithm calculates the ratio between the height and width of the eye using specific facial landmarks:
- **Left Eye Landmarks**: [33, 160, 158, 133, 153, 144]
- **Right Eye Landmarks**: [362, 385, 387, 263, 373, 380]
- **Blink Threshold**: 0.20 (when EAR drops below this value, a blink is detected)

### Data Collection Process
1. **Recording Setup**: 60-second video recordings using webcam
2. **Test Conditions**:
   - **Baseline/YouTube Session**: Passive viewing of YouTube content
   - **Focus/Reading Session**: Active reading of PDF documents
3. **Automated Analysis**: Frame-by-frame processing to extract EAR values and count blinks

## Dataset
- **recording_1.mp4**: Initial test recording (58.6 seconds)
- **recording_2(60sec_youtube).mp4**: YouTube viewing session
- **recording_3(60_sec_pdf).mp4**: PDF reading session (59.5 seconds)

## Results

### Quantitative Analysis
| Activity | Total Blinks | Duration | Blink Rate |
|----------|-------------|----------|------------|
| YouTube/Baseline | 8 | 58.6s | 0.14 blinks/sec |
| Reading/Focus | 3 | 59.5s | 0.05 blinks/sec |

**Key Finding**: **64% reduction** in blink rate during focused reading compared to passive viewing

### Visualizations
The analysis generates a comprehensive figure (`blink_research_summary.png`) containing:
1. **Time-Series Plot**: Real-time EAR values showing blink patterns over the entire recording
2. **Bar Chart**: Direct comparison of blink rates between activities

## Files Structure
```
cv_module5/
├── blink_rate.py                    # Main recording script with real-time blink detection
├── analysis_dashboard.py            # Post-processing analysis and visualization
├── blink_research_summary.png       # Generated analysis figure
├── README.md                         # This file
└── video_recording/                 # Recorded videos directory
    ├── recording_1.mp4
    ├── recording_2(60sec_youtube).mp4
    └── recording_3(60_sec_pdf).mp4
```

## How to Run

### Prerequisites
```bash
pip install opencv-python mediapipe matplotlib numpy
```

### Recording New Data
```bash
python3 blink_rate.py
```
- Records 60 seconds of video
- Displays real-time blink count
- Saves video as `recording_N.mp4` (auto-incremented)
- Press 'q' to stop early

### Analyzing Recordings
```bash
python3 analysis_dashboard.py
```
- Processes specified video files
- Generates visualization
- Saves results as `blink_research_summary.png`

## Scientific Context

### Cognitive Load and Blink Rate
Research has shown that blink rate is inversely correlated with cognitive load:
- **High cognitive demand** (reading, problem-solving) → **Lower blink rate**
- **Low cognitive demand** (passive viewing) → **Higher blink rate**

### Applications
- **Education**: Monitor student engagement and cognitive load
- **UX Research**: Assess interface complexity and user attention
- **Health Monitoring**: Detect eye strain and fatigue
- **Driver Safety**: Monitor alertness levels

## Conclusions

1. **Validation of Hypothesis**: The significant reduction in blink rate during reading confirms the relationship between cognitive load and blinking behavior.

2. **Technical Success**: The EAR algorithm with MediaPipe face mesh provides reliable blink detection without specialized hardware.

3. **Practical Implications**: This system could be integrated into:
   - E-learning platforms to optimize content delivery
   - Workplace monitoring for ergonomic assessments
   - Gaming/VR systems for user experience optimization

## Future Improvements
- Add support for multiple face detection
- Implement real-time cognitive load estimation
- Create a web-based interface for easier access
- Expand dataset with more participants and activities
- Add eye movement tracking for comprehensive analysis

## Author
Elisha Theetla

## License
MIT License

## Acknowledgments
- MediaPipe team for the robust face mesh model
- OpenCV community for comprehensive computer vision tools
- Research papers on EAR algorithm and cognitive load measurement