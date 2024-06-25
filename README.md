
# Detect QR Live and Uploaded Video Analysis

## Project Purpose

This project aims to provide a tool for analyzing QR codes in both live video streams and pre-recorded video files. It identifies QR codes (Aruco markers), annotates them with relevant information (such as ID, distance, yaw angle, and more), and provides real-time feedback and controls through a graphical interface.

## Features

- **Live Video Analysis**: Analyze QR codes from a live camera feed.
- **Uploaded Video Analysis**: Analyze QR codes from a pre-recorded video file.
- **Annotations**: Display QR code information such as ID, distance, and yaw angle directly on the video.
- **Controls**: Interactive controls for pausing, stepping through frames, and saving frames during live analysis.
- **CSV Logging**: Log the detected QR code information to a CSV file.
- **Video Saving**: Save the analyzed live video feed to a file.

## Requirements

- Python 3.10 or later
- OpenCV 4.10.0 or later
- Tkinter (for the graphical interface)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/qr-code-analysis.git
   cd qr-code-analysis
   ```

2. **Install Dependencies**

   Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Main Script**

   ```bash
   python src/main.py
   ```

2. **Graphical Interface**

   The graphical interface provides two main options:
   
   - **Select Video File**: Choose a pre-recorded video file for analysis.
   - **Live Camera**: Start live analysis using your computer's camera.

### Controls

- **Exit**: Press 'q' or 'e'
- **Step Right**: Press 'd' (Displays a right arrow with "Turn Right" text)
- **Step Left**: Press 'a' (Displays a left arrow with "Turn Left" text)
- **Step Up**: Press 'w' (Displays an up arrow with "Turn Up" text)
- **Step Down**: Press 's' (Displays a down arrow with "Turn Down" text)
- **Save Frame**: Press 'p' (Saves the current frame as `TargetFrame.jpg` in the `Output` folder)

## File Structure

### 1. `main.py`

The entry point of the application. It initializes the graphical interface and provides buttons for selecting a video file or starting the live camera analysis.

**Main Functions:**
- `choose_video_file()`: Opens a file dialog to select a video file and starts the analysis.
- `start_live_video()`: Starts the live camera analysis.
- `main()`: Sets up the Tkinter GUI and binds the buttons to their respective functions.

### 2. `upload_video_analysis.py`

Handles the analysis of pre-recorded video files.

**Main Functions:**
- `draw_annotations()`: Draws annotations on the video frames.
- `calculate_pose_and_distance()`: Calculates the pose and distance of detected QR codes.
- `process_video()`: Processes the video frames, detects QR codes, annotates them, and writes the output to a new video file.
- `start_analysis()`: Sets up the paths and starts the video processing.

### 3. `live_video_analysis.py`

Handles the live camera analysis.

**Main Functions:**
- `draw_annotations()`: Draws annotations on the video frames.
- `calculate_pose_and_distance()`: Calculates the pose and distance of detected QR codes.
- `draw_controls()`: Draws control arrows on the video frames based on user input.
- `process_live_video()`: Processes the live video frames, detects QR codes, annotates them, handles user input, and writes the output to a new video file.
- `start_live_analysis()`: Sets up the paths, initializes the video writer, and starts the live video processing.

### 4. `utils.py`

Contains utility functions for setting up paths.

**Main Functions:**
- `setup_paths()`: Sets up and returns the paths for input video, output video, CSV file, and log file.

## Logging

Logs are saved to the `Output/app.log` file. This log file contains information about the application's execution, including any errors encountered during the video processing.

## Output

- **Processed Video**: The processed video file with annotations is saved to the `Output/output_video.mp4`.
- **CSV Log**: A CSV file containing the detected QR code information is saved to `Output/output_data.csv`.
- **Saved Frames**: Any frames saved during live analysis are stored as `TargetFrame.jpg` in the `Output` folder.

## License

This project is licensed under the MIT License. 

## Authors

- Roni Michaeli
- Elor Israeli
- Naor Ladani
- Roi Asraf