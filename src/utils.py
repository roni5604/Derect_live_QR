import os

def setup_paths():
    """
    Sets up the paths for input video, output video, CSV file, and log file.

    Returns:
        tuple: Paths for input video, output video, CSV file, and log file.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    video_path = os.path.join(base_dir, "TestVideos", "classVideo.mp4")
    output_video_path = os.path.join(base_dir, "Output", "output_video.mp4")
    csv_path = os.path.join(base_dir, "Output", "output_data.csv")
    log_path = os.path.join(base_dir, "Output", "app.log")
    return video_path, output_video_path, csv_path, log_path
