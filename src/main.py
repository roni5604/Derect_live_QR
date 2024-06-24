import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import os  # Add this import
from upload_video_analysis import start_analysis
from live_video_analysis import start_live_analysis
from utils import setup_paths  # Import setup_paths from utils.py

# Get paths and setup logging
_, _, log_path = setup_paths()
logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def choose_video_file():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if file_path:
        start_analysis(file_path)
    else:
        messagebox.showwarning("No file selected", "Please select a video file to proceed.")

def live_camera():
    start_live_analysis()

def main():
    logging.info("Application started.")

    # Create the main window
    root = tk.Tk()
    root.title("Aruco Marker Detection")

    # Create buttons
    btn_file = tk.Button(root, text="Select Video File", command=choose_video_file, width=20, height=2)
    btn_file.pack(pady=20)

    btn_live = tk.Button(root, text="Live Camera", command=live_camera, width=20, height=2)
    btn_live.pack(pady=20)

    # Run the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
