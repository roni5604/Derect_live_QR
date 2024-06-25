import tkinter as tk
from tkinter import filedialog
from upload_video_analysis import start_analysis
from live_video_analysis import start_live_analysis
import logging
from utils import setup_paths

# Setup logging
_, _, _, log_path = setup_paths()
logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


def choose_video_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        start_analysis(file_path)


def start_live_video():
    start_live_analysis()


def main():
    root = tk.Tk()
    root.title("Video Analysis")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    choose_file_button = tk.Button(frame, text="Select Video File", command=choose_video_file, width=20)
    choose_file_button.grid(row=0, column=0, padx=5, pady=5)

    live_video_button = tk.Button(frame, text="Live Camera", command=start_live_video, width=20)
    live_video_button.grid(row=1, column=0, padx=5, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
