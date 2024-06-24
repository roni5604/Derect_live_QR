import cv2
import cv2.aruco as aruco
import numpy as np
import logging
import csv
from utils import setup_paths  # Import setup_paths from utils.py

# Constants
ARUCO_MARKER_SIZE = 0.05  # Size of the Aruco marker in meters
# Camera calibration parameters for the camera used to capture the video
CAMERA_MATRIX = np.array([[921.170702, 0.000000, 459.904354],
                          [0.000000, 919.018377, 351.238301],
                          [0.000000, 0.000000, 1.000000]])
DISTORTION_COEFFICIENTS = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

def initialize_video_capture(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error("Cannot open video file")
        raise IOError("Cannot open video file")
    logging.info("Video capture initialized.")
    return cap

def setup_video_writer(cap, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    logging.info("Video writer setup complete.")
    return out

def draw_annotations(frame, corners, ids, distances, yaw_angles):
    for i, corner in enumerate(corners):
        pts = corner.reshape((4, 2)).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=5)
        if ids is not None:
            text_position = (pts[3][0][0], pts[3][0][1] + 30)
            cv2.putText(frame, f'ID: {ids[i][0]}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text_position = (pts[3][0][0], pts[3][0][1] + 60)
            cv2.putText(frame, f'Distance: {distances[i]:.2f}m', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text_position = (pts[3][0][0], pts[3][0][1] + 90)
            cv2.putText(frame, f'Yaw: {yaw_angles[i]:.2f} deg', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def calculate_pose_and_distance(corners):
    rotation_vec, translation_vec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_SIZE, CAMERA_MATRIX, DISTORTION_COEFFICIENTS)
    distance = np.linalg.norm(translation_vec)
    yaw_angle = np.arctan2(translation_vec[0][0][0], translation_vec[0][0][2])
    yaw_angle_degrees = np.degrees(yaw_angle)
    yaw_angle_degrees = (yaw_angle_degrees + 180) % 360 - 180  # Normalize to -180 to 180 degrees
    matrix_rotation_vec, _ = cv2.Rodrigues(rotation_vec[0][0])
    pitch_angle = np.degrees(np.arctan2(matrix_rotation_vec[1, 0], matrix_rotation_vec[0, 0]))
    roll_angle = np.degrees(np.arctan2(matrix_rotation_vec[2, 1], matrix_rotation_vec[2, 2]))
    return distance, yaw_angle_degrees, pitch_angle, roll_angle, translation_vec, rotation_vec

def process_video(cap, out, csv_writer):
    logging.info("Processing video.")
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video file reached.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)

        if ids is not None:
            distances = []
            yaw_angles = []
            roll_angles = []
            for i, aruco_id in enumerate(ids):
                distance, yaw_angle, pitch_angle, roll_angle, translation_vector, rotation_vector = calculate_pose_and_distance(corners[i])
                distances.append(distance)
                yaw_angles.append(yaw_angle)
                roll_angles.append(roll_angle)
                csv_writer.writerow(
                    [frame_id, aruco_id[0],
                     f"[{corners[i][0][0]}, {corners[i][0][1]}, {corners[i][0][2]}, {corners[i][0][3]}]",
                     f"[{distance}, {yaw_angle}, {pitch_angle}, {roll_angle}]"])
                logging.info(f"Frame {frame_id}: Aruco ID {aruco_id[0]} detected with distance {distance}, yaw angle {yaw_angle}, pitch angle {pitch_angle}, roll angle {roll_angle}.")
            draw_annotations(frame, corners, ids, distances, yaw_angles)

        out.write(frame)
        frame_id += 1

def release_resources(cap, out):
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("Resources released.")

def show_output_video(output_video_path_result):
    cap = cv2.VideoCapture(output_video_path_result)
    if not cap.isOpened():
        logging.error("Error: Could not open the output video.")
        return

    paused = False
    delay = 30

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            controls_img = np.zeros((220, frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(controls_img, "Controls", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(controls_img, "- Pause/Resume: Press 'p' or 'space'", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(controls_img, "- Exit: Press 'q' or 'e'", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(controls_img, "- Step Forward: Press 'd'", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(controls_img, "- Step Backward: Press 'a'", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            combined_img = np.vstack((frame, controls_img))
            cv2.imshow('Output Video', combined_img)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q') or key == ord('e'):
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused
        elif key == ord('d') or key == ord('D'):
            paused = True
            ret, frame = cap.read()
            if not ret:
                break
            combined_img = np.vstack((frame, controls_img))
            cv2.imshow('Output Video', combined_img)
        elif key == ord('a') or key == ord('A'):
            paused = True
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - 2, 0))
            ret, frame = cap.read()
            if not ret:
                break
            combined_img = np.vstack((frame, controls_img))
            cv2.imshow('Output Video', combined_img)

    cap.release()
    cv2.destroyAllWindows()

def start_analysis(source):
    output_video_path, csv_path, _ = setup_paths()

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame ID', 'QR id', 'QR 2D', 'QR 3D'])
        logging.info("CSV file initialized.")

        cap = initialize_video_capture(source)
        out = setup_video_writer(cap, output_video_path)
        process_video(cap, out, csv_writer)
        release_resources(cap, out)
        show_output_video(output_video_path)

    logging.info("CSV file closed.")
