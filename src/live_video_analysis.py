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

def process_live_video(source, csv_writer):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        raise IOError("Cannot open camera")

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Cannot read from camera")
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

        cv2.imshow('Live Video', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('e'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Live video processing ended.")

def start_live_analysis():
    _, csv_path, _ = setup_paths()

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame ID', 'QR id', 'QR 2D', 'QR 3D'])
        logging.info("CSV file initialized.")
        process_live_video(0, csv_writer)

    logging.info("CSV file closed.")