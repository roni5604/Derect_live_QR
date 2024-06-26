import os
import cv2
import cv2.aruco as aruco
import numpy as np
import logging
import csv
from utils import setup_paths

# Constants
ARUCO_MARKER_SIZE = 0.05  # Size of the Aruco marker in meters
# Camera calibration parameters for the camera used to capture the video
CAMERA_MATRIX = np.array([[950.0, 0.0, 640.0],
                          [0.0, 950.0, 360.0],
                          [0.0, 0.0, 1.0]])

DISTORTION_COEFFICIENTS = np.array([0.1, -0.25, 0.001, 0.0, 0.0])


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
    """
    Calculates the pose and distance of the Aruco marker.

    Args:
        corners (np.ndarray): Corners of the detected Aruco marker.

    Returns:
        tuple: Distance, yaw angle, pitch angle, roll angle, translation vector, and rotation vector.
    """
    rotation_vec, translation_vec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_SIZE, CAMERA_MATRIX,
                                                                           DISTORTION_COEFFICIENTS)
    distance = np.linalg.norm(translation_vec)
    yaw_angle = np.arctan2(translation_vec[0][0][0], translation_vec[0][0][2])
    yaw_angle_degrees = np.degrees(yaw_angle)
    yaw_angle_degrees = (yaw_angle_degrees + 180) % 360 - 180  # Normalize to -180 to 180 degrees

    # Convert rotation vector to rotation matrix
    matrix_rotation_vec, _ = cv2.Rodrigues(rotation_vec[0][0])

    # Calculate pitch, yaw, and roll from the rotation matrix
    pitch_angle = np.degrees(np.arctan2(matrix_rotation_vec[1, 0], matrix_rotation_vec[0, 0]))
    roll_angle = np.degrees(np.arctan2(matrix_rotation_vec[2, 1], matrix_rotation_vec[2, 2]))

    return distance, yaw_angle_degrees, pitch_angle, roll_angle, translation_vec, rotation_vec


def draw_controls(frame, direction):
    h, w, _ = frame.shape
    arrow_color = (0, 255, 0)
    thickness = 6
    arrow_length = 100

    if direction == "turn_right":
        cv2.arrowedLine(frame, (w - 120, h // 2), (w - 120 + arrow_length, h // 2), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Turn Right", (w - 300, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "turn_left":
        cv2.arrowedLine(frame, (120, h // 2), (120 - arrow_length, h // 2), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Turn Left", (200, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "turn_up":
        cv2.arrowedLine(frame, (w // 2, 120), (w // 2, 120 - arrow_length), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Turn Up", (w // 2 - 70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "turn_down":
        cv2.arrowedLine(frame, (w // 2, h - 120), (w // 2, h - 120 + arrow_length), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Turn Down", (w // 2 - 100, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "move_left":
        cv2.arrowedLine(frame, (120, h // 2), (120 - arrow_length, h // 2), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Move Left", (200, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "move_right":
        cv2.arrowedLine(frame, (w - 120, h // 2), (w - 120 + arrow_length, h // 2), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Move Right", (w - 300, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "move_forward":
        cv2.arrowedLine(frame, (w // 2, 120), (w // 2, 120 - arrow_length), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Move Forward", (w // 2 - 70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)
    elif direction == "move_backward":
        cv2.arrowedLine(frame, (w // 2, h - 120), (w // 2, h - 120 + arrow_length), arrow_color, thickness, tipLength=0.5)
        cv2.putText(frame, "Move Backward", (w // 2 - 100, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)


def process_live_video(source, csv_writer, out):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        raise IOError("Cannot open camera")

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    frame_id = 0
    direction = None
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    target_frame_path = os.path.join(base_dir, "Output", "TargetFrame.jpg")

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

        if direction:
            draw_controls(frame, direction)
            direction = None  # Reset direction after drawing

        controls_img = np.zeros((320, frame.shape[1], 3), dtype=np.uint8)  # Increased height for controls section
        cv2.putText(controls_img, "Controls", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(controls_img, "- Exit: Press 'q' or 'e'", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(controls_img, "- Turn Right: Press 'd'  |  Move Right: Press 'l'", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(controls_img, "- Turn Left: Press 'a'  |  Move Left: Press 'j'", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(controls_img, "- Turn Up: Press 'w'  |  Move Forward: Press 'o'", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(controls_img, "- Turn Down: Press 's'  |  Move Backward: Press 'k'", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(controls_img, "- Save Frame: Press 'p'", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Create a larger combined image for display
        combined_img = np.zeros((frame.shape[0] + controls_img.shape[0], frame.shape[1], 3), dtype=np.uint8)
        combined_img[:frame.shape[0], :] = frame
        combined_img[frame.shape[0]:, :] = controls_img

        cv2.imshow('Live Video', combined_img)
        out.write(frame)  # Write frame to output video

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('e'):
            break
        elif key == ord('d'):
            direction = "turn_right"
        elif key == ord('a'):
            direction = "turn_left"
        elif key == ord('w'):
            direction = "turn_up"
        elif key == ord('s'):
            direction = "turn_down"
        elif key == ord('l'):
            direction = "move_right"
        elif key == ord('j'):
            direction = "move_left"
        elif key == ord('o'):
            direction = "move_forward"
        elif key == ord('k'):
            direction = "move_backward"
        elif key == ord('p'):
            cv2.imwrite(target_frame_path, frame)
            logging.info(f"Saved current frame as {target_frame_path}")

        frame_id += 1

    cap.release()
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()
    logging.info("Live video processing ended.")


def start_live_analysis():
    _, output_video_path, csv_path, _ = setup_paths()

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame ID', 'QR id', 'QR 2D', 'QR 3D'])
        logging.info("CSV file initialized.")

        # Setup video writer
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open camera")
            raise IOError("Cannot open camera")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        cap.release()

        process_live_video(0, csv_writer, out)

    logging.info("CSV file closed.")
