import cv2
import numpy as np
import glob

# Define the chessboard size
chessboard_size = (7, 6)  # Number of inner corners per a chessboard row and column

# Define arrays to store object points and image points
obj_points = []
img_points = []

# Prepare the object points, like (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Load calibration images
images = glob.glob('Output/TargetFrame.jpg')

gray = None  # Initialize gray outside the loop

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        img_points.append(corners)
        obj_points.append(objp)
        # Optional: Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Check if any valid images were found
if len(obj_points) > 0 and len(img_points) > 0:
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None,
                                                                        None)

    if ret:
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)
    else:
        print("Camera calibration failed")
else:
    print("No valid images found for calibration")
