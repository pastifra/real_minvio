import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from pathlib import Path
from tqdm import tqdm
import utils


def detect_checkerboard_points(img_path: Path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cam_img_size = None

    images = img_path.glob("*.bmp")
    for fname in tqdm(images):
        img = cv2.imread(str(fname))

        if cam_img_size is None:
            cam_img_size = img.shape[:2]
        else:
            assert cam_img_size == img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
    return objpoints, imgpoints, cam_img_size


def create_K_undistort(
    undistort_img_size, 
    undistort_img_vert_fov, 
    undistort_img_horiz_fov):
    W, H = undistort_img_size

    fx = W / (2 * np.tan(undistort_img_horiz_fov / 2))
    fy = H / (2 * np.tan(undistort_img_vert_fov / 2))

    cx = W / 2
    cy = H / 2

    K_undistort = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])

    return K_undistort
    
def run_camera_calibration():
    undistort_img_size = (1024, 1024)
    undistort_img_vert_fov = np.deg2rad(70)
    undistort_img_horiz_fov = np.deg2rad(70)
    cached_points_path = utils.get_data_path() / "camera-calibration" / "cached_points.pkl"
    load_cached_points = cached_points_path.exists()
    img_path = utils.get_data_path() / "camera-calibration" / "2.5mm lens"

    if load_cached_points:
        with open(cached_points_path, "rb") as f:
            D = pickle.load(f)
            objpoints = D["objpoints"]
            imgpoints = D["imgpoints"]
            cam_img_size = D["cam_img_size"]
    else:
        objpoints, imgpoints, cam_img_size = detect_checkerboard_points(img_path)
        with open(cached_points_path, "wb") as f:
            D = {
                "objpoints": objpoints,
                "imgpoints": imgpoints,
                "cam_img_size": cam_img_size
            }
            pickle.dump(D, f)

    ## Calibrate the camera
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, cam_img_size[::-1], None, None)
    K_undistort = create_K_undistort(undistort_img_size, undistort_img_vert_fov, undistort_img_horiz_fov)

    img = cv2.imread(str(utils.get_data_path() / "camera-calibration" / "2.5mm lens" / "Basler_daA1920-160uc__40445204__20251214_143617580_0000.bmp"))
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, K_undistort, undistort_img_size[::-1], cv2.CV_32FC1)

    # Set the roi to keep FOV=90 degrees
    half_vert_dist = np.tan(undistort_img_vert_fov / 2) * K_undistort[1,1] # px
    half_horiz_dist = np.tan(undistort_img_horiz_fov / 2) * K_undistort[0,0] # px
    roi = np.asarray(
        [K_undistort[0,2] - half_horiz_dist, K_undistort[1,2] - half_vert_dist, 
           half_horiz_dist * 2, half_vert_dist * 2])
    roi = np.round(roi).astype(int)

    # Crop the remap parameters to the target FOV
    x, y, w, h = roi
    map_x = map_x[y:y+h, x:x+w]
    map_y = map_y[y:y+h, x:x+w]
    # Adjust the offset in K_new
    K_undistort[:2,2] -= np.asarray([x, y])

    # crop the image to the target FOV
    img_undistort = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Undistorted Image", img_undistort)
    cv2.waitKey(0)

    ## Reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("reprojection error: {} px".format(mean_error/len(objpoints)))

    ## Save the undistortion parameters
    undistortion_parameters = {
        "remap_params": {
            "map1": map_x,
            "map2": map_y,
            "interpolation": cv2.INTER_CUBIC
        },
        "K_cam": K, # Intrinsic matrix of the original image
        "K_undistort": K_undistort, # Intrisic matrix of the undistorted image
        "undistort_img_size": map_x.shape,
        "cam_img_size": cam_img_size,
        "undistort_img_vert_fov": undistort_img_vert_fov, 
        "undistort_img_horiz_fov": undistort_img_horiz_fov, 
    }
    with open(utils.get_data_path() / "camera-calibration" / "undistortion_params.pkl", "wb") as f:
        pickle.dump(undistortion_parameters, f)


if __name__ == "__main__":
    run_camera_calibration()