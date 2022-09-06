import cv2
import glob
import os
import numpy as np


class CameraCalibration:
    def __init__(self, calibration_matrix_path=None, imgs_shape=(1920, 1080), imgs_dir="", checkerboard_dim=(6, 9)):
        """
        It creates an instance based calibration_matrix_path input. if path is provided, it will read
        the matrix from that path, otherwise all the other variable needs to be provided in order to calibrate
        the matrix.

        Args:
            calibration_matrix_path: Indicating if calibration matrix needs to be calculated or read from path
            imgs_shape: (width, height) of the checkboard image, the input images you want to undistort.
            imgs_dir: checkboard images directory path
            checkerboard_dim: checkerboard dimensions (height, width). the demension are the corners inside checkerboard
        Returns:

        """
        if calibration_matrix_path is None:
            if os.path.isdir(imgs_dir):
                self.calibration_matrix = self.get_calibrationmatrix(imgs_dir, imgs_shape, checkerboard_dim)
            else:
                print("calibration path is None, please provide proper imgs_dir path")
        else:
            calib_data = np.load(calibration_matrix_path, allow_pickle=True)
            self.calibration_matrix = calib_data.item()

        mtx = self.calibration_matrix["camera_matrix"]
        dist = self.calibration_matrix["distortion"]

        self.width, self.height = imgs_shape

        # Refining the camera matrix using parameters obtained by calibration
        newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1,
                                                               (self.width, self.height))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (self.width, self.height), 5)

    @staticmethod
    def get_calibrationmatrix(imgs_dir, imgs_shape, checkerboard_dim):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, checkerboard_dim[0] * checkerboard_dim[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2)

        imgs_paths_list = glob.glob(f"{imgs_dir}/*.*")
        for fname in imgs_paths_list:
            img = cv2.imread(fname)
            if (img.shape[1], img.shape[0]) != imgs_shape:
                print(f"{fname} is not of the desired shape i.e. {imgs_shape}")
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                # If desired number of corners are found in the image then ret = true
                ret, corners = cv2.findChessboardCorners(
                    gray, checkerboard_dim,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

                """
                If desired number of corner are detected, we refine the pixel coordinates 
                and display them on the images of checker board
                """
                if ret:
                    objpoints.append(objp)
                    # refining pixel coordinates for given 2d points.
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, checkerboard_dim, corners2, ret)
                    print("Press any key to previews next image")
                    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgs_shape, None, None)

        # calculate error rate
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"total error: {mean_error/len(objpoints)}")

        calibration_dictionary = {"camera_matrix": mtx,
                                  "distortion": dist,
                                  "rotation_vectors": rvecs,
                                  "translation_vectors": tvecs}

        np.save("camera_calibration_values.npy", calibration_dictionary)
        return calibration_dictionary

    def undistort_frame(self, img):
        """
        undistort frame based on the known calibration matrix
        """
        if img.shape[:2] != (self.height, self.width):
            print(f"Input image {img.shape[:2][::-1]} is not of the required shape i.e. {(self.width, self.height)}")
        else:
            dst = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
            return dst

    def undistort_images(self, input_path, output_dir="undistorted"):
        """
        save undistort images directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_types = ("jpg", "jpeg", "png")  # the tuple of file types
        files_grabbed = []

        # handling directory path
        if os.path.isdir(input_path):
            for t_name in file_types:
                files_grabbed.extend(glob.glob(f"{input_path}/*.{t_name}"))

        # single image path
        elif os.path.isfile(input_path):
            file_extension = os.path.basename(input_path).split(".")[-1]
            files_grabbed = [input_path] if file_extension in file_types else []

        # processing the required images
        for f_path in files_grabbed:
            img_name = os.path.basename(f_path)
            img = cv2.imread(f_path)

            out_img = self.undistort_frame(img)
            cv2.imwrite(os.path.join(output_dir, img_name), out_img)

            # Displaying the undistorted images
            cv2.namedWindow("Original Vs Undistorted", cv2.WINDOW_NORMAL)
            cv2.imshow("Original Vs Undistorted", np.hstack((img, out_img)))
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def original_image_point_in_undistorted_image(self, point):
        """
        This function get you the shifted original point in undistorted image.
        Args:
            point: x,y cordinates from original image

        Returns:
        a 1D-list having x,y cordinate.
        """
        sparse_img = np.zeros((self.height, self.width), dtype=bool)
        sparse_img[point[1]][point[0]] = True

        sparse_img = sparse_img.astype(np.float32)

        sparse_undist = self.undistort_frame(sparse_img)
        true_cord = np.argwhere(sparse_undist).tolist()
        return true_cord[0][::-1]
