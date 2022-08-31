"""
This file shows the usage of CameraCalibration class from calibrate_camera.py.
"""
import cv2
import numpy as np
from calibrate_camera import CameraCalibration


def mouse_point(event, x, y, flags, params):
    """
    function from OpenCV tutorial to get mouse points
    """
    img, points = params
    if len(points) <= 3 and event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        points.append((x, y))
        print(points)


def visualize_straight_lines_distort_vs_undistort(orig_img, undist_img):
    """
    compare the straight lines drawn with OpenCV on both images i.e. orignal and undistorted
    """
    all_imgs = {"orig_img": orig_img, "undist_img": undist_img}
    for img_cat in list(all_imgs.keys()):
        img = all_imgs[img_cat]
        points = []
        print("provide 3 point to draw lines across them")
        while True:
            cv2.namedWindow(img_cat, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(img_cat, mouse_point,(img, points))
            cv2.imshow(img_cat, img)
            cv2.waitKey(1)
            if len(points) >= 3:
                cv2.destroyAllWindows()
                break

        for i in range(len(points)-1):
            cv2.line(img,points[i],points[i+1],color=(0,0,255),thickness=3)

        all_imgs[img_cat] = img

    cv2.namedWindow("Orignal Vs Undistorted", cv2.WINDOW_NORMAL)
    cv2.imshow("Orignal Vs Undistorted", np.hstack((all_imgs["orig_img"], all_imgs["undist_img"])))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    matrix_path = "logitech_old/camera_caliberation_values.npy"
    imgs_shape = (1024, 576)
    img_direct_path = ""
    cam_calib = CameraCalibration(calibration_matrix_path=matrix_path, imgs_shape=imgs_shape)

    # compare lines of normal and undistort image
    img_path = "logitech_old/images/1.jpg"
    im = cv2.imread(img_path)
    undist = cam_calib.undistort_frame(im)
    visualize_straight_lines_distort_vs_undistort(im, undist)
    #
    # # undistort a directory
    # cam_calib.undistort_images(input_path=img_direct_path)

