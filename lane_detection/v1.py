import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import path

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold
correct = True
# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20


def roi_mask(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, correct):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    try:
        if correct:
            draw_lanes(line_img, lines)
        else:
            draw_lines(line_img, lines)
    except TypeError:
        pass
    return line_img


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2)
                                 for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1)
                    for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + \
        [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])

    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1)
             for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


def pipeline(img, process=None, **params):
    """
    :param int blur_ksize: default to `5`  # Gaussian blur kernel size
    :param int canny_lthreshold: default to `50`  # Canny edge detection low threshold
    :param int canny_hthreshold: default to `150`  # Canny edge detection high threshold
    - Hough transform parameters
    :param int rho: default to `1`
    :param float theta: default to `np.pi / 180`
    :param int threshold: default to `15`
    :param int min_line_length: default to `40`
    :param int max_line_gap: default to `20`
    """
    global_dicts = globals()
    for key in params:
        global_dicts[key] = params[key]
    roi_vtx = params['roi_vtx'] if 'roi_vtx' in params else np.array(
        [[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta,
                           threshold, min_line_length, max_line_gap, correct)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    if process:
        plt.figure()
        plt.imshow(gray, cmap='gray')
        plt.savefig(path.join(process, 'gray.png'), bbox_inches='tight')
        plt.figure()
        plt.imshow(blur_gray, cmap='gray')
        plt.savefig(path.join(process, 'blur_gray.png'), bbox_inches='tight')
        plt.figure()
        plt.imshow(edges, cmap='gray')
        plt.savefig(path.join(process, 'edges.png'), bbox_inches='tight')
        plt.figure()
        plt.imshow(roi_edges, cmap='gray')
        plt.savefig(path.join(process, 'roi_edges.png'), bbox_inches='tight')
        plt.figure()
        plt.imshow(line_img, cmap='gray')
        plt.savefig(path.join(process, 'line_img.png'), bbox_inches='tight')
        plt.figure()
        plt.imshow(res_img)
        plt.savefig(path.join(process, 'res_img.png'), bbox_inches='tight')
        # plt.show()

    return res_img
