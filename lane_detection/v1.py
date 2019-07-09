import cv2
import matplotlib.pyplot as plt
import numpy
from os import path


def roi_mask(img, vertices):
    mask = numpy.zeros_like(img)
    # (255, 255, 255) or (255, 255, 255, 255)
    mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, mask_color)
    return cv2.bitwise_and(img, mask)


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,
                correct=False, draw_color=[255, 0, 0], draw_thickness=8):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, numpy.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = numpy.zeros((img.shape[0], img.shape[1], 3), dtype=numpy.uint8)
    if lines is None:
        return line_img
    if correct:
        draw_lanes(line_img, lines, draw_color, draw_thickness, correct)
    else:
        draw_lines(line_img, lines, draw_color, draw_thickness)
    return line_img


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lanes(img, lines, color, thickness, correct=(0, 0)):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            (left_lines if k < 0 else right_lines).append(line)
    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img
    clean_lines(left_lines, 0.1), clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, *_ in line]
    left_points += [(x2, y2) for line in left_lines for *_, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, *_ in line]
    right_points += [(x2, y2) for line in right_lines for *_, x2, y2 in line]

    cv2.line(img, *calc_lane_vertices(left_points, *correct), color, thickness)
    cv2.line(img, *calc_lane_vertices(right_points, *correct), color, thickness)


def clean_lines(lines, threshold):
    slope = [(d - b) / (c - a) for line in lines for a, b, c, d in line]
    while len(lines) > 0:
        mean = numpy.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = numpy.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = numpy.polyfit(y, x, 1)
    fit_fn = numpy.poly1d(fit)
    return [(int(fit_fn(ymin)), ymin), (int(fit_fn(ymax)), ymax)]


DEFAULT_SETTINGS = {
    'blur_ksize': 5,
    'canny_lthreshold': 50,
    'canny_hthreshold': 150,
    'correct': False,
    'draw_color': [255, 0, 0],
    'draw_thickness': 8,
    'max_line_gap': 20,
    'min_line_length': 40,
    'rho': 1,
    'roi_vtx': None,
    'save_dir': None,
    'show_image': False,
    'theta': numpy.pi / 180,
    'threshold': 15,
}


def pipeline(img, **partial_settings):
    """
    :param int blur_ksize: default to `5`  # Gaussian blur kernel size
    :param int canny_lthreshold: default to `50`  # Canny edge detection low threshold
    :param int canny_hthreshold: default to `150`  # Canny edge detection high threshold
    - Hough transform parameters
    :param int rho: default to `1`
    :param float theta: default to `numpy.pi / 180`
    :param int threshold: default to `15`
    :param int min_line_length: default to `40`
    :param int max_line_gap: default to `20`
    """
    settings = dict(DEFAULT_SETTINGS)
    settings.update(partial_settings)
    roi_vtx = settings['roi_vtx']
    save_dir = settings['save_dir']
    show_image = settings['show_image']
    blur_ksize = settings['blur_ksize']
    if type(blur_ksize) is not tuple:
        blur_ksize = (blur_ksize, blur_ksize)
    canny_lthreshold = settings['canny_lthreshold']
    canny_hthreshold = settings['canny_hthreshold']
    hough_lines_args = (settings['rho'], settings['theta'], settings['threshold'],
                        settings['min_line_length'], settings['max_line_gap'],
                        settings['correct'], settings['draw_color'], settings['draw_thickness'])

    # process image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, blur_ksize, 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx) if roi_vtx is not None else edges
    line_img = hough_lines(roi_edges, *hough_lines_args)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    if save_dir is not None:
        plt.figure()
        plt.imshow(gray, cmap='gray')
        plt.savefig(path.join(save_dir, 'gray.png'), bbox_inches='tight')

        plt.figure()
        plt.imshow(blur_gray, cmap='gray')
        plt.savefig(path.join(save_dir, 'blur_gray.png'), bbox_inches='tight')

        plt.figure()
        plt.imshow(edges, cmap='gray')
        plt.savefig(path.join(save_dir, 'edges.png'), bbox_inches='tight')

        plt.figure()
        plt.imshow(roi_edges, cmap='gray')
        plt.savefig(path.join(save_dir, 'roi_edges.png'), bbox_inches='tight')

        plt.figure()
        plt.imshow(line_img, cmap='gray')
        plt.savefig(path.join(save_dir, 'line_img.png'), bbox_inches='tight')

        plt.figure()
        plt.imshow(res_img)
        plt.savefig(path.join(save_dir, 'res_img.png'), bbox_inches='tight')

        if settings['show_image']:
            plt.show()

    return res_img
