import cv2
import functools
import matplotlib.pyplot as plt
import numpy
from os import path

current_frame = 0
prev_frame = 0
prev_lines = None
prev_vertices = None
prev_central_x = ()


def roi_mask(img, vertices, out=False):
    mask = numpy.zeros_like(img)
    # (255, 255, 255) or (255, 255, 255, 255)
    mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
    if out:
        mask.fill(255)
        mask_color = (0,) * img.shape[2] if len(img.shape) > 2 else 0
    cv2.fillPoly(mask, vertices, mask_color)
    return cv2.bitwise_and(img, mask)


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, central_gap,
                slope_range, correct=False, draw_color=[255, 0, 0], draw_thickness=8):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, numpy.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = numpy.zeros((img.shape[0], img.shape[1], 3), dtype=numpy.uint8)
    lines = prev_lines if lines is None else lines
    if lines is None:
        return line_img
    draw_lanes(line_img, lines, draw_color, draw_thickness,
               central_gap, slope_range, correct)
    return line_img


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def calc_central_points(left_points, right_points, points_len=8):
    n = 0
    while n < points_len:
        center_left = left_points[n][0] + right_points[n][0]
        center_right = left_points[n][1] + right_points[n][1]
        yield (center_left / 2, center_right / 2)
        n += 1


def draw_prev(img, color, thickness):
    if prev_lines is None or prev_vertices is None:
        return
    cv2.line(img, *prev_vertices[0], color, thickness)
    cv2.line(img, *prev_vertices[1], color, thickness)


def is_slope_in_range(vertices, range):
    slopes = [((l[1] - r[1]), (l[0] - r[0])) for l, r in vertices]
    slopes = [y / x if x else 0 for y, x in slopes]
    return abs(sum(slopes)) < range


def draw_lanes(img, lines, color, thickness, central_gap, slope_range, correct=None):
    if correct is None:
        return draw_lines(img, lines, color, thickness)

    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            (left_lines if k < 0 else right_lines).append(line)
    clean_lines(left_lines, 0.1), clean_lines(right_lines, 0.1)
    if len(left_lines) <= 0 or len(right_lines) <= 0:
        return draw_prev(img, color, thickness)
    left_points = [(x1, y1) for line in left_lines for x1, y1, *_ in line]
    left_points += [(x2, y2) for line in left_lines for *_, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, *_ in line]
    right_points += [(x2, y2) for line in right_lines for *_, x2, y2 in line]
    points_len = min(len(left_points), len(right_points))
    central_points = [p for p in calc_central_points(
        left_points, right_points, points_len)]
    central_points_y = [int(y) for x, y in central_points]
    if max(central_points_y) - min(central_points_y) > central_gap:
        return draw_prev(img, color, thickness)
    vertices = (calc_lane_vertices(
        left_points, *correct), calc_lane_vertices(right_points, *correct))
    if not is_slope_in_range(vertices, slope_range):
        return draw_prev(img, color, thickness)
    global_dicts = globals()
    global_dicts['prev_lines'] = lines
    global_dicts['prev_vertices'] = vertices
    global_dicts['prev_frame'] = current_frame
    avrg_central_y = sum(central_points_y) // points_len
    global_dicts['prev_central_x'] = (avrg_central_y,) + \
        prev_central_x[:9] if prev_central_x else (avrg_central_y,) * 10
    draw_prev(img, color, thickness)


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


def draw_central_mask(img, width):
    if not prev_central_x:
        return img
    left = (prev_central_x[0] - width, img.shape[0])
    right = (prev_central_x[0] + width, img.shape[0])
    vertices = numpy.array([[left, right, (right[0], 0), (left[0], 0)]])
    img = roi_mask(img, vertices, True)
    return img


def get_turnning_diff(turnning):
    if not prev_central_x:
        return (0,) * 4
    diff = prev_central_x[0] + prev_central_x[1]
    diff -= prev_central_x[8] + prev_central_x[9]
    diff = diff // 2
    if abs(diff) < turnning:
        return (0,) * 4
    if diff < 0:
        # turn right
        inner_diff = (0, diff)
        outer_diff = (diff, 0)
    else:
        # turn left
        inner_diff = (diff, 0)
        outer_diff = (0, diff)
    return inner_diff + outer_diff


def draw_outer_inner_masks(img, width, turnning):
    if prev_vertices is None:
        return img
    # init vars
    offset = int(width * 1.618)
    left_line, right_line = prev_vertices
    right_line = list(right_line)
    right_line.reverse()
    diff = get_turnning_diff(turnning)

    # inner
    inner = [(x + offset + diff[0], y) for x, y in left_line]
    inner += [(x - offset + diff[1], y) for x, y in right_line]
    if inner[3][0] < inner[0][0]:
        center = (inner[3][0] + inner[0][0] - diff[0] - diff[1]) // 2
        inner[0] = (center, inner[0][1])
        inner[3] = (center, inner[3][1])
    # outer
    outer = [(x - offset + diff[2], y) for x, y in left_line]
    outer += [(x + offset + diff[3], y) for x, y in right_line]
    if outer[3][0] < outer[0][0]:
        center = (outer[3][0] + outer[0][0] -  diff[2] - diff[3]) // 2
        outer[0] = (center - width, outer[0][1])
        outer[3] = (center + width, outer[3][1])
    outer.reverse()
    mask_points = inner + outer
    img = roi_mask(img, numpy.array([mask_points]))
    return img, mask_points


def draw_extra_masks(img, extra_masks):
    for mask in extra_masks:
        img = roi_mask(img, mask, True)
    return img


def draw_roi_masks(img, roi_vtx, central_gap, extra_masks, max_frame, turnning):
    # if current_frame - prev_frame > max_frame and roi_vtx is not None:
    #     roi_vtx = roi_vtx.copy()
    #     roi_vtx[0][0][0] = 0
    #     roi_vtx[0][1][0] = 0
    #     roi_vtx[0][2][0] = img.shape[1]
    #     roi_vtx[0][3][0] = img.shape[1]
    #     img = roi_mask(img, roi_vtx)
    # elif prev_vertices is not None:
    mask_points = None
    if prev_vertices is not None:
        img, mask_points = draw_outer_inner_masks(img, central_gap, turnning)
    elif roi_vtx is not None:
        img = draw_central_mask(img, central_gap)
        img = roi_mask(img, roi_vtx)
    if extra_masks:
        img = draw_extra_masks(img, extra_masks)
    return img, mask_points


DEFAULT_SETTINGS = {
    'blur_ksize': 5,
    'canny_lthreshold': 50,
    'canny_hthreshold': 150,
    'central_gap': 32,
    'correct': False,
    'draw_color': [255, 0, 0],
    'draw_thickness': 8,
    'extra_masks': (),
    'max_frame': 15,
    'max_line_gap': 20,
    'min_line_length': 40,
    'rho': 1,
    'roi_vtx': None,
    'save_dir': None,
    'show_image': False,
    'slope_range': 0.2,
    'theta': numpy.pi / 180,
    'threshold': 15,
    'turnning': 24,
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
    globals()['current_frame'] += 1
    globals()['temp'] = None
    sts = dict(DEFAULT_SETTINGS)
    sts.update(partial_settings)
    roi_vtx = sts['roi_vtx']
    save_dir = sts['save_dir']
    show_image = sts['show_image']
    blur_ksize = sts['blur_ksize']
    slope_range = sts['slope_range']
    if type(blur_ksize) is not tuple:
        blur_ksize = (blur_ksize, blur_ksize)
    canny_lthreshold = sts['canny_lthreshold']
    canny_hthreshold = sts['canny_hthreshold']
    roi_masks_args = (
        roi_vtx, sts['central_gap'], sts['extra_masks'], sts['max_frame'], sts['turnning'])
    hough_lines_args = (sts['rho'], sts['theta'], sts['threshold'], sts['min_line_length'],
                        sts['max_line_gap'], sts['central_gap'], sts['slope_range'],
                        sts['correct'], sts['draw_color'], sts['draw_thickness'])

    # process image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, blur_ksize, 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges, mask_points = draw_roi_masks(edges, *roi_masks_args)
    line_img = hough_lines(roi_edges, *hough_lines_args)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    if mask_points is not None:
        for i in range(1, len(mask_points)):
            cv2.line(res_img, mask_points[i - 1], mask_points[i], sts['draw_color'])

    if save_dir is not None:
        plt.figure()
        plt.imshow(img)
        plt.savefig(path.join(save_dir, 'origin.png'), bbox_inches='tight')

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

        if sts['show_image']:
            plt.show()

    return res_img
