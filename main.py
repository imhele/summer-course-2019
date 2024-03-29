from lane_detection import v1, v2
# import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from moviepy.editor import VideoFileClip
import numpy
from os import path

settings = {
    'canny_lthreshold': 180,
    'canny_hthreshold': 240,
    'central_gap': 20,
    'correct': (256, 356),
    # 'draw_color': [255, 0, 0],
    # 'draw_thickness': 8,
    'max_line_gap': 32,
    'min_line_length': 45,
    'rho': 1,
    # 'save_dir': path.abspath('build'),
    # 'show_image': True,
    'threshold': 15,
    'roi_vtx': numpy.array([[(70, 356), (230, 256), (324, 256), (500, 356)]]),
    'extra_masks': (numpy.array([[(0, 350), (0, 334), (288, 334), (288, 350)]]),)
}

# img = VideoFileClip('D:/LaneLineDet/lane_v1.avi').get_frame(10.5)
# res_img = v1.pipeline(img, **settings)

clip = VideoFileClip('D:/LaneLineDet/lane_v1.avi').subclip(10, 30)
# clip.write_gif('C:/Users/me/Downloads/lane_v1.out_12_18.gif', fps=10, program='ffmpeg')
out_clip = clip.fl_image(lambda img: v1.pipeline(img, **settings))
out_clip.write_videofile(
    'C:/Users/me/Downloads/lane_v1.out_10_30.mp4', bitrate='10240k', audio=False, codec='mpeg4')
