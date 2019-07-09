from lane_detection import v1, v2
# import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from moviepy.editor import VideoFileClip
import numpy as np
from os import path

settings = {
    'canny_lthreshold': 200,
    'canny_hthreshold': 250,
    'correct': (256, 328),
    # 'draw_color': [255, 0, 0],
    # 'draw_thickness': 8,
    'max_line_gap': 16,
    'min_line_length': 45,
    'rho': 1,
    # 'save_dir': path.abspath('build'),
    # 'show_image': False,
    'threshold': 15,
    'roi_vtx': np.array([[(100, 328), (230, 256), (342, 256), (482, 328)]]),
}

# img = VideoFileClip('D:/LaneLineDet/lane_v1.avi').get_frame(12)
# res_img = v1.pipeline(img, path.abspath('build'), **settings)

# plt.figure()
# plt.imshow(res_img)
# plt.savefig(path.abspath('test_images/out.jpg'), bbox_inches='tight')
# plt.show()

clip = VideoFileClip('D:/LaneLineDet/lane_v1.avi').subclip(10, 15)
out_clip = clip.fl_image(lambda img: v1.pipeline(img, **settings))
out_clip.write_videofile(
    'C:/Users/me/Downloads/lane_v1.out_10_15.avi', audio=False, codec='rawvideo')
