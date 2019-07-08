from lane_detection import v1, v2
# import matplotlib.pyplot as plt
# import matplotlib.image as mplimg
from moviepy.editor import VideoFileClip
import numpy as np
from os import path

settings = {
    'correct': False,
    # 'blur_ksize': 3,  # Gaussian blur kernel size
    # 'canny_lthreshold': 50,  # Canny edge detection low threshold
    # 'canny_hthreshold': 200,  # Canny edge detection high threshold
    # 'rho': 3,
    # 'theta': np.pi / 180,
    # 'threshold': 20,
    # 'min_line_length': 3,
    # 'max_line_gap': 40,
    'roi_vtx': np.array(
        [[(0, 210), (225, 110), (285, 110), (480, 210)]])
}
img = VideoFileClip('C:/Users/me/Downloads/lane_v2.avi').get_frame(1)
res_img = v1.pipeline(img, path.abspath('build'), **settings)

# plt.figure()
# plt.imshow(res_img)
# plt.savefig(path.abspath('test_images/out.jpg'), bbox_inches='tight')
# plt.show()

clip = VideoFileClip('C:/Users/me/Downloads/lane_v2.avi').subclip(10, 15)
out_clip = clip.fl_image(lambda img: v1.pipeline(img, **settings))
out_clip.write_videofile(
    'C:/Users/me/Downloads/lane_v2.out_10_15.avi', audio=False, codec='rawvideo')
