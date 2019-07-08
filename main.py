from lane_detection import v1, v2
import matplotlib.pyplot as plt
# import matplotlib.image as mplimg
from moviepy.editor import VideoFileClip
from os import path

img = VideoFileClip('C:/Users/me/Downloads/lane_v2.avi').get_frame(0)
res_img = v1.pipeline(img, path.abspath('build'))

# plt.figure()
# plt.imshow(res_img)
# plt.savefig(path.abspath('test_images/out.jpg'), bbox_inches='tight')
# plt.show()

# clip = VideoFileClip('C:/Users/me/Downloads/lane_v2.avi').subclip(0, 10)
# out_clip = clip.fl_image(v1.pipeline)
# out_clip.write_videofile('C:/Users/me/Downloads/lane_v2.out.avi', audio=False, codec='rawvideo')
