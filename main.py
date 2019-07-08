from lane_detection import helper
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
# from moviepy.editor import VideoFileClip
from os import path

img = mplimg.imread(path.abspath('test_images/lane.jpg'))
res_img = helper.process_an_image(img)

plt.figure()
plt.imshow(res_img)
plt.savefig(path.abspath('test_images/out.jpg'), bbox_inches='tight')
plt.show()

# output = '../resources/video_1_sol.mp4'
# clip = VideoFileClip("../resources/video_1.mp4")
# out_clip = clip.fl_image(helper.process_an_image)
# out_clip.write_videofile(output, audio=False)
