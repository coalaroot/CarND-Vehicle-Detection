import pickle
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from lessons_functions import *


def process_image(image):
    boxes = find_cars(image, 400, 464, 1.0, color_space, hog_channel, svc,
                      orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 416, 480, 1.0, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 400, 496, 1.5, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 432, 528, 1.5, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 400, 528, 2.0, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 432, 560, 2.0, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 400, 596, 3.2, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)
    boxes += find_cars(image, 464, 680, 3.2, color_space, hog_channel, svc,
                       orient, pix_per_cell, cell_per_block)

    vehicles.add(boxes)

    heatmap_img = np.zeros_like(image[:, :, 0])

    for box in vehicles.prev_boxes:
        heatmap_img = add_heat(heatmap_img, box)

    heatmap_img = apply_threshold(heatmap_img, 1 + len(vehicles.prev_boxes) // 2)
    labels = label(heatmap_img)
    draw_img, rects = draw_labeled_boxes(np.copy(image), labels)

    return draw_img


pckl = pickle.load(open('../classifier.p', 'rb'))
svc = pckl['svc']

vehicles = Vehicles()

parsed_video = "../output_video.mp4"
clip1 = VideoFileClip("../project_video.mp4")
parsed_clip = clip1.fl_image(process_image)
parsed_clip.write_videofile(parsed_video, audio=False)


# test_out_file = '../test_video_out.mp4'
# clip_test = VideoFileClip('../test_video.mp4')
# clip_test_out = clip_test.fl_image(process_image)
# clip_test_out.write_videofile(test_out_file, audio=False)
