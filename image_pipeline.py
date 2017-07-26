import glob
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from lessons_functions import *

# # Read in cars and notcars
# cars = glob.glob('vehicles/**/*.png')
# notcars = glob.glob('non-vehicles/**/*.png')
# # print(len(cars), len(notcars))
#
# car_features = extract_features(cars, cspace=color_space, orient=orient,
#                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                                 hog_channel=hog_channel)
# notcar_features = extract_features(notcars, cspace=color_space, orient=orient,
#                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                                    hog_channel=hog_channel)
#
# # Create an array stack of feature vectors
# X = np.vstack((car_features, notcar_features)).astype(np.float64)
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
#
# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
#
# print(orient, 'orientations', pix_per_cell,
#       'pixels per cell and', cell_per_block, 'cells per block')
# print('Feature vector length:', len(X_train[0]))
#
# svc = LinearSVC()  # Use a linear SVC
# svc.fit(X_train, y_train)  # Train the classifier
#
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
#
# # dst_pickle = {'svc': svc}
# # pickle.dump(dst_pickle, open('classifier.p', 'wb'))
#
pckl = pickle.load(open('classifier.p', 'rb'))
svc = pckl['svc']

boxes = []
# for image_p in glob.glob('test_images/test*.jpg'):
#     image = mpimg.imread(image_p)
#     draw_image = np.copy(image)
#
#     boxes = find_cars(image, 400, 464, 1.0, color_space, hog_channel, svc,
#                       orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 416, 480, 1.0, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 400, 496, 1.5, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 432, 528, 1.5, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 400, 528, 2.0, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 432, 560, 2.0, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 400, 596, 3.2, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#     boxes += find_cars(image, 464, 680, 3.2, color_space, hog_channel, svc,
#                        orient, pix_per_cell, cell_per_block)
#
#     heatmap_img = np.zeros_like(image[:, :, 0])
#     heatmap_img = add_heat(heatmap_img, boxes)
#     heatmap_img = apply_threshold(heatmap_img, 1)
#     labels = label(heatmap_img)
#     draw_img, rects = draw_labeled_boxes(np.copy(image), labels)
print("Done.")

image = mpimg.imread('test_images/test1.jpg')

boxes += find_cars(image, 400, 464, 1.0, color_space, hog_channel, svc,
                  orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 416, 480, 1.0, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 400, 496, 1.5, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 432, 528, 1.5, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 400, 528, 2.0, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 432, 560, 2.0, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 400, 596, 3.2, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)
boxes += find_cars(image, 464, 680, 3.2, color_space, hog_channel, svc,
                   orient, pix_per_cell, cell_per_block, show_all_rectangles=True)

draw_img2 = np.copy(image)
res = draw_boxes(draw_img2, boxes, color=(255, 0, 0), thick=2)
cv2.imwrite("output_images/boxes_range.jpg", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
