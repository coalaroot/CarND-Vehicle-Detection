**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extracted HOG features in `image_pipeline.py` in lines #22 & #25. I used function extract_features() from Udacity lessons that under the hood uses hog function from skimage.feature.

#### 2. Explain how you settled on your final choice of HOG parameters.

 I tried different values for orientations (8, 9, 10 and 11) and pixels_per_cell (8 and 16) and found the best combination of those. Later I was supposed to look closer at effect of cell_per_block parameter but I was happy enought with the results

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using my extracted features. After I got over 98% accuracy I saved it in a pickle. These are lines from #13 to #50 in file `image_pipeline.py`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I looked at some testing images and measured what size can a car appear to be in which parts of the frame and got a few scales and areas out of it. Here's how it looks like:

<src href="output_images/boxes_range.jpg" width="500"/>

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features. I tried to bring color histogram but had too many troubles and too little time to work this out. Here are some example images:


<src href="output_images/test1.jpg" width="500"/>
<src href="output_images/test2.jpg" width="500"/>
<src href="output_images/test3.jpg" width="500"/>
<src href="output_images/test4.jpg" width="500"/>
<src href="output_images/test5.jpg" width="500"/>
<src href="output_images/test6.jpg" width="500"/>


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


<src href="output_images/boxes_range.jpg" width="500"/>


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded (with threshold of 1) that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.


### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had problems with python mostly since I don't know it that well yet, mainly knowing what variable is what type and what can I do with it (i.e. problems with broadcasting because of shape incompatibility).
I could use color histograms and spacial binning to make my pipeline more accurate. This pipeline is most likely to fail in "not detecting cars it should".
