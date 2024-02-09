import csv
from genericpath import sameopenfile
import uuid
from PIL import Image
import os
import cv2
import json
import subprocess
import numpy as np
from skimage.feature import hog
from scipy.spatial import distance
import shutil
import time
import random

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 255  # Number of histogram bins

hist_feat = True # Histogram features on or off
hog_feat = False # HOG features on or off
spatial_feat = True # Spatial features on or off

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True: # Call with two outputs if vis==True to visualize the HOG
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      # Otherwise call with one output
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(16, 16)):
    return cv2.resize(img, size).ravel() 

# Define a function to compute color histogram features 
def color_hist(img, nbins=32):
    ch1 = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))[0]#We need only the histogram, no bins edges
    ch2 = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))[0]
    ch3 = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))[0]
    hist = np.hstack((ch1, ch2, ch3))
    return hist

def sift_compare(img1, img2):
    MIN_MATCH_COUNT = 10
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 90)
    #feature matching
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        # print(f"number of good matches is {len(good)}")
    except:
        return False
    return len(good)>MIN_MATCH_COUNT

def is_same_with_sift(frame_num, orientation_1, bounding_box_in_orientation_1, orientation_2, bounding_box_in_orientation_2, tmp_dir_to_use_for_comparison):
    
    output_dir = f"comparison_dir/{tmp_dir_to_use_for_comparison}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_cropped_image(get_frame_image(orientation_1, frame_num), bounding_box_in_orientation_1, f"{output_dir}/o1.jpg")
    create_cropped_image(get_frame_image(orientation_2, frame_num), bounding_box_in_orientation_2, f"{output_dir}/o2.jpg")
    img1 = cv2.imread(f"{output_dir}/o1.jpg")
    img2 = cv2.imread(f"{output_dir}/o2.jpg")
    return sift_compare(img1, img2)
    
    

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        use_flipped=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        file_features = []
        # NOTE: Assume cv2.imread has already been called. Remove if want to use raw images
#        image = cv2.imread(image) # Read in each imageone by one
        # apply color conversion if other than 'RGB'
#        if color_space != 'RGB':
#            if color_space == 'HSV':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#            elif color_space == 'LUV':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
#            elif color_space == 'HLS':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#            elif color_space == 'YUV':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
#            elif color_space == 'YCrCb':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
#        else: feature_image = np.copy(image)      

        feature_image = np.copy(image)    
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, spatial_size)
        features.append(np.concatenate(file_features))
        if use_flipped:
            feature_image=cv2.flip(feature_image,1) # Augment the dataset with flipped images
            file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                            pix_per_cell, cell_per_block, hog_channel, spatial_size)
            features.append(np.concatenate(file_features))
    return features # Return list of feature vectors

def img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, spatial_size):
    file_features = []
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #print 'spat', spatial_features.shape
        file_features.append(spatial_features)
    if hist_feat == True:
         # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #print 'hist', hist_features.shape
        file_features.append(hist_features)
    if hog_feat == True:
        input(f"getting hog features")
        hog_feature_image = np.copy(feature_image)
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(hog_feature_image.shape[2]):
                hog_features.append(get_hog_features(hog_feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
        else:
            hog_feature_image = cv2.cvtColor(hog_feature_image, cv2.COLOR_LUV2RGB)
            hog_feature_image = cv2.cvtColor(hog_feature_image, cv2.COLOR_RGB2GRAY)
            hog_features = get_hog_features(hog_feature_image[:,:], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                #print 'hog', hog_features.shape
            # Append the new feature vector to the features list

        file_features.append(hog_features)

    return file_features

def get_objects_dict_from_mot_file(mot_file_path):
    dict_to_create = {}
    with open(mot_file_path, "r") as f:
        s = csv.reader(f)
        for row in s:
            # row is an array. 0th element is frame num. for now we want only 0 and 6
            # if int(row[0]) > 6:
            #     continue
            frame_num = int(row[0])
            object_id = int(row[1])
            x0 = int(row[2])
            y0 = int(row[3])
            x1 = int(row[4])
            y1 = int(row[5])
            if frame_num not in dict_to_create:
                dict_to_create[frame_num] = {}
            dict_to_create[frame_num][object_id] = (x0, y0, x1, y1)
    return dict_to_create

def create_cropped_image(original_image, bounding_box, output_image):
    if os.path.isfile(output_image):
        return
    im = Image.open(original_image)
    # print(f"bounding box is {bounding_box}")
    im1 = im.crop(bounding_box)
    im1.save(output_image)

def get_euclidean_distance_between_hog_features(frame_num, orientation_1, bounding_box_in_orientation_1, orientation_2, bounding_box_in_orientation_2, tmp_dir_to_use_for_comparison):
    output_dir = f"comparison_dir/{tmp_dir_to_use_for_comparison}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_cropped_image(get_frame_image(orientation_1, frame_num), bounding_box_in_orientation_1, f"{output_dir}/o1.jpg")
    create_cropped_image(get_frame_image(orientation_2, frame_num), bounding_box_in_orientation_2, f"{output_dir}/o2.jpg")
    img1 = cv2.imread(f"{output_dir}/o1.jpg")
    img2 = cv2.imread(f"{output_dir}/o2.jpg")
    features_img1, features_img2 = extract_features([img1, img2])
    hog_distance = distance.euclidean(features_img1, features_img2) 
    return hog_distance

def get_frame_image(orientation, frame_num):
    # old style for small dataset
    return f"{orientation}/frames/frame{frame_num}.jpg"
    # return f"/disk/Code/projects/centroids-reid/mot/seattle-city/{orientation}/frame{frame_num}.jpg"

def are_two_objects_the_same(frame_num, orientation_1, bounding_box_in_orientation_1, orientation_2, bounding_box_in_orientation_2, tmp_dir_to_use_for_comparison):
    output_dir = f"comparison_dir/{tmp_dir_to_use_for_comparison}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_cropped_image(get_frame_image(orientation_1, frame_num), bounding_box_in_orientation_1, f"{output_dir}/o1.jpg")
    create_cropped_image(get_frame_image(orientation_2, frame_num), bounding_box_in_orientation_2, f"{output_dir}/o2.jpg")
    img1 = cv2.imread(f"{output_dir}/o1.jpg")
    img2 = cv2.imread(f"{output_dir}/o2.jpg")
    features_img1, features_img2 = extract_features([img1, img2])
    
    # https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL), # gives 1 for same img
	# ("Chi-Squared", cv2.HISTCMP_CHISQR), # close to 0 for same img. as it is distance
	# ("Intersection", cv2.HISTCMP_INTERSECT), # high number for same image
	 ("Hellinger", cv2.HISTCMP_BHATTACHARYYA) # gives 0 for same img
    ) 
    comparison_scores = {}
    for (methodName, method) in OPENCV_METHODS:
        comparison_scores[methodName] = cv2.compareHist(hist1, hist2, method)
    return comparison_scores

def dedup_two_orientations(orientation_1, orientation_2, mot_results_dir, frame_start=0, frame_end=6000):
    mot_results_1 = {} # frame to object to bounding box
    mot_results_2 = {} # frame to object to bounding box
    mot_results_1 = get_objects_dict_from_mot_file(f"{mot_results_dir}/{orientation_1}.txt")
    mot_results_2 = get_objects_dict_from_mot_file(f"{mot_results_dir}/{orientation_2}.txt")
    counter = int(time.time()) 
    frame_num = frame_start
    while frame_num <= frame_end:
    # for frame_num in range(1049):
        print(f"frame number {frame_num}")
        if frame_num not in mot_results_1 or frame_num not in mot_results_2:
            frame_num += 6
            continue
        objects_in_1 = mot_results_1[frame_num]
        objects_in_2 = mot_results_2[frame_num]
        object_ids_in_1 = list(objects_in_1.keys())
        object_ids_in_2 = list(objects_in_2.keys())
        
        random.shuffle(object_ids_in_1)
        random.shuffle(object_ids_in_2)
        for o1 in object_ids_in_1:
            for o2 in object_ids_in_2:
                tmp_dir_name = str(uuid.uuid4())
                comparison_scores = are_two_objects_the_same(frame_num, orientation_1, objects_in_1[o1], orientation_2, objects_in_2[o2], tmp_dir_name)
                hog_distance = get_euclidean_distance_between_hog_features(frame_num, orientation_1, objects_in_1[o1], orientation_2, objects_in_2[o2], tmp_dir_name)
                match_with_sift = is_same_with_sift(frame_num, orientation_1, objects_in_1[o1], orientation_2, objects_in_2[o2], tmp_dir_name)
                same_object = hog_distance < 20000 and is_same_with_sift and comparison_scores["Correlation"] > 0.97 and comparison_scores["Hellinger"] < 0.2 # these values have been chosen empirically. i manually checked the numbers and the images to see if they are the same for ~50 image pairs. 
                same_object = comparison_scores["Correlation"] > 0.97 and comparison_scores["Hellinger"] < 0.2
                if same_object:
                    print(f"i think the two objects are duplicates. comparison id: {tmp_dir_name}. and distance is {hog_distance}. sift: {match_with_sift}")
                    subprocess.Popen(["xdg-open", f"comparison_dir/{tmp_dir_name}"])
                    user_input = input(f"am i right (y/n)? here is my reasoningcomparison score is {json.dumps(comparison_scores, indent=2)} \n")
                    if user_input == "y":
                        output_dir = f"comparison_dir/{str(uuid.uuid4())}"
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        create_cropped_image(get_frame_image(orientation_1, frame_num), objects_in_1[o1], f"{output_dir}/o1.jpg")
                        create_cropped_image(get_frame_image(orientation_2, frame_num), objects_in_2[o2], f"{output_dir}/o2.jpg")
                        shutil.copy(f"{output_dir}/o1.jpg", f"/disk/Code/projects/mot-analysis-madeye/my_dataset/{counter}a.jpg")
                        shutil.copy(f"{output_dir}/o2.jpg", f"/disk/Code/projects/mot-analysis-madeye/my_dataset/{counter}b.jpg")
                        counter += 1
                    
                # else:
                #     print(f"objects are not same. distance is {hog_distance}. sift: {match_with_sift}")
            
        frame_num = frame_num + 6


def generating_ground_truth():
    # with small dataset
    o1 = "180-0-1"
    o2 = "180--30-1"
    mot_results = "/disk/Code/projects/mot-analysis-madeye/mot-results"
    # done 0 to 276. 480 to 606. 840 to 1230.
    dedup_two_orientations(o1, o2, mot_results, 840, 6200)
    
    # with seattle-city 1
    # o1 = "0-0-1"
    # o2 = "0--30-1"
    # mot_results = "/disk/Code/projects/centroids-reid/mot/seattle-city-mot-results/yolov4/0-6327-6"
    # done 0 to 280.1200 to 1266. 1296 to 1332. 6000 to 6237. 4026 to 4236.
    # dedup_two_orientations(o1, o2, mot_results, 4026, 6000)

    # with seattle-city 2
    # o1 = "60--30-1"
    # o2 = "60-0-1"
    # done frame 0 to 276. then 366 to 480. 600 to 624. 840 to 990. 1200 to 1206
    # 2400 to 2500. 2520 to 2520. 2640 to 2700. 2840 to 6237 .
    # dedup_two_orientations(o1, o2, mot_results, 2840, 6237)

def are_two_images_same_for_eval(img1_file, img2_file, is_neighbor=False):
    # TODO: if is_neighbor is True, lower the threshold for declaring objects the same (i.e. make it easier to mark objects as duplicates)
    # 1st is color histogram
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    
    # https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL), # gives 1 for same img
	# ("Chi-Squared", cv2.HISTCMP_CHISQR), # close to 0 for same img. as it is distance
	# ("Intersection", cv2.HISTCMP_INTERSECT), # high number for same image
	 ("Hellinger", cv2.HISTCMP_BHATTACHARYYA) # gives 0 for same img
    ) 
    comparison_scores = {}
    for (methodName, method) in OPENCV_METHODS:
        comparison_scores[methodName] = cv2.compareHist(hist1, hist2, method)
    
    same_object = comparison_scores["Correlation"] > 0.9 and comparison_scores["Hellinger"] < 0.3
    
    # # 2nd is color binning
    # features_img1, features_img2 = extract_features([img1, img2])
    # hog_distance = distance.euclidean(features_img1, features_img2)
    # same_object = same_object and hog_distance < 20000

    # break early if this comparison failed
    if not same_object:
        return False
    # # 3rd is sift
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    same_object = same_object and sift_compare(img1, img2)
    return same_object
    

def evaluate_performance():
    ground_truth_dataset = "/disk/Code/projects/mot-analysis-madeye/my_dataset"
    list_of_files = os.listdir(ground_truth_dataset)
    list_of_files = [f"/disk/Code/projects/mot-analysis-madeye/my_dataset/{x}" for x in list_of_files]
    list_of_objects = list(set(([x[:-5] for x in list_of_files])))
    print(f"number of objects is {len(list_of_objects)}")
    comparisons_to_do = list()
    for x in range(len(list_of_objects)):
        for y in range(len(list_of_objects)):
            if x != y:
                comparisons_to_do.append((x, y))

    comparisons_to_do = comparisons_to_do[:150]
    
    number_of_ground_truth_comparisons_that_are_different_objects = 0
    number_of_comparisons_we_detected_as_different = 0
    # evaluate if our algo correctly tells objects apart
    # all the comparison in this are between two different objects
    for comparisons in comparisons_to_do:
        obj1 = comparisons[0]
        obj2 = comparisons[1]
        file_path_1 = list_of_objects[obj1] + "a.jpg" # just compare the a versions of both
        file_path_2 = list_of_objects[obj2] + "a.jpg"
        is_same = are_two_images_same_for_eval(file_path_1, file_path_2)
        number_of_ground_truth_comparisons_that_are_different_objects += 1
        if not is_same:
            number_of_comparisons_we_detected_as_different += 1
    print(f"identified as different: {number_of_comparisons_we_detected_as_different}. actually different: {number_of_ground_truth_comparisons_that_are_different_objects}. ratio: {number_of_comparisons_we_detected_as_different/number_of_ground_truth_comparisons_that_are_different_objects}")

    identified_by_us_as_same = 0
    actually_same = 0
    for file in list_of_objects:
        actually_same += 1

        is_same = are_two_images_same_for_eval(file+"a.jpg", file+"b.jpg")
        if is_same:
            identified_by_us_as_same += 1

    print(f"identified as same: {identified_by_us_as_same}. actually same: {actually_same}. ratio: {identified_by_us_as_same/actually_same}")
    

if __name__ == "__main__":
    # generating_ground_truth()
    evaluate_performance()
