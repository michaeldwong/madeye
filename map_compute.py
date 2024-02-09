import tempfile
import subprocess
import uuid
import pandas as pd
import copy
import csv
import sys
import os
import evaluation_tools

from madeye_utils import parse_orientation_string, extract_pan, extract_tilt, extract_zoom, find_tilt_dist, find_pan_dist
from main import generate_all_orientations
from main import CAR_CONFIDENCE_THRESH, PERSON_CONFIDENCE_THRESH, MODELS, SKIP
from dedupe_helper import are_two_images_same_for_eval, create_cropped_image

from DetectedObject import DetectedObject

tmpdirname = tempfile.TemporaryDirectory()

def read_cache(map_cache_file, model):
    try:
        with open(map_cache_file) as f:
            csv_data = csv.reader(f)
            map_cache = {}
            for row in csv_data:
                current_model = row[0]
                if model == current_model:
                    frame = int(row[1])
                    orientation = row[2]
                    obj = row[3]
                    map_val = float(row[4])
                    map_cache[(frame, orientation, obj)] = map_val
            return map_cache
    except:
        return {}

def write_cache(model: str, frame: int, orientation: str, type: str, result: float, map_cache_file: str):
    if len(map_cache_file) == 0:
        return
    try:
        with open(map_cache_file, "a") as file:
            file.write(f"{model},{frame},{orientation},{type},{result}\n")
    except Exception as e:
        print(f"Couldn't write to file: {e}")


def parse_map_output(output: str) -> float:
    string_output = output.split("\n")[0].split("=")[0]
    try:
        return float(string_output)
    except ValueError:
        raise ValueError(f"Failure: invalid mAP score: {output}")


def get_representative_bounding_box_for_each_object(orientation_to_list_of_detections):
    obj_id_to_representative_obj = {}
    obj_id_to_orientation_of_representative_box = {}
    obj_id_to_area_of_representative_box = {}
    
    for o in orientation_to_list_of_detections: # go through each orientation
        for det in orientation_to_list_of_detections[o]: # go through each DetectedObject in each orientation
            if det.obj_id not in obj_id_to_area_of_representative_box:
                obj_id_to_area_of_representative_box[det.obj_id] = det.area()
                obj_id_to_orientation_of_representative_box[det.obj_id] = o
                obj_id_to_representative_obj[det.obj_id] = det
            else:
                if obj_id_to_area_of_representative_box[det.obj_id] < det.area():
                    obj_id_to_area_of_representative_box[det.obj_id] = det.area()
                    obj_id_to_orientation_of_representative_box[det.obj_id] = o
                    obj_id_to_representative_obj[det.obj_id] = det

    return obj_id_to_representative_obj

# gets the set of ground truth bounding boxes from an inference file
# an inference file is specific to: {frame, orientation, model}
def retrieve_detections_from_inference_file(inference_file, frame, orientation, model):
    if os.stat(inference_file).st_size == 0:
        print(inference_file, ' is empty')
        return 0,0
    inference_df = pd.read_csv(inference_file)
    car_detections = []
    person_detections = []
    for idx, row in inference_df.iterrows():
        if row['class'] == 'car' and row['confidence'] >= CAR_CONFIDENCE_THRESH:
            obj = DetectedObject(row['left'], row['top'], row['right'], row['bottom'], row['class'], row['confidence'], -1, orientation, frame, model)
            if obj.is_zoomed_in():
                obj.zoom_out_bounding_box()
            car_detections.append(obj)
        if row['class'] == 'person' and row['confidence'] >= PERSON_CONFIDENCE_THRESH:
            obj = DetectedObject(row['left'], row['top'], row['right'], row['bottom'], row['class'], row['confidence'], -1, orientation, frame, model)
            if obj.is_zoomed_in():
                obj.zoom_out_bounding_box()
            person_detections.append(obj)
    return car_detections, person_detections

def load_objects_from_inference_file(inference_frames_dir, frame, model, type_of_objects, orientations):
    # /scratch/mdwong/inference-frames/seattle-traffic/yolov4/270-0-2/frame2104.csv"
    orientation_to_detections = {}
    for orientation in orientations:
        inference_file_path = os.path.join(inference_frames_dir, model, orientation, f"frame{frame}.csv")
        car_detections, person_detections = retrieve_detections_from_inference_file(inference_file_path, frame, orientation, model)
        all_detections = []
        if type_of_objects == "car":
            all_detections = car_detections
        elif type_of_objects == "person":
            all_detections = person_detections
        else: # must be "both"
            all_detections = car_detections
            all_detections.extend(person_detections)
        orientation_to_detections[orientation] = all_detections
    return orientation_to_detections

def is_same_detected_object(obj1, obj2, rectlinear_dir):
    try:
        if obj1.type != obj2.type:
            print(f"different object types")
            return False
        bounding_box_1 = (obj1.left, obj1.top, obj1.right, obj1.bottom)
        bounding_box_2 = (obj2.left, obj2.top, obj2.right, obj2.bottom)
        source_image_1 = os.path.join(rectlinear_dir, obj1.orientation, f"frame{obj1.frame_num}.jpg")
        source_image_2 = os.path.join(rectlinear_dir, obj2.orientation, f"frame{obj2.frame_num}.jpg")
        is_same_image = False
        orientation_of_object_1 = obj1.orientation
        orientation_of_object_2 = obj2.orientation
        is_far_away = find_pan_dist(extract_pan(orientation_of_object_1), extract_pan(orientation_of_object_2)) >= 90
        if is_far_away: # orientations that are too far away cannot yield duplicates
            return False
        is_neighbor = True # TODO: later, set this correctly based on the two orientations above
        cropped_image_1 = os.path.join(tmpdirname.name, str(hash(f"{source_image_1}_{bounding_box_1}"))+".jpg")
        create_cropped_image(source_image_1, bounding_box_1, cropped_image_1)
        cropped_image_2 = os.path.join(tmpdirname.name, str(hash(f"{source_image_2}_{bounding_box_2}"))+".jpg")
        create_cropped_image(source_image_2, bounding_box_2, cropped_image_2)
        is_same_image = are_two_images_same_for_eval(cropped_image_1, cropped_image_2, is_neighbor)
        # uncomment this to get a chance to pause everything and check if dedupe is doing the right thing
        # input(f"returning {is_same_image} for {cropped_image_1} and {cropped_image_2}")
        return is_same_image
    except Exception as e:
        print("the following exception was caught when trying to determine if two objects were the same")
        print(e)
        return False

def dedupe_and_assign_id_to_objects(orientation_to_objects, rectlinear_dir):
    # reset all object ids in input so they're all -1 (uninitialized)
    for orientation in orientation_to_objects:
        for i in range(len(orientation_to_objects[orientation])):
            orientation_to_objects[orientation][i].obj_id = -1

    orientations = list(orientation_to_objects.keys())
    
    # the indices of items in representative_objects and bins are correlated
    # representative_objects[0] would have a representative object whose
    # corresponding bin is in bins[0]. the representative object itself would 
    # also be a part of it's bin. 
    
    representative_objects = [] 
    bins = [] # list of lists. each item is a bin. 
    
    for o in orientations:
        for obj in orientation_to_objects[o]:
            is_object_newly_discovered = True
            for ind, rep_obj in enumerate(representative_objects):
                if is_same_detected_object(rep_obj, obj, rectlinear_dir): # would call dedupe code
                    is_object_newly_discovered = False
                    obj.obj_id = rep_obj.obj_id
                    bins[ind].append(obj)
                    if rep_obj.area() < obj.area():
                        representative_objects[ind] = obj
                    break
            if is_object_newly_discovered:
                bins.append([obj]) # new_bin is inserted at new_bin_index_number
                obj.obj_id = uuid.uuid4() # this would be the unique id for this object and it's duplicates going forward
                representative_objects.append(obj)

def write_objects_to_disk_for_map_score_generation(list_of_objects, file_to_write_to, include_confidence_score=False):
    # print('writing to ', file_to_write_to)
    # print('list of objs ', list_of_objects)
    with open(file_to_write_to, "w") as f:
        for obj in list_of_objects:
            if include_confidence_score:
                f.write(f"{obj.type} {obj.confidence} {obj.left} {obj.top} {obj.right} {obj.bottom}\n")
            else:
                f.write(f"{obj.type} {obj.left} {obj.top} {obj.right} {obj.bottom}\n")

def does_intersect(obj1, obj2):
    x1_intersects = obj1.left <= obj2.left and obj1.right >= obj2.left
    y1_intersects = obj1.top <= obj2.top and obj1.bottom >= obj2.top
    x2_intersects = obj1.left <= obj2.right and obj1.right >= obj2.right
    y2_intersects = obj1.top <= obj2.bottom and obj1.bottom >= obj2.bottom
    top_left_intersects = x1_intersects and y1_intersects
    bot_right_intersects = x2_intersects and y2_intersects
    top_right_intersects = x2_intersects and y1_intersects
    bot_left_intersects = x1_intersects and y2_intersects
    return top_left_intersects or bot_right_intersects or top_right_intersects or bot_left_intersects

def find_intersection(obj1, obj2):
    x_inter1 = max(obj1.left, obj2.left)
    y_inter1 = max(obj1.top, obj2.top)
    x_inter2 = min(obj1.right, obj2.right)
    y_inter2 = min(obj1.bottom, obj2.bottom)
    return [x_inter1, y_inter1, x_inter2, y_inter2]

def iou(obj1, obj2):
    if not does_intersect(obj1, obj2) and not does_intersect(obj2, obj1):
        return 0.0
    a1 = obj1.area()
    a2 = obj2.area()
    inter_points = find_intersection(obj1, obj2)

    e_inter = DetectedObject(inter_points[0], inter_points[1], inter_points[2], inter_points[3], None, 0.0, -1, None, 0, 'None')
    area_of_intersection = e_inter.area()
    return area_of_intersection  / (a1 + a2 - area_of_intersection)

def compute_map(frame_begin, 
                frame_limit, 
                inference_frames_dir, 
                map_dir, 
                map_cache_file, 
                rectlinear_dir, 
                type_of_objects, 
                orientations,
                frame_to_model_to_orientation_to_car_map,
                frame_to_model_to_orientation_to_person_map,
                frame_to_model_to_orientation_to_cars_detected,
                frame_to_model_to_orientation_to_people_detected
                ):
    sys.path.append(map_dir)
    from map_computer import main_map_compute
    for model in MODELS:
        f = frame_begin
        m = model
        # if len(existing_map_cache) > 0: 
        #     print(f"not regenerating map cache for {model} as it is already on disk at {map_cache_file}")
        #     continue # map cache is already computed, so i am exiting without doing anything.
            # alert: this is a potentially incorrect assumption: map cache could be partly populated. 

        while f <= frame_limit:
            if f % SKIP != 0:
                f += 1
                continue
            if type_of_objects == 'car' and f in frame_to_model_to_orientation_to_car_map and m in frame_to_model_to_orientation_to_car_map[f]:
                if len(frame_to_model_to_orientation_to_car_map[f][m]) >=  len(orientations):
                    f += 1
                    continue
            if type_of_objects == 'person' and f in frame_to_model_to_orientation_to_person_map and m in frame_to_model_to_orientation_to_person_map[f]:
                if len(frame_to_model_to_orientation_to_person_map[f][m]) >=  len(orientations):
                    f += 1
                    continue
            print(f"load inference info from disk")
#            orientation_to_objects = load_objects_from_inference_file(inference_frames_dir, f, model, type_of_objects, orientations)
            if type_of_objects == 'car':
                orientation_to_objects = frame_to_model_to_orientation_to_cars_detected[f][model]
            else:
                orientation_to_objects = frame_to_model_to_orientation_to_people_detected[f][model]

            if len(list(orientation_to_objects.values())) == 0:
                # No objects in the scene
                for o in orientation_to_objects:
                    map_score = 0.0 
                    if type_of_objects == 'car':  
                        if f not in frame_to_model_to_orientation_to_car_map:
                            frame_to_model_to_orientation_to_car_map[f] = {}
                        if m not in frame_to_model_to_orientation_to_car_map[f]:
                            frame_to_model_to_orientation_to_car_map[f][m] = {}
                        frame_to_model_to_orientation_to_car_map[f][model][o] = map_score
                    else:
                        if f not in frame_to_model_to_orientation_to_person_map:
                            frame_to_model_to_orientation_to_person_map[f] = {}
                        if m not in frame_to_model_to_orientation_to_person_map[f]:
                            frame_to_model_to_orientation_to_person_map[f][m] = {}
                        frame_to_model_to_orientation_to_person_map[f][model][o] = map_score
                    write_cache(model, f, o, type_of_objects, map_score, map_cache_file);
                f += 1
                continue

            print(f"deduplicating objects frame ", f, " model ", model)
            dedupe_and_assign_id_to_objects(orientation_to_objects, rectlinear_dir)
            print(f"finding representatives for each object")
            object_id_to_representative_obj = get_representative_bounding_box_for_each_object(orientation_to_objects)


            if len(list(object_id_to_representative_obj.values())) == 0:
                # No objects in the scene
                for o in orientation_to_objects:
                    map_score = 0.0 
                    if type_of_objects == 'car':  
                        if f not in frame_to_model_to_orientation_to_car_map:
                            frame_to_model_to_orientation_to_car_map[f] = {}
                        if m not in frame_to_model_to_orientation_to_car_map[f]:
                            frame_to_model_to_orientation_to_car_map[f][m] = {}
                        frame_to_model_to_orientation_to_car_map[f][model][o] = map_score
                    else:
                        if f not in frame_to_model_to_orientation_to_person_map:
                            frame_to_model_to_orientation_to_person_map[f] = {}
                        if m not in frame_to_model_to_orientation_to_person_map[f]:
                            frame_to_model_to_orientation_to_person_map[f][m] = {}
                        frame_to_model_to_orientation_to_person_map[f][model][o] = map_score
                    write_cache(model, f, o, type_of_objects, map_score, map_cache_file);
                f += 1
                continue


            object_id_to_representative_obj_backup = copy.deepcopy(object_id_to_representative_obj)
            max_number_of_objects_per_orientation = max([len(orientation_to_objects[o]) for o in orientation_to_objects])
            number_of_far_away_objects_to_retain_in_ground_truth = int(1.7 * max_number_of_objects_per_orientation)
            for o in orientation_to_objects:

                if type_of_objects == 'car':
                    if f not in frame_to_model_to_orientation_to_car_map:
                        frame_to_model_to_orientation_to_car_map[f] = {}
                    if m not in frame_to_model_to_orientation_to_car_map[f]:
                        frame_to_model_to_orientation_to_car_map[f][m] = {}
                    if o in frame_to_model_to_orientation_to_car_map[f][m]:
                        continue
                if type_of_objects == 'person':
                    if f not in frame_to_model_to_orientation_to_person_map:
                        frame_to_model_to_orientation_to_person_map[f] = {}
                    if m not in frame_to_model_to_orientation_to_person_map[f]:
                        frame_to_model_to_orientation_to_person_map[f][m] = {}
                    if o in frame_to_model_to_orientation_to_person_map[f][m]:
                        continue
                # now for each orientation, we're going to compute a map score

                # restore ground truth to original values for each orientation
                object_id_to_representative_obj = copy.deepcopy(object_id_to_representative_obj_backup)
                has_object_in_ground_truth_been_moved = {}
                for obj_id in object_id_to_representative_obj:
                    has_object_in_ground_truth_been_moved[obj_id] = False

                # for each object in the current orientation, recenter it's ground truth
                for obj in orientation_to_objects[o]:
                    assert(obj.obj_id in object_id_to_representative_obj) # something is very wrong if we encounter an object for which we have not yet determined the representative
                    
                    # don't recenter a ground truth multiple times as the operation is not idempotent. 
                    # this check will never fail outside of test settings as an orientation cannot have an object more than once.
                    if not has_object_in_ground_truth_been_moved[obj.obj_id]: 
                        # Note: use overlay to coincide the left top corners of both
                        # boxes instead of centering one on another. if needed.
                        # object_id_to_representative_obj[obj.obj_id].overlay_on_another_object(obj)
                        object_id_to_representative_obj[obj.obj_id].recenter_on_another_object(obj)
                        has_object_in_ground_truth_been_moved[obj.obj_id] = True
                
                number_of_objects_in_ground_truth_outside_current_orientation = [has_object_in_ground_truth_been_moved[obj_id] for obj_id in object_id_to_representative_obj].count(False)
                number_of_far_away_objects_to_keep = number_of_far_away_objects_to_retain_in_ground_truth - number_of_objects_in_ground_truth_outside_current_orientation
                # print(f"number_of_objects_in_ground_truth_outside_current_orientation: {number_of_objects_in_ground_truth_outside_current_orientation}")
                # print(f"number_of_far_away_objects_to_retain_in_ground_truth: {number_of_far_away_objects_to_retain_in_ground_truth}")
                # print(f"number_of_far_away_objects_to_keep: {number_of_far_away_objects_to_keep}")
                # we have too many objects in ground truth
                keys_to_delete = []
#                for obj_id1 in object_id_to_representative_obj:
#
#                    if len(object_id_to_representative_obj) - len(keys_to_delete) <= number_of_far_away_objects_to_keep:
#                        continue
#                    # find which elements in ground truth are outside current orientation
#                    if not has_object_in_ground_truth_been_moved[obj_id]:
#                        for obj_id2 in object_id_to_representative_obj:
#                            if obj_id1 == obj_id2 or obj_id1 in keys_to_delete:
#                                continue
#                            if iou(object_id_to_representative_obj[obj_id1], object_id_to_representative_obj[obj_id2]) >= 0.5:
# 
#                                # and delete them from the ground truth
#                                keys_to_delete.append(obj_id1)
#                print('Num of keus removing ', len(keys_to_delete))

                for obj_id in object_id_to_representative_obj:
                    # find which elements in ground truth are outside current orientation
                    if not has_object_in_ground_truth_been_moved[obj_id] and number_of_far_away_objects_to_keep < 0 and obj_id not in keys_to_delete:
                        # and delete them from the ground truth
                        keys_to_delete.append(obj_id)
                        number_of_far_away_objects_to_keep += 1



                # print(f"number_of_far_away_objects_to_keep: {number_of_far_away_objects_to_keep}")
                # print(f"len(object_id_to_representative_obj): {len(object_id_to_representative_obj)}")
                for key in keys_to_delete:
                    del has_object_in_ground_truth_been_moved[key]
                    del object_id_to_representative_obj[key]
                            
                
                # if number_of_far_away_objects_to_keep < 0:
                #     print("too many objects in ground truth. unable to fix.")

                # print(f"number_of_far_away_objects_to_keep after removal: {number_of_far_away_objects_to_keep}")
                # print(f"len(object_id_to_representative_obj): {len(object_id_to_representative_obj)}")
                # now we'll hopefully have the right number of objects in 
                # the ground truth, so let us now modify those outside the current orientation
                # for anything in the ground truth that hasn't been recented
                # we move those far away as the object isn't present in current orientation
                for obj_id in object_id_to_representative_obj:
                    if not has_object_in_ground_truth_been_moved[obj_id]:
                        # any object not yet recentered does not occur in the current orientation o
                        # so let us move it far away from the picture
                        object_id_to_representative_obj[obj_id].move_far_away()
                # print(f"after moving far away len(object_id_to_representative_obj): {len(object_id_to_representative_obj)}")
                # input(f"number of ground truth values is {len(list(object_id_to_representative_obj.values()))}")
                # let us now write the recentered ground truth to disk
                # write_objects_to_disk_for_map_score_generation(list(object_id_to_representative_obj.values()), f"{map_dir}/input/ground-truth/madeye.txt")
                
                # and also the objects in the current orientation
                # write_objects_to_disk_for_map_score_generation(orientation_to_objects[o], f"{map_dir}/input/detection-results/madeye.txt", include_confidence_score=True)
                ground_truth_for_map_cache = list(object_id_to_representative_obj.values())
                detection_results_for_map_cache = orientation_to_objects[o]
                result = main_map_compute(ground_truth_for_map_cache, detection_results_for_map_cache)
                map_score = parse_map_output(result)
#                if map_score > 0.0:
                    # TODO: this is for debug purposes. 
                    # remove in prod. 
#                    if map_score > 0.1:
#                        print(f'ground truth file {map_dir}/input/ground-truth/madeye.txt')
#                        print(f'current file {map_dir}/input/detection-results/madeye.txt')
#                        input('ENter to continue') 
#                    print(map_score)
#                    input(f'Check files -- frame {f} orientation {o}')
               
                if type_of_objects == 'car':  
                    frame_to_model_to_orientation_to_car_map[f][model][o] = map_score
                else:
                    frame_to_model_to_orientation_to_person_map[f][model][o] = map_score
                write_cache(model, f, o, type_of_objects, map_score, map_cache_file);
            f += 1
#    tmpdirname.cleanup()

def generate_all_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    for r1 in range(0,120,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            for z in [1,2,3]:
                orientations.append(f'{r1}-{r2}-{z}')
    return orientations
if __name__ == "__main__":
    map_file = f"map-cache-testing.csv"
    inference_frames_dir = "/disk2/mdwong/inference-results/seattle-city/"
    (frame_begin, frame_limit) = (1,161)
    map_dir = "/home/mikewong/Documents/mAP-copy-for-murali"
    rectlinear_dir = "/disk2/mdwong/rectlinear-output/seattle-city"
    type_of_obj = "person"
    compute_map(frame_begin, frame_limit, inference_frames_dir, map_dir, map_file, rectlinear_dir, type_of_obj, generate_all_orientations(), {}, {})

