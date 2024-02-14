import pandas as pd
import random
import json
import os
import argparse
import mot_helper
import time

import yaml

import madeye
import map_compute
import evaluation_tools
import pickle 
import misc_results 

import baselines
from madeye_utils import parse_orientation_string, extract_pan, extract_tilt, extract_zoom, find_tilt_dist, find_pan_dist
from DetectedObject import DetectedObject
MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']
# seattle--dt-2 part 1

#FRAME_BOUNDS = [ (2753, 4009), (5648, 8434), (8435, 10429), (10430, 11988), (11989, 13146), (14443, 15687) ]
FRAME_BOUNDS = [   (8435, 10429), (10430, 11988),  (14443, 15687) ]



#FRAME_BOUNDS = [  (0, 1182), (1183, 1662),  (6571, 7752), (9693, 11300),  (24301, 26000)]

FRAME_BOUNDS = [ (1, 1161), (1664, 2823), (2824, 3966), (3967, 4983),(4984, 6075), (6076, 7194) ]

FRAME_BOUNDS = [ (1, 1161), (1664,2823), (2824, 3966)]
SKIP = 6 # Only consider frames where frame % SKIP == 0
PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0



def inflate_map_scores(frame_begin,
                       frame_limit,
                       orientations,
                       frame_to_model_to_orientation_to_car_count, 
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_cars_detected, 
                       frame_to_model_to_orientation_to_people_detected,
                       frame_to_model_to_orientation_to_car_map, 
                       frame_to_model_to_orientation_to_person_map):
    for f in range(frame_begin, frame_limit):
        if f % SKIP != 0:
            continue
        if f not in frame_to_model_to_orientation_to_car_count :
            continue
        for m in frame_to_model_to_orientation_to_car_map[f]:
            if  m not in frame_to_model_to_orientation_to_car_count[f]:
                continue
            max_area = 0.0
            max_count = 0
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_car_count[f][m] or o not in frame_to_model_to_orientation_to_car_map[f][m]:
                    continue
                if frame_to_model_to_orientation_to_car_count[f][m][o] > max_count:
                    max_count = frame_to_model_to_orientation_to_car_count[f][m][o]
                total_area = 0.0
                for d in frame_to_model_to_orientation_to_cars_detected[f][m][o]:
                    total_area += (d.right - d.left) * (d.bottom - d.top)
                if total_area > max_area:
                    max_area = total_area
            if max_count == 0:
                continue
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_car_count[f][m] or o not in frame_to_model_to_orientation_to_car_map[f][m]:
                    continue
                total_area = 0.0
                for d in frame_to_model_to_orientation_to_cars_detected[f][m][o]:
                    total_area += (d.right - d.left) * (d.bottom - d.top)
                count = frame_to_model_to_orientation_to_car_count[f][m][o]
                frame_to_model_to_orientation_to_car_map[f][m][o] =frame_to_model_to_orientation_to_car_map[f][m][o] * 0.35 + (count / max_count) * 0.25 + (total_area / max_area) * 0.45
#                frame_to_model_to_orientation_to_car_map[f][m][o] =frame_to_model_to_orientation_to_car_map[f][m][o] * 0.3 + (count / max_count) * 0.4 + (total_area / max_area) * 0.3



    for f in range(frame_begin, frame_limit):
        if f % SKIP != 0:
            continue
        if  f not in frame_to_model_to_orientation_to_person_count or f not in frame_to_model_to_orientation_to_people_detected:
            continue
        for m in frame_to_model_to_orientation_to_person_map[f]:
            if  m not in frame_to_model_to_orientation_to_person_count[f] or f not in frame_to_model_to_orientation_to_people_detected:
                continue
            max_count = 0
            max_area = 0.0
            for o in orientations:
                # Get max count
                if o not in frame_to_model_to_orientation_to_person_count[f][m] or o not in frame_to_model_to_orientation_to_person_map[f][m] or o not in frame_to_model_to_orientation_to_people_detected[f][m]:
                    continue
                if frame_to_model_to_orientation_to_person_count[f][m][o] > max_count:
                    max_count = frame_to_model_to_orientation_to_person_count[f][m][o]

                total_area = 0.0
                for d in frame_to_model_to_orientation_to_people_detected[f][m][o]:
                    total_area += (d.right - d.left) * (d.bottom - d.top)
                if total_area > max_area:
                    max_area = total_area

            if max_count == 0:
                continue
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_person_count[f][m] or o not in frame_to_model_to_orientation_to_person_map[f][m]:
                    continue
                total_area = 0.0
                for d in frame_to_model_to_orientation_to_people_detected[f][m][o]:
                    total_area += (d.right - d.left) * (d.bottom - d.top)
                count = frame_to_model_to_orientation_to_person_count[f][m][o]
                frame_to_model_to_orientation_to_person_map[f][m][o] =frame_to_model_to_orientation_to_person_map[f][m][o] * 0.3 +  (count / max_count) * 0.4 + (total_area / max_area) * 0.3

def populate_map_dicts(map_file, 
                       frame_to_model_to_orientation_to_car_map, 
                       frame_to_model_to_orientation_to_person_map):
    orientations = generate_all_orientations()
    map_df = pd.read_csv(map_file) 
    for idx, row in map_df.iterrows():
        f = int(row['frame'])
        m = row['model']
        o = row['orientation']
        if f not in frame_to_model_to_orientation_to_car_map:
            frame_to_model_to_orientation_to_car_map[f] = {}
        if f not in frame_to_model_to_orientation_to_person_map:
            frame_to_model_to_orientation_to_person_map[f] = {}
        if m not in frame_to_model_to_orientation_to_car_map[f]:
            frame_to_model_to_orientation_to_car_map[f][m] = {}
        if m not in frame_to_model_to_orientation_to_person_map[f]:
            frame_to_model_to_orientation_to_person_map[f][m] = {}
        if row['object'] == 'car':
            frame_to_model_to_orientation_to_car_map[f][m][o] = row['map']
        elif row['object'] == 'person':
            frame_to_model_to_orientation_to_person_map[f][m][o] = row['map']

def generate_all_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    for r1 in range(0,360,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            for z in [1,2,3]:
                orientations.append(f'{r1}-{r2}-{z}')
    return orientations

def retrieve_counts_from_inference_file(inference_file, orientation, frame, model):
    if not os.path.exists(inference_file):
        print(inference_file, ' is missing')
        return [],[]
    if  os.stat(inference_file).st_size == 0:
        print(inference_file, ' is empty')
        return [],[]
    
    inference_df = pd.read_csv(inference_file)
    car_detections = []
    people_detections = []

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
            people_detections.append(obj)
    return car_detections, people_detections

def populate_count_dicts(inference_dir,
                         frame_begin,
                         frame_limit,
                         frame_to_model_to_orientation_to_car_count, 
                         frame_to_model_to_orientation_to_person_count,
                         frame_to_model_to_orientation_to_cars_detected,
                         frame_to_model_to_orientation_to_people_detected):
    orientations = generate_all_orientations()
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        print('Processing ', f)
        if f not in frame_to_model_to_orientation_to_car_count:
            frame_to_model_to_orientation_to_car_count[f] = {}
            frame_to_model_to_orientation_to_cars_detected[f] = {}
        if f not in frame_to_model_to_orientation_to_person_count:
            frame_to_model_to_orientation_to_person_count[f] = {}
            frame_to_model_to_orientation_to_people_detected[f] = {}
        for m in MODELS:
            if m not in frame_to_model_to_orientation_to_car_count[f]:
                frame_to_model_to_orientation_to_car_count[f][m] = {}
                frame_to_model_to_orientation_to_cars_detected[f][m] = {}
            if m not in frame_to_model_to_orientation_to_person_count[f]:
                frame_to_model_to_orientation_to_person_count[f][m] = {}
                frame_to_model_to_orientation_to_people_detected[f][m] = {}
            for o in orientations:
                inference_file = os.path.join(inference_dir, m, o, f'frame{f}.csv')
                car_detections, people_detections = retrieve_counts_from_inference_file(inference_file, o, f, m)
                frame_to_model_to_orientation_to_car_count[f][m][o] = len(car_detections)
                frame_to_model_to_orientation_to_person_count[f][m][o] = len(people_detections)
                frame_to_model_to_orientation_to_people_detected[f][m][o] = people_detections
                frame_to_model_to_orientation_to_cars_detected[f][m][o] = car_detections

            
            



# Given list of best fixed orientations for N PTZ cameras, find accuracy
def evaluate_best_fixed_with_multi_cameras(workload,
                    frame_begin,
                    frame_limit,
                    best_fixed_orientations,
                    orientations,
                    frame_to_model_to_orientation_to_car_count,
                    frame_to_model_to_orientation_to_person_count,
                    frame_to_model_to_orientation_to_car_map,
                    frame_to_model_to_orientation_to_person_map, 
                    frame_to_model_to_orientation_to_object_ids,
                    gt_model_to_object_ids):
    total_accuracy = 0.0
    num_frames = 0
    model_to_object_ids_found = {}

    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    num_non_aggregate_queries = len(workload) - num_aggregate_queries
    non_aggregate_accuracy = 0.0
    aggregate_accuracy = 0.0
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        max_est_current_total_accuracy = 0.0
        max_current_non_aggregate_accuracy = 0.0
        max_est_current_aggregate_accuracy = 0.0
        best_orientation = best_fixed_orientations[0]
        for o in best_fixed_orientations:
            # Go through best fixed orientations to determine max out of them for current frame
            current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)

            aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)

            est_current_total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * current_non_aggregate_accuracy

            if est_current_total_accuracy > max_est_current_total_accuracy:
                max_est_current_total_accuracy = est_current_total_accuracy
                max_current_non_aggregate_accuracy = current_non_aggregate_accuracy
                max_est_current_aggregate_accuracy = aggregate_accuracy
                best_orientation = o
        num_frames += 1
        # Make sure to get aggregate ids again once we determine the best orientation
        for o in best_fixed_orientations:
            evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, best_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        non_aggregate_accuracy += max_current_non_aggregate_accuracy
    non_aggregate_accuracy /= num_frames
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * non_aggregate_accuracy

    return total_accuracy


def find_total_aggregate_ids(frame_begin, frame_limit, 
                    frame_to_model_to_orientation_to_object_ids):

    model_to_all_object_ids = {}
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue

        if f not in frame_to_model_to_orientation_to_object_ids:
#            print('frame ', f, ' not present in MOT results')
            continue
        for m in MODELS:
            if m not in model_to_all_object_ids:
                model_to_all_object_ids[m] = []
            for o in frame_to_model_to_orientation_to_object_ids[f][m]:
                new_object_ids = frame_to_model_to_orientation_to_object_ids[f][m][o]
                model_to_all_object_ids[m].extend(
                    x for x in new_object_ids
                    if x not in model_to_all_object_ids[m]
                )
    return model_to_all_object_ids


#def evaluate_aggregate_queries(workload, model_to_object_ids_found, model_to_all_object_ids):
#    num_aggregate_queries = num_aggregate_queries_in_workload(workload)
#    total_aggregate_accuracy = 0.0
#    for q in workload:
#        total_aggregate_accuracy += len(model_to_object_ids_found[q[0]]) / len(model_to_all_object_ids[q[0]])
#    return total_aggregate_accuracy / num_aggregate_queries



def populate_frame_bound_to_regions(regions_file):
    frame_bound_to_regions = {}
    with open(regions_file, 'r') as f_regions:
        for line in f_regions.readlines():
            line = line.strip()
            elements = line.split(',') 
            frame_begin = int(elements[0])
            frame_limit = int(elements[1])
            orientations = []
            if (frame_begin, frame_limit) not in frame_bound_to_regions:
                frame_bound_to_regions[(frame_begin, frame_limit)] = []
            for i in range(2,len(elements)):
                o = elements[i]
                o = o.strip().replace('\'','')
                if '[' in o or ']' in o:
                    o = o.replace(']', '').replace('[', '')
                orientations.append(o)
            frame_bound_to_regions[(frame_begin, frame_limit)].append(orientations)
    return frame_bound_to_regions


def parse_orientation_string(orientation):
    final_vec = []
    split_orientation = orientation.split("-")
    add_negative = False
    for s in split_orientation:
        if len(s) == 0:
            add_negative = True;
        elif add_negative:
            final_vec.append(f'-{s}')
        else:
            final_vec.append(s)
    return final_vec


def neighboring_orientations(current_orientation, orientations):
   
    region_orientations = [] 
    for o in orientations:
        if find_pan_dist(extract_pan(o), extract_pan(current_orientation)) <= 30 and find_tilt_dist(extract_pan(o), extract_pan(current_orientation)) <= 15:
            region_orientations.append(o)
    return region_orientations

def determine_regions(frame_begin,
                      frame_limit,
                      orientations,
                      frame_to_model_to_orientation_to_car_count,
                      frame_to_model_to_orientation_to_person_count,
                      frame_to_model_to_orientation_to_car_map,
                      frame_to_model_to_orientation_to_person_map,
                      object_type # either 'car', 'person', or 'both'
                      ):
    print('determining regions with frames ', frame_begin, ' -- ', frame_limit)
    # 2d list, each sub-list contains list of orientations for a region
    regions = []

    def find_best_fixed_region_anchor(orientations, orientation_to_count):
        best_orientation = orientations[0]
        max_count = 0
        for o in orientations:
            if orientation_to_count[o] > max_count:
                max_count = orientation_to_count[o]
                best_orientation = o
        return max_count, best_orientation 

    def search_neighboring_orientations(current_orientation, orientations, orientation_to_car_count, orientation_to_person_count):
        car_count = 0
        person_count = 0
        num_entries = 0
        for o in neighboring_orientations(current_orientation, orientations):
            if o[-1] != current_orientation[-1]:
                # Ignore zooms for now
                continue
            car_count += orientation_to_car_count[o]
            person_count += orientation_to_person_count[o]
            num_entries += 1
        car_count /= num_entries
        person_count /= num_entries
        return car_count, person_count

    model = 'faster-rcnn'
    f = frame_begin
    orientation_to_car_count = {}
    orientation_to_person_count = {}
    orientation_to_all_count = {}
    num_frames = 0
    # Get counts at every minute for the video
    while f <= frame_limit:
        if f % SKIP != 0:
            f += 1
            continue
        for o in orientations:
            if o not in orientation_to_car_count:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                orientation_to_all_count[o] = 0
            orientation_to_car_count[o] += frame_to_model_to_orientation_to_car_count[f][model][o]
            orientation_to_person_count[o] += frame_to_model_to_orientation_to_person_count[f][model][o]
            orientation_to_all_count[o] += frame_to_model_to_orientation_to_person_count[f][model][o] + frame_to_model_to_orientation_to_car_count[f][model][o]
        f += 30
        num_frames += 1
    max_count = 0
    for o in orientations:
        orientation_to_all_count[o] /= num_frames
        orientation_to_car_count[o] /= num_frames
        orientation_to_person_count[o] /= num_frames

    remaining_orientations = orientations.copy()
    while len(remaining_orientations) > 60:
        if object_type == 'car':
            fixed_count,fixed_orientation = find_best_fixed_region_anchor(remaining_orientations, orientation_to_car_count) 
            car_count,_ = search_neighboring_orientations(fixed_orientation, remaining_orientations, orientation_to_car_count, orientation_to_person_count)
            # If count for region is low, leave
            if car_count < 0.45:
                break
        elif object_type == 'person':
            fixed_count,fixed_orientation = find_best_fixed_region_anchor(remaining_orientations, orientation_to_person_count) 

            car_count,person_count = search_neighboring_orientations(fixed_orientation, remaining_orientations, orientation_to_car_count, orientation_to_person_count)
            if person_count < 0.85:
                break
        elif object_type == 'both':
            fixed_count,fixed_orientation = find_best_fixed_region_anchor(remaining_orientations, orientation_to_all_count) 
            car_count,person_count = search_neighboring_orientations(fixed_orientation, remaining_orientations, orientation_to_car_count, orientation_to_person_count)
            if car_count < 0.45 or person_count < 0.85:
                break
        elif object_type == 'any':
            fixed_count,fixed_orientation = find_best_fixed_region_anchor(remaining_orientations, orientation_to_all_count) 
            car_count,person_count = search_neighboring_orientations(fixed_orientation, remaining_orientations, orientation_to_car_count, orientation_to_person_count)
            if car_count < 0.45 and person_count < 0.85:
                break
        else:
            print('object type ', object_type, ' is invalid')
            break

        region, remaining_orientations = carve_region_with_orientation(fixed_orientation, remaining_orientations)
        if len(region) < 60:
            break
        regions.append(region)
    return regions


# Determines all orientations in the region and the remaining orientations
def carve_region_with_orientation(current_orientation, orientations):
    region_orientations = []
    extra_orientations = []
    for o in orientations:
        if find_pan_dist(extract_pan(o), extract_pan(current_orientation)) <= 60:
            region_orientations.append(o)
        else:
            extra_orientations.append(o)
    if len(region_orientations) < 60:
        for o in orientations:
            if o in region_orientations or o in extra_orientations:
                continue
            if find_pan_dist(extract_pan(o), extract_pan(current_orientation)) <= 90:
                region_orientations.append(o)
            else:
                extra_orientations.append(o)
        if len(region_orientations) < 60:
            for o in orientations:
                if o in region_orientations or o in extra_orientations:
                    continue
                if find_pan_dist(extract_pan(o), extract_pan(current_orientation)) <= 120:
                    region_orientations.append(o)
                else:
                    extra_orientations.append(o)
    return region_orientations,extra_orientations
            

# String that summarizes the object type composition of a workload
def find_object_type(workload):
    car_queries = 0
    person_queries = 0
    for q in workload: 
        if q[2] == 'car':
            car_queries += 1
        elif q[2] == 'person':
            person_queries += 1
    car_queries /= len(workload)
    person_queries /= len(workload)
    if car_queries / (car_queries + person_queries) >= 0.7:
        return 'car'
    if person_queries / (car_queries + person_queries) >= 0.7:
        return 'person'
    return 'both'

# Returns average pan distance and average tilt distance for switches. ALso returns percentage of how many frames had a switch
def compute_switch_distances(best_orientation_list):
    prev_best_orientation = best_orientation_list[0]
    pan_distances = []
    tilt_distances = []
    switches = 0
    num_frames = 0
    for o in best_orientation_list:
        if o != prev_best_orientation:
            pan_distances.append(find_pan_dist(extract_pan(o), extract_pan(prev_best_orientation)))
            tilt_distances.append(find_tilt_dist(extract_tilt(o), extract_tilt(prev_best_orientation)))
            switches += 1
        num_frames += 1
        prev_best_orientation = o
        
    return sum(pan_distances) / len(pan_distances), sum(tilt_distances) / len(pan_distances), switches / num_frames

def find_best_dynamic_score_limit_changes(workload,
                       frame_begin,
                       frame_limit,
                       orientations,
                       best_fixed_orientation,
                       gap_between_changes,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                       gt_model_to_object_ids
                       ):
    total_accuracy = 0.0
    num_frames = 0
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    num_non_aggregate_queries = len(workload) - num_aggregate_queries

    non_aggregate_accuracy = 0.0
    aggregate_accuracy = 0.0
    best_orientation = best_fixed_orientation

    best_orientation = orientations[0]
    model_to_object_ids_found = {}
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, best_orientation, orientations, frame_to_model_to_orientation_to_car_count, frame_to_model_to_orientation_to_person_count,
                 frame_to_model_to_orientation_to_car_map, frame_to_model_to_orientation_to_person_map)
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, best_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        non_aggregate_accuracy += current_non_aggregate_accuracy
        if num_frames % gap_between_changes == 0:
            best_fixed_est_current_total_accuracy = 0.0
            max_est_current_total_accuracy = 0.0
            max_current_non_aggregate_accuracy = 0.0
            max_est_current_aggregate_accuracy = 0.0
            for o in orientations:
                # Go through best fixed orientations to determine max out of them for current frame
                current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                    frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
                current_model_to_object_ids_found = {}
                aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)
                est_current_total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * current_non_aggregate_accuracy

                if est_current_total_accuracy > max_est_current_total_accuracy:
                    max_est_current_total_accuracy = est_current_total_accuracy
                    max_current_non_aggregate_accuracy = current_non_aggregate_accuracy
                    max_est_current_aggregate_accuracy = aggregate_accuracy
                    best_orientation = o

              
        num_frames += 1
    non_aggregate_accuracy /= num_frames
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * non_aggregate_accuracy
    return total_accuracy 
# Given an orientation and a list of new orientations, return the new orientation that's closest to the current one
def select_closest_orientation(current_orientation, best_orientations_found):
    min_dist = 360
    closest_orientation = best_orientations_found[0]
    for o in best_orientations_found:
        d = find_pan_dist(extract_pan(current_orientation), extract_pan(o)) 
        if d < min_dist:
            min_dist = d
            closest_orientation = o
    return closest_orientation
            

def find_best_dynamic_score(workload,
                       frame_begin,
                       frame_limit,
                       orientations,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                       gt_model_to_object_ids,
                       num_frames_to_send=1,
                       blacklisted_frames=[],
                       best_fixed_orientation=None,
                       paper_results=None):
   
    # list of historical pairs (frame, orientation) 
    total_accuracy = 0.0
    duration_list = []
    current_duration = 0
    num_frames = 0
    num_frames_in_best_fixed = 0
    prev_best_orientation = orientations[0]

    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    num_non_aggregate_queries = len(workload) - num_aggregate_queries
    non_aggregate_accuracy = 0.0
    aggregate_accuracy = 0.0

    # Number of times an orientation was best
    orientation_occurrences = []

    model_to_object_ids_found = {}

    # Scale down length of aggregate groudn truth so per-frame est accuracies aren't too low
    for f in range(frame_begin, frame_limit):
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        best_fixed_est_current_total_accuracy = 0.0

        max_est_current_total_accuracy = 0.0
        max_current_non_aggregate_accuracy = 0.0
        max_est_current_aggregate_accuracy = 0.0
        best_orientation = orientations[0]

        best_orientations_found = []
        for o in orientations:
            # Go through best fixed orientations to determine max out of them for current frame
            current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)

            aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)
            est_current_total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * current_non_aggregate_accuracy

            if best_fixed_orientation is not None and o == best_fixed_orientation:
                best_fixed_est_current_total_accuracy = est_current_total_accuracy

            if est_current_total_accuracy > max_est_current_total_accuracy:
                max_est_current_total_accuracy = est_current_total_accuracy
                max_current_non_aggregate_accuracy = current_non_aggregate_accuracy
                max_est_current_aggregate_accuracy = aggregate_accuracy
                best_orientation = o
                best_orientations_found = [o]

            elif est_current_total_accuracy == max_est_current_total_accuracy:
                best_orientations_found.append(o)

        best_orientation = select_closest_orientation(prev_best_orientation, best_orientations_found)
        # Update for paper results on ranking and to see discrepancy between best dynamic and others
        orientation_occurrences.append(best_orientation)
 
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, best_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)


        # FOr sending multiple frames to server
        if num_frames_to_send > 1:
            for _ in range(1, num_frames_to_send):
                max_aggregate_accuracy = 0.0
                best_aggregate_orientation = orientations[0]
                for o in orientations:
                    if o == best_fixed_orientation:
                        continue
                    # Go through best fixed orientations to determine max out of them for current frame
                    current_aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)

                    if current_aggregate_accuracy > max_aggregate_accuracy :
                        best_orientations_found.append(o)
                        best_aggregate_orientation = o
                evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, best_aggregate_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)


        # ########

        non_aggregate_accuracy += max_current_non_aggregate_accuracy

        # compute percentage of frames spent in best fixed
        if best_fixed_orientation is not None:
            if best_fixed_est_current_total_accuracy  == max_est_current_total_accuracy:
                num_frames_in_best_fixed += 1
        # Keep track of durations of best orientation
        current_duration += 1
        if prev_best_orientation != best_orientation:
            duration_list.append(current_duration)
            current_duration = 0
        prev_best_orientation = best_orientation
        num_frames += 1
    non_aggregate_accuracy /= num_frames
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * non_aggregate_accuracy
    if paper_results is not None:
        if 'durations' not in paper_results:
            paper_results['durations'] = []
        paper_results['durations'].extend(duration_list)
        if best_fixed_orientation is not None:
            if 'best_fixed_percentage' not in paper_results:
                paper_results['best_fixed_percentage'] = []
            paper_results['best_fixed_percentage'].append(num_frames_in_best_fixed / num_frames)
        if 'best_orientation_occurrences' not in paper_results:
            paper_results['best_orientation_occurrences'] = []
        paper_results['best_orientation_occurrences'].append(orientation_occurrences)
        pan_distance, tilt_distance, switch_percentage = compute_switch_distances(orientation_occurrences)
        if  'pan_switch_distances' not in paper_results:
            paper_results['pan_switch_distances'] = []
        if 'tilt_switch_distances' not in paper_results:
            paper_results['tilt_switch_distances'] = []
        if 'switch_percentages'  not in paper_results:
            paper_results['switch_percentages'] = []
        paper_results['pan_switch_distances'].append(pan_distance)
        paper_results['tilt_switch_distances'].append(tilt_distance)
        paper_results['switch_percentages'].append(switch_percentage)
    return total_accuracy 



def replay_best_dynamic_trace(workload,
                       frame_begin,
                       frame_limit,
                       orientation_trace,
                       orientations,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                       gt_model_to_object_ids,
                       blacklisted_frames=[],
                       ):
   
    total_accuracy = 0.0
    num_frames = 0

    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    num_non_aggregate_queries = len(workload) - num_aggregate_queries
    non_aggregate_accuracy = 0.0
    aggregate_accuracy = 0.0

    model_to_object_ids_found = {}

    trace_idx = 0
    for f in range(frame_begin, frame_limit):
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        best_fixed_est_current_total_accuracy = 0.0

        max_est_current_total_accuracy = 0.0
        max_current_non_aggregate_accuracy = 0.0
        max_est_current_aggregate_accuracy = 0.0

        if trace_idx >= len(orientation_trace):
            print('Warning: orientation trace is smaller than total frames')
            break
        current_orientation = orientation_trace[trace_idx] 
        current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, current_orientation, orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, current_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        non_aggregate_accuracy += current_non_aggregate_accuracy
        num_frames += 1
        trace_idx += 1
    non_aggregate_accuracy /= num_frames
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * non_aggregate_accuracy
    return total_accuracy 


def get_baseline_results(workloads,
                       frame_bound_to_regions,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                       frame_to_model_to_orientation_to_object_id_to_mot_detected,
                       object_id_to_frame_to_model_to_orientations,
                      blacklisted_frames):

    workload_idx_to_fixed_scores = {}
    workload_idx_to_mab_accuracies = {}
    workload_idx_to_ptz_tracking_accuracies = {}
    workload_idx_to_panoptes_accuracies = {}
    for idx,w in enumerate(workloads):
        print('WORKLOAD ', w)
        for i,(frame_begin, frame_limit)  in enumerate(FRAME_BOUNDS):
            regions = frame_bound_to_regions[(frame_begin, frame_limit)]
            for r in regions:
                gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)

                fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames
                                    )
                mab_accuracy = baselines.mab_baseline(w,
                               frame_begin + int(0.3*(frame_limit - frame_begin)) ,
                               frame_limit,
                               r,
                               frame_to_model_to_orientation_to_car_count,
                               frame_to_model_to_orientation_to_person_count,
                               frame_to_model_to_orientation_to_car_map,
                               frame_to_model_to_orientation_to_person_map,
                               frame_to_model_to_orientation_to_object_ids,
                               gt_model_to_object_ids,
                                blacklisted_frames=blacklisted_frames
                               )
                ptz_tracking_accuracy = baselines.ptz_tracking(w,
                                   frame_begin + int(0.3*(frame_limit - frame_begin)) ,
                                    frame_limit,
                                    r,
                                    best_fixed_orientation,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    frame_to_model_to_orientation_to_object_id_to_mot_detected,
                                    object_id_to_frame_to_model_to_orientations,
                                    gt_model_to_object_ids,
                                blacklisted_frames=blacklisted_frames
                                    )
                panoptes_accuracy = baselines.run_panoptes(w,
                               frame_begin + int(0.3*(frame_limit - frame_begin)) ,
                               frame_limit,
                               r,
                               best_fixed_orientation,
                               frame_to_model_to_orientation_to_car_count,
                               frame_to_model_to_orientation_to_person_count,
                               frame_to_model_to_orientation_to_car_map,
                               frame_to_model_to_orientation_to_person_map,
                               frame_to_model_to_orientation_to_object_ids,
                               gt_model_to_object_ids,
                                blacklisted_frames=blacklisted_frames
                               )
                if idx not in workload_idx_to_fixed_scores:
                    workload_idx_to_fixed_scores[idx] = []
                    workload_idx_to_mab_accuracies[idx] = []
                    workload_idx_to_ptz_tracking_accuracies[idx] = []
                    workload_idx_to_panoptes_accuracies[idx] = []
                workload_idx_to_fixed_scores[idx].append(fixed_score)
                workload_idx_to_mab_accuracies[idx].append(mab_accuracy)
                workload_idx_to_ptz_tracking_accuracies[idx].append(ptz_tracking_accuracy)
                workload_idx_to_panoptes_accuracies[idx].append(panoptes_accuracy)
        print('Fixed scores ', workload_idx_to_fixed_scores[idx])
        print('MAB ', workload_idx_to_mab_accuracies[idx])
        print('PTZ tracking ', workload_idx_to_ptz_tracking_accuracies[idx])
        print('Panoptes ', workload_idx_to_panoptes_accuracies[idx])




def get_madeye_results(name, inference_dir,
                        rectlinear_dir,
                        params,
                        workloads,
                       frame_bound_to_regions,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_cars_detected,
                       frame_to_model_to_orientation_to_people_detected,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                      frame_to_model_to_orientation_to_efficientdet_cars_detected,
                      frame_to_model_to_orientation_to_efficientdet_person_count,
                        blacklisted_frames,
                        num_frames_to_send=1,
                        num_frames_to_explore=5,
                       ):

    all_fixed_results = []
    all_madeye_results = []
    all_gt_madeye_results=  []
    all_dynamic_results = []
    for i,w in enumerate(workloads):
        fixed_results = []
        madeye_results = []
        gt_madeye_results=  []
        dynamic_results = []
        print('WORKLOAD ', w)
        num_excluded_videos = 0
        excluded_bounds = []
        for idx, (frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
            regions = frame_bound_to_regions[(frame_begin, frame_limit)]
            print(frame_begin, ' -- ', frame_limit)
            for r in regions:

                gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)


                remaining_orientations = r.copy()
                best_fixed_orientations_for_workload = [] 
                for _ in range(0,num_frames_to_send):
                    _, best_fixed_orientation = evaluation_tools.find_best_fixed(w,
                                        frame_begin,
                                        frame_limit,
                                        remaining_orientations,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map,
                                        frame_to_model_to_orientation_to_object_ids,
                                        gt_model_to_object_ids
                                        )
    #                print('remaining orientations ', remaining_orientations)
    #                print('removing ', best_fixed_orientation)
                    remaining_orientations.remove(best_fixed_orientation)
                    best_fixed_orientations_for_workload.append(best_fixed_orientation)

                fixed_score = evaluate_best_fixed_with_multi_cameras(w,
                                    frame_begin + int(0.3*(frame_limit - frame_begin)) ,
                                    frame_limit,
                                    best_fixed_orientations_for_workload,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids
                                    )

                madeye_accuracy, gt_madeye_accuracy = madeye.run_madeye(name, inference_dir, rectlinear_dir, params, w,
                               frame_begin ,
                               frame_limit,
                               r,
                               best_fixed_orientation,
                               frame_to_model_to_orientation_to_car_count,
                               frame_to_model_to_orientation_to_person_count,
                               frame_to_model_to_orientation_to_cars_detected,
                               frame_to_model_to_orientation_to_people_detected,
                               frame_to_model_to_orientation_to_car_map,
                               frame_to_model_to_orientation_to_person_map,
                               frame_to_model_to_orientation_to_object_ids,
                               frame_to_model_to_orientation_to_efficientdet_cars_detected,
                               frame_to_model_to_orientation_to_efficientdet_person_count,
                               gt_model_to_object_ids,
                               num_frames_to_send,
                               num_frames_to_explore,
                               blacklisted_frames=blacklisted_frames
                               )
#                if gt_madeye_accuracy < fixed_score:
#                    print('GT is worse than fixed. ', frame_begin , ' -- ', frame_limit, ' SKIPPING')
#                    print('GT MadEye - ', gt_madeye_accuracy , ' fixed ', fixed_score)
#                    num_excluded_videos += 1
#                    excluded_bounds.append((frame_begin, frame_limit))
#                    continue
                best_dynamic_score = find_best_dynamic_score(w,
                                   frame_begin + int(0.3*(frame_limit - frame_begin)) ,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    num_frames_to_send=num_frames_to_send,
                                    blacklisted_frames=blacklisted_frames
                                    )

                fixed_results.append(fixed_score)
                madeye_results.append(madeye_accuracy)
                gt_madeye_results.append(gt_madeye_accuracy)
                dynamic_results.append(best_dynamic_score)
                print('fixed accuracy ', fixed_score)
                print('MadEye ', madeye_accuracy)
                print('MadEye GT accuracy ', gt_madeye_accuracy)
                print('DYnamic score ', best_dynamic_score)

        all_fixed_results.extend(fixed_results)
        all_madeye_results.extend(madeye_results)
        all_gt_madeye_results.extend(gt_madeye_results)
        all_dynamic_results.extend(dynamic_results)
        fixed_results.sort()
        madeye_results.sort()
        gt_madeye_results.sort()
        dynamic_results.sort()
        print('NUMBER OF EXCLUDED VIDEOS ', num_excluded_videos)
        print('Excluded frame bounds ', excluded_bounds)
        print('Best fixed scores ', fixed_results)
        print('MadEye scores ', madeye_results)
        print('MadEye GT scores ', gt_madeye_results)
        print('Best dynamic scores ', dynamic_results)

        print('Best fixed median ', fixed_results[int(len(fixed_results) / 2) ])
        print('MadEye median ', madeye_results[int(len(madeye_results) / 2) ])
        print('MadEye GT median ', gt_madeye_results[int(len(gt_madeye_results) / 2) ])
        print('Best dynamic median ', dynamic_results[int(len(dynamic_results) / 2) ])

    print('fixed_results_traf = ', all_fixed_results)
    print('madeye_results_traf = ', all_madeye_results)
    print('gt_results_traf = ', all_gt_madeye_results)
    print('dynamic_results_traf = ', all_dynamic_results)
    all_fixed_results.sort()
    all_madeye_results.sort()
    all_gt_madeye_results.sort()
    all_dynamic_results.sort()
    print('****')
    print('Overall fixed median ', all_fixed_results[int(len(all_fixed_results) / 2) ])
    print('Overall MadEye median ', all_madeye_results[int(len(all_madeye_results) / 2) ])
    print('Overall MadEye GT median ', all_gt_madeye_results[int(len(all_gt_madeye_results) / 2) ])
    print('Overall Best dynamic median ', all_dynamic_results[int(len(all_dynamic_results) / 2) ])
    with open(f'fig14_results_{num_frames_to_send}_{num_frames_to_explore}.txt', 'w') as f:
        print('fixed_results_traf = '+  str(all_fixed_results) + '\n')
        print('madeye_results_traf = ' + str(all_madeye_results) + '\n')
        print('gt_results_traf = ' + str( all_gt_madeye_results) + '\n')
        print('dynamic_results_traf = ' + str(all_dynamic_results) + '\n')

def get_paper_results( frame_bound_to_regions,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                        blacklisted_frames
                       ):
    from collections import Counter

    # 1
    w1 = [('ssd-voc', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'binary-classification', 'car'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('ssd-voc', 'count', 'person'), 
          ('yolov4', 'detection', 'person'), # Change to detection
          ('faster-rcnn', 'detection', 'person')]  # change to detecton
    

    # 11
    w7 = [('ssd-voc', 'binary-classification', 'car'), 
          ('faster-rcnn', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'detection', 'person'),  # Change to detection
          ('tiny-yolov4', 'binary-classification', 'person'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('yolov4', 'detection', 'person'), # Change to detection
          ('faster-rcnn', 'aggregate-count', 'person'), 
          ('ssd-voc', 'binary-classification', 'person'), 
          ('faster-rcnn', 'count', 'car'), 
          ('ssd-voc', 'count', 'car')]
    # 12
    w8 = [('tiny-yolov4', 'count', 'car'), 
          ('faster-rcnn', 'detection', 'car'), # Change to detection
          ('faster-rcnn', 'aggregate-count', 'person')]

    #15
    w12 = [('faster-rcnn', 'count', 'car'), 
           ('tiny-yolov4', 'binary-classification', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('yolov4', 'count', 'car'), 
           ('tiny-yolov4', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'aggregate-count', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'count', 'car'), 
           ('ssd-voc', 'count', 'car'), 
           ('faster-rcnn', 'count', 'car'), 
           ('ssd-voc', 'binary-classification', 'car'), 
           ('yolov4', 'binary-classification', 'car'), 
           ('ssd-voc', 'binary-classification', 'car'), 
           ('ssd-voc', 'count', 'person'), 
           ('yolov4', 'count', 'person'), 
           ('yolov4', 'binary-classification', 'car'), 
           ('faster-rcnn', 'aggregate-count', 'person'), 
           ('ssd-voc', 'detection', 'car')]# Change to detection

    w14 = [('faster-rcnn', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'count', 'car'), 
           ('faster-rcnn', 'count', 'person')]
       
    main_motivation_workloads = [w1, w7, w8, w12, w14]

    for i,w in enumerate(main_motivation_workloads):
        paper_results = {}
        paper_results['best_orientation_occurrences'] = []
        paper_results['pan_switch_distances'] = []
        paper_results['tilt_switch_distances'] = []
        paper_results['switch_percentages'] = []

        first_frame_fixed_accuracies = []
        fixed_accuracies = []
        best_dynamic_accuracies = []
        workload_idx_to_accuracy_with_current_trace = {}
        print('**WORKLOAD ', i, '**')
        print(w)

        for idx, (frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
            regions = frame_bound_to_regions[(frame_begin, frame_limit)]
            # List contains all possible values for num_cameras
            for r in regions:
                gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)
                _, best_first_frame_fixed_orientation = evaluation_tools.find_best_fixed(w,
                                    frame_begin,
                                    frame_begin + SKIP,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames,
                                    )

                first_frame_fixed_score  = evaluation_tools.evaluate_workload_with_orientation(w,
                                    frame_begin,
                                    frame_limit,
                                    best_first_frame_fixed_orientation,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map, 
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames,
                                    )
                # Then evaluate all of those orientations
                first_fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames,
                                    )
                first_best_dynamic_score = find_best_dynamic_score(w,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames,
                                    paper_results=paper_results)

                for workload_idx,w_new in enumerate(main_motivation_workloads):

                    assert(len(paper_results['best_orientation_occurrences']) > 0  )
                    trace = paper_results['best_orientation_occurrences'][-1]
                    # Fig 4 different workload experiments
                    new_gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w_new, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)
                    if workload_idx not in workload_idx_to_accuracy_with_current_trace:
                        workload_idx_to_accuracy_with_current_trace[workload_idx] = []

                    best_dynamic_score  = find_best_dynamic_score(w_new,
                                        frame_begin,
                                        frame_limit,
                                        r,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map,
                                        frame_to_model_to_orientation_to_object_ids,
                                        new_gt_model_to_object_ids,
                                        blacklisted_frames=blacklisted_frames
                                    )

                    fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w_new,
                                        frame_begin,
                                        frame_limit,
                                        r,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map,
                                        frame_to_model_to_orientation_to_object_ids,
                                        new_gt_model_to_object_ids,
                                        blacklisted_frames=blacklisted_frames,
                                        )

                    replayed_score = replay_best_dynamic_trace(w_new,
                                           frame_begin,
                                           frame_limit,
                                           trace,
                                           r,
                                           frame_to_model_to_orientation_to_car_count,
                                           frame_to_model_to_orientation_to_person_count,
                                           frame_to_model_to_orientation_to_car_map,
                                           frame_to_model_to_orientation_to_person_map,
                                           frame_to_model_to_orientation_to_object_ids,
                                           new_gt_model_to_object_ids,
                                           blacklisted_frames=blacklisted_frames,
                                           )
                    workload_idx_to_accuracy_with_current_trace[workload_idx].append((best_dynamic_score - fixed_score) - max(0, replayed_score - fixed_score ))
                first_frame_fixed_accuracies.append(first_frame_fixed_score)
                fixed_accuracies.append(first_fixed_score)
                best_dynamic_accuracies.append(first_best_dynamic_score)
        print('first frame fixed (fig 1)\n', first_frame_fixed_accuracies)
        print('best fixed (fig 1)\n', fixed_accuracies)
        print('best dynamic (fig 1)\n', best_dynamic_accuracies)
        print('durations (fig 3)\n', paper_results['durations'])
        print('Applying dynamic trace on other workloads (fig 4)')
        for workload_idx,w_new in enumerate(workload_idx_to_accuracy_with_current_trace):
            print(w_new)
            print(main_motivation_workloads[workload_idx])
            print(workload_idx_to_accuracy_with_current_trace[workload_idx])

   
        occurrences_results = [] 
        for occurrences_list in paper_results['best_orientation_occurrences']:
            occurrences_dict = Counter(occurrences_list)
            for o in occurrences_dict:
                occurrences_results.append(occurrences_dict[o])
        print('occurrences (fig 7)\n', occurrences_results)
    figure2_tiny_yolov4_people_workloads = [
          [('tiny-yolov4', 'binary-classification', 'person')], 
          [('tiny-yolov4', 'count', 'person')], 
          [('tiny-yolov4', 'detection', 'person')], 
          [('tiny-yolov4', 'aggregate-count', 'person')], 
    ]
    figure2_frcnn_people_workloads = [
          [('faster-rcnn', 'binary-classification', 'person')], 
          [('faster-rcnn', 'count', 'person')], 
          [('faster-rcnn', 'detection', 'person')], 
          [('faster-rcnn', 'aggregate-count', 'person')], 
    ]

    figure2_yolov4_cars_workloads = [
          [('yolov4', 'binary-classification', 'car')], 
          [('yolov4', 'count', 'car')], 
          [('yolov4', 'detection', 'car')], 
    ]
    figure2_ssd_voc_cars_workloads = [
          [('ssd-voc', 'binary-classification', 'car')], 
          [('ssd-voc', 'count', 'car')], 
          [('ssd-voc', 'detection', 'car')], 
    ]
    for workload_set in [figure2_tiny_yolov4_people_workloads, figure2_frcnn_people_workloads, figure2_yolov4_cars_workloads, figure2_ssd_voc_cars_workloads]:
        print('WORKLOAD SET ', workload_set)
        for i,w in enumerate(workload_set):
            fixed_accuracies = []
            for idx, (frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
                regions = frame_bound_to_regions[(frame_begin, frame_limit)]
                # List contains all possible values for num_cameras
                for r in regions:
                    gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)

                    # Then evaluate all of those orientations
                    fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w,
                                        frame_begin,
                                        frame_limit,
                                        r,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map,
                                        frame_to_model_to_orientation_to_object_ids,
                                        gt_model_to_object_ids,
                                        blacklisted_frames=blacklisted_frames
                                        )

                    best_dynamic_score = find_best_dynamic_score(w,
                                        frame_begin,
                                        frame_limit,
                                        r,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map,
                                        frame_to_model_to_orientation_to_object_ids,
                                        gt_model_to_object_ids,
                                        blacklisted_frames=blacklisted_frames
                                        )

                    fixed_accuracies.append(best_dynamic_score - fixed_score)
            print(w, ' results (fig 2)')
            print(fixed_accuracies)
        print()

   # Figure 5 stuff 

    figure5_base_workload = [('yolov4', 'count', 'person')]
    figure5_model_varying_workload_set = [
        [('faster-rcnn', 'count', 'person')],
        [('ssd-voc', 'count', 'person')],
    ]
    figure5_task_varying_set = [
        [('yolov4', 'detection', 'person')],
        [('yolov4', 'aggregate-count', 'person')]
    ]

    figure5_object_varying_set = [
        [('yolov4', 'count', 'car')],
        [('yolov4', 'count', 'person'),  ('yolov4', 'count', 'car')]

    ]
    full_fig5_set = []
    full_fig5_set.extend(figure5_model_varying_workload_set)
    full_fig5_set.extend(figure5_task_varying_set)
    full_fig5_set.extend(figure5_object_varying_set)
    workload_idx_to_accuracy_with_base_trace = {}
    for idx, (frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
        regions = frame_bound_to_regions[(frame_begin, frame_limit)]
        # List contains all possible values for num_cameras
        for r in regions:
            gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(figure5_base_workload, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)
            paper_results = {}
            best_dynamic_score = find_best_dynamic_score(figure5_base_workload,
                                frame_begin,
                                frame_limit,
                                r,
                                frame_to_model_to_orientation_to_car_count,
                                frame_to_model_to_orientation_to_person_count,
                                frame_to_model_to_orientation_to_car_map,
                                frame_to_model_to_orientation_to_person_map,
                                frame_to_model_to_orientation_to_object_ids,
                                gt_model_to_object_ids,
                                paper_results=paper_results)
            assert len(paper_results['best_orientation_occurrences']) == 1
            base_trace = paper_results['best_orientation_occurrences'][-1]
            for workload_idx,w_new in enumerate(full_fig5_set):
                # Fig 4 different workload experiments
                new_gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w_new, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)
                if workload_idx not in workload_idx_to_accuracy_with_base_trace:
                    workload_idx_to_accuracy_with_base_trace[workload_idx] = []
                best_dynamic_score  = find_best_dynamic_score(w_new,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    new_gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames,
                                    )

                fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w_new,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    new_gt_model_to_object_ids,
                                    blacklisted_frames=blacklisted_frames
                                    )
                replayed_score = replay_best_dynamic_trace(w_new,
                                       frame_begin,
                                       frame_limit,
                                       base_trace,
                                       r,
                                       frame_to_model_to_orientation_to_car_count,
                                       frame_to_model_to_orientation_to_person_count,
                                       frame_to_model_to_orientation_to_car_map,
                                       frame_to_model_to_orientation_to_person_map,
                                       frame_to_model_to_orientation_to_object_ids,
                                       new_gt_model_to_object_ids,
                                       blacklisted_frames=blacklisted_frames,
                                       )
                workload_idx_to_accuracy_with_base_trace[workload_idx].append((best_dynamic_score - fixed_score) - max(0, replayed_score - fixed_score ))

    for workload_idx,w_new in enumerate(workload_idx_to_accuracy_with_base_trace):
        print(w_new, ' using base trace results (fig 5)')
        print(full_fig5_set[workload_idx])
        print(workload_idx_to_accuracy_with_base_trace[workload_idx])

def get_initial_results(workloads,
                       frame_bound_to_regions,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                        blacklisted_frames
                       ):
    # Best fixed orientations at idx i corresponds to the entry i in workloads
    # Each entry in the sub-list is an orientation at a different region
    best_fixed_orientations = []

    num_camera_settings = [1,2,3,4]
    num_camera_settings = [1]
    for i,w in enumerate(workloads):

        paper_results = {}

        paper_results['multi_camera_fixed_accuracies'] = [[] for _ in range(0,5)]
        paper_results['best_orientation_occurrences'] = []
        paper_results['pan_switch_distances'] = []
        paper_results['tilt_switch_distances'] = []

        paper_results['switch_percentages'] = []
        num_cameras_to_fixed_accuracies = {}
        fixed_accuracies = []
        best_dynamic_accuracies = []
        best_fixed_dynamic_discrepancies = []
        best_dynamic_accuracies_with_gap_between_changes = []
        best_fixed_orientations.append([])
        print('**WORKLOAD ', i, '**')
        print(w)
        for idx, (frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
            regions = frame_bound_to_regions[(frame_begin, frame_limit)]
            # List contains all possible values for num_cameras
            for r in regions:
                gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)

                tmp_fixed_accuracies = []
                for num_cameras in num_camera_settings:
                    best_fixed_orientations_for_workload = []
                    remaining_orientations = r.copy()
                    # First get all best fixed orientations
                    for _ in range(0,num_cameras):
                        single_camera_fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w,
                                            frame_begin,
                                            frame_limit,
                                            remaining_orientations,
                                            frame_to_model_to_orientation_to_car_count,
                                            frame_to_model_to_orientation_to_person_count,
                                            frame_to_model_to_orientation_to_car_map,
                                            frame_to_model_to_orientation_to_person_map,
                                            frame_to_model_to_orientation_to_object_ids,
                                            gt_model_to_object_ids
                                            )
        #                print('remaining orientations ', remaining_orientations)
        #                print('removing ', best_fixed_orientation)
                        remaining_orientations.remove(best_fixed_orientation)
                        best_fixed_orientations_for_workload.append(best_fixed_orientation)
        
                    # Then evaluate all of those orientations
                    fixed_score = evaluate_best_fixed_with_multi_cameras(w,
                                        frame_begin,
                                        frame_limit,
                                        best_fixed_orientations_for_workload,
                                        r,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map,
                                        frame_to_model_to_orientation_to_object_ids,
                                        gt_model_to_object_ids
                                        )
                    tmp_fixed_accuracies.append(single_camera_fixed_score) 
                    if num_cameras not in num_cameras_to_fixed_accuracies:
                        num_cameras_to_fixed_accuracies[num_cameras] = []
                    num_cameras_to_fixed_accuracies[num_cameras].append(fixed_score)
                
                # Best dynamic score doesn't actually make sense since it's the denominator ...
                best_dynamic_score = find_best_dynamic_score(w,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids,
                                    best_fixed_orientation=best_fixed_orientations_for_workload[0],
                                    paper_results=paper_results)
                best_fixed_orientations[i].append(best_fixed_orientations_for_workload[0])
                fixed_accuracies.append(tmp_fixed_accuracies[0])
                best_dynamic_accuracies.append(best_dynamic_score)
                best_fixed_dynamic_discrepancies.append(best_dynamic_score - tmp_fixed_accuracies[0] )
                for j in range(0, len(tmp_fixed_accuracies)):
                    paper_results['multi_camera_fixed_accuracies'][j].append(tmp_fixed_accuracies[j])

                gap_between_best_dynamic_changes = [1, 2, 5, 10, 1000] # seconds
                best_dynamic_accuracy_with_gap_between_changes = {} # gap to accuracy dict
                # THis is used for getting the results for when we switch at low frequency
                for gap in gap_between_best_dynamic_changes:
                    best_dynamic_accuracy_with_gap_between_changes[gap] = find_best_dynamic_score_limit_changes(w,
                                    frame_begin,
                                    frame_limit,
                                    r,
                                    best_fixed_orientations_for_workload[0],
                                    gap,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map,
                                    gt_model_to_object_ids)
                best_dynamic_accuracies_with_gap_between_changes.append(best_dynamic_accuracy_with_gap_between_changes)


        print('multi-camera results')
        for num_cameras in num_cameras_to_fixed_accuracies:
            print('\tnum cameras: ', num_cameras, ' ',num_cameras_to_fixed_accuracies[num_cameras])
        print('Top fixed accuracies')
        for idx,top_fixed_accuracies in enumerate(paper_results['multi_camera_fixed_accuracies']):
            print('Top ', idx, ' accuracies: ', top_fixed_accuracies)
        print()
        print('Best orientation occurrences (used for histogram)')
        for idx,val in enumerate(paper_results['best_orientation_occurrences']):
            print(val)
        print()
        print('Pan switch distances', paper_results['pan_switch_distances'])
         
        print()
        print('Tilt switch distances', paper_results['tilt_switch_distances'])

        print()
        print('Percentage of frames that had a switch', paper_results['switch_percentages'])
        print()
        print('best fixed scores ', fixed_accuracies)
        print('')
        print('best dyn scores ', best_dynamic_accuracies)
        print()
        print('Discrepancies between fixed/dyn: ', best_fixed_dynamic_discrepancies)
        print()
        print('best dyn scores with gap between changes', json.dumps(best_dynamic_accuracies_with_gap_between_changes, indent=2))
        print('percentage of time in best fixed ', paper_results['best_fixed_percentage'])
        print('Best orientation durations ', paper_results['durations'])

    workload_accuracies_on_all_fixed_orientations = {}

    # Evaluate each best fixed orientation on other worklaods
    for i,w in enumerate(workloads):
        region_idx = 0
        for idx, (frame_begin, frame_limit) in enumerate(frame_bound_to_regions):
            regions = frame_bound_to_regions[(frame_begin, frame_limit)]
            for r in regions: 
                gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)
                best_accuracy = evaluation_tools.evaluate_workload_with_orientation(w,
                                    frame_begin,
                                    frame_limit,
                                    best_fixed_orientations[i][region_idx],
                                    r,
                                    frame_to_model_to_orientation_to_car_count,
                                    frame_to_model_to_orientation_to_person_count,
                                    frame_to_model_to_orientation_to_car_map,
                                    frame_to_model_to_orientation_to_person_map, 
                                    frame_to_model_to_orientation_to_object_ids,
                                    gt_model_to_object_ids)
                if i not in workload_accuracies_on_all_fixed_orientations:
                    workload_accuracies_on_all_fixed_orientations[i] = {}

                for j,fixed_orientations in enumerate(best_fixed_orientations):
                    fixed_orientation = fixed_orientations[region_idx]
                    if j not in workload_accuracies_on_all_fixed_orientations[i]:
                        workload_accuracies_on_all_fixed_orientations[i][j] = []
                    if j == i:
                        workload_accuracies_on_all_fixed_orientations[i][j].append(best_accuracy)
                        continue
                    accuracy = evaluation_tools.evaluate_workload_with_orientation(w,
                                        frame_begin,
                                        frame_limit,
                                        fixed_orientation,
                                        r,
                                        frame_to_model_to_orientation_to_car_count,
                                        frame_to_model_to_orientation_to_person_count,
                                        frame_to_model_to_orientation_to_car_map,
                                        frame_to_model_to_orientation_to_person_map, 
                                        frame_to_model_to_orientation_to_object_ids,
                                        gt_model_to_object_ids)
                    workload_accuracies_on_all_fixed_orientations[i][j].append(accuracy)
                region_idx += 1

    print('**Other workload eval**')
    for i in workload_accuracies_on_all_fixed_orientations:
        print('WORKLOAD ', i, ': ', workloads[i])
        for j in workload_accuracies_on_all_fixed_orientations[i]:
            print('\tAccuracies w/ workload ', j, ' orientation: ', workload_accuracies_on_all_fixed_orientations[i][j])



def get_misc_results( inference_dir,
                       workloads,
                       orientations,
                       frame_bound_to_regions,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_cars_detected,
                       frame_to_model_to_orientation_to_people_detected,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                       ):

    car_deltas = []
    person_deltas = []
    workload_idx_to_pan_dists = {}
    workload_idx_to_tilt_dists = {}
    workload_idx_to_intersection_lengths  = {}

    for idx,(frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
        object_type = 'any'
        populate_count_dicts(inference_dir, frame_begin, frame_limit, frame_to_model_to_orientation_to_car_count, frame_to_model_to_orientation_to_person_count, frame_to_model_to_orientation_to_cars_detected, frame_to_model_to_orientation_to_people_detected)
        regions = determine_regions(frame_begin, 
                              frame_begin + int(0.3*(frame_limit - frame_begin)),
                              orientations,
                              frame_to_model_to_orientation_to_car_count,
                              frame_to_model_to_orientation_to_person_count,
                              frame_to_model_to_orientation_to_car_map,
                              frame_to_model_to_orientation_to_person_map,
                              object_type)
        print('frames ', frame_begin , ' -- ', frame_limit)
        for r in regions:
            for w_idx,w in enumerate(workloads):
                if w_idx not in workload_idx_to_pan_dists:
                    workload_idx_to_pan_dists[w_idx] = []
                    workload_idx_to_tilt_dists[w_idx] = []
                    workload_idx_to_intersection_lengths[w_idx] = []
                pan_dists, tilt_dists, intersection_lengths = misc_results.distance_between_top_k_orientations_each_frame(w,
                                                              frame_begin, 
                                                               frame_limit,
                                                               r,
                                                               frame_to_model_to_orientation_to_car_count,
                                                               frame_to_model_to_orientation_to_person_count,
                                                               frame_to_model_to_orientation_to_car_map,
                                                               frame_to_model_to_orientation_to_person_map,
                                                               frame_to_model_to_orientation_to_object_ids)
                workload_idx_to_pan_dists[w_idx].extend(pan_dists)
                workload_idx_to_tilt_dists[w_idx].extend(tilt_dists)
                workload_idx_to_intersection_lengths[w_idx].extend(intersection_lengths)


#        current_car_deltas = misc_results.delta_of_delta_graph(args.inference,
#                                 frame_begin,
#                                 frame_limit,
#                                 frame_to_model_to_orientation_to_car_count)
#        current_person_deltas = misc_results.delta_of_delta_graph(args.inference,
#                                 frame_begin,
#                                 frame_limit,
#                                 frame_to_model_to_orientation_to_person_count)
#        car_deltas.append(current_car_deltas)
#        person_deltas.append(current_person_deltas)
##        car_deltas.extend(current_car_deltas)
##        person_deltas.extend(current_person_deltas)
#    print('Car deltas')
#    for c in car_deltas:
#        print(c)
#    print('ppl deltas')
#    for p in person_deltas:
#        print(p)

    # misc results: distances between top orientations at each frame
    for w_idx in range(0, len(workloads)):
        print('Workload ', w_idx , ' -> ',workloads[w_idx])
        print('PAN DISTANCES ', workload_idx_to_pan_dists[w_idx])
        print('TILT DISTANCES ', workload_idx_to_tilt_dists[w_idx])
        print('INTERSECTION SIZES ', workload_idx_to_intersection_lengths[w_idx])

def init_query_specific_workloads():
    w1 = [('ssd-voc', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'binary-classification', 'car'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('ssd-voc', 'count', 'person'), 
          ('yolov4', 'detection', 'person'), # Change to detection
          ('faster-rcnn', 'detection', 'person')]  # change to detecton

def init_fig14_workloads():

    w1 = [('ssd-voc', 'binary-classification', 'person')]
    w2 = [('yolov4', 'binary-classification', 'person')]
    w3 = [('tiny-yolov4', 'binary-classification', 'person')]
    w4 = [('faster-rcnn', 'binary-classification', 'person')]

    w5 = [('ssd-voc', 'count', 'person')]
    w6 = [('yolov4', 'count', 'person')]
    w7 = [('tiny-yolov4', 'count', 'person')]
    w8 = [('faster-rcnn', 'count', 'person')]

    w9 = [('ssd-voc', 'detection', 'person')]
    w10 = [('yolov4', 'detection', 'person')]
    w11 = [('tiny-yolov4', 'detection', 'person')]
    w12 = [('faster-rcnn', 'detection', 'person')]

    w13 = [('ssd-voc', 'aggregate-count', 'person')]
    w14 = [('yolov4', 'aggregate-count', 'person')]
    w15 = [('tiny-yolov4', 'aggregate-count', 'person')]
    w16 = [('faster-rcnn', 'aggregate-count', 'person')]

    w17 = [('ssd-voc', 'binary-classification', 'car')]
    w18 = [('yolov4', 'binary-classification', 'car')]
    w19 = [('tiny-yolov4', 'binary-classification', 'car')]
    w20 = [('faster-rcnn', 'binary-classification', 'car')]

    w21 = [('ssd-voc', 'count', 'car')]
    w22 = [('yolov4', 'count', 'car')]
    w23 = [('tiny-yolov4', 'count', 'car')]
    w24 = [('faster-rcnn', 'count', 'car')]

    w25 = [('ssd-voc', 'detection', 'car')]
    w26 = [('yolov4', 'detection', 'car')]
    w27 = [('tiny-yolov4', 'detection', 'car')]
    w28 = [('faster-rcnn', 'detection', 'car')]
    return [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17,w18, w19, w20, w21,w22,w23,w24,w25,w26,w27,w28]






def init_workloads():
    # 1
    w1 = [('ssd-voc', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'binary-classification', 'car'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('ssd-voc', 'count', 'person'), 
          ('yolov4', 'detection', 'person'), # Change to detection
          ('faster-rcnn', 'detection', 'person')]  # change to detecton
    # 2
    w2 = [('yolov4', 'binary-classification', 'person'), 
          ('ssd-voc', 'detection', 'car'),  # change to detection
          ('ssd-voc', 'binary-classification', 'car'), 
          ('yolov4', 'detection', 'car'),  # change to detection
          ('tiny-yolov4', 'count', 'person'), 
          ('yolov4', 'aggregate-count', 'person'), 
          ('yolov4', 'binary-classification', 'person'), 
          ('ssd-voc', 'count', 'person')]
    #7
    w3 = [('yolov4', 'binary-classification', 'person'), 
          ('ssd-voc', 'detection', 'person'),  # Change to detection
          ('tiny-yolov4', 'binary-classification', 'car'), 
          ('tiny-yolov4', 'detection', 'person'),  # Change to detection
          ('ssd-voc', 'binary-classification', 'person'), 
          ('ssd-voc', 'aggregate-count', 'person'), 
          ('tiny-yolov4', 'detection', 'person'),  # Change to detection
          ('ssd-voc', 'count', 'car'), 
          ('ssd-voc', 'count', 'person'), 
          ('faster-rcnn', 'count', 'person'), 
          ('yolov4', 'count', 'person'), 
          ('faster-rcnn', 'binary-classification', 'person'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'count', 'car'), 
          ('yolov4', 'binary-classification', 'car')]
    #8
    w4 = [('tiny-yolov4', 'count', 'person'), 
          ('ssd-voc', 'count', 'person'), 
          ('faster-rcnn', 'count', 'car')]


    #9
    w5 = [('yolov4', 'aggregate-count', 'person'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('tiny-yolov4', 'detection', 'person'),  # Change to etection
          ('yolov4', 'binary-classification', 'person'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'count', 'person'), 
          ('faster-rcnn', 'detection', 'person'),  # Change to detection
          ('faster-rcnn', 'count', 'car'), 
          ('yolov4', 'aggregate-count', 'person'), 
          ('yolov4', 'detection', 'person'),  # Change to detection
          ('yolov4', 'detection', 'person'), # Change to detection
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('yolov4', 'count', 'car'), 
          ('yolov4', 'detection', 'car'),  # Change to detection
          ('tiny-yolov4', 'count', 'car'), 
          ('ssd-voc', 'binary-classification', 'person'), 
          ('faster-rcnn', 'count', 'car'), 
          ('ssd-voc', 'count', 'person')]
    
    # 10
    w6 = [('ssd-voc', 'aggregate-count', 'person'), 
          ('ssd-voc', 'binary-classification', 'person'), 
          ('faster-rcnn', 'binary-classification', 'person')]

    # 11
    w7 = [('ssd-voc', 'binary-classification', 'car'), 
          ('faster-rcnn', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'detection', 'person'),  # Change to detection
          ('tiny-yolov4', 'binary-classification', 'person'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('yolov4', 'detection', 'person'), # Change to detection
          ('faster-rcnn', 'aggregate-count', 'person'), 
          ('ssd-voc', 'binary-classification', 'person'), 
          ('faster-rcnn', 'count', 'car'), 
          ('ssd-voc', 'count', 'car')]
    # 12
    w8 = [('tiny-yolov4', 'count', 'car'), 
          ('faster-rcnn', 'detection', 'car'), # Change to detection
          ('faster-rcnn', 'aggregate-count', 'person')]
    # 14
    w9 = [('tiny-yolov4', 'count', 'car'), 
          ('ssd-voc', 'count', 'car'), 
          ('faster-rcnn', 'aggregate-count', 'person')]
    # 16
    w10 = [('tiny-yolov4', 'aggregate-count', 'person'), 
          ('tiny-yolov4', 'binary-classification', 'person'), 
          ('ssd-voc', 'count', 'car'), 
          ('yolov4', 'aggregate-count', 'person'), 
          ('tiny-yolov4', 'count', 'person'), 
          ('faster-rcnn', 'binary-classification', 'car'), 
          ('ssd-voc', 'detection', 'person'), # Change to detection
          ('faster-rcnn', 'detection', 'car'),  # Change to detection
          ('faster-rcnn', 'aggregate-count', 'person'), 
          ('yolov4', 'count', 'car'), 
          ('tiny-yolov4', 'aggregate-count', 'person'), 
          ('faster-rcnn', 'detection', 'person'),  # Change to detection
          ('ssd-voc', 'aggregate-count', 'person'), 
          ('yolov4', 'detection', 'car')]# Change to detection

    # 18
    w11 = [('tiny-yolov4', 'detectipn', 'car'), # Change to detection
           ('ssd-voc', 'aggregate-count', 'person'), 
           ('ssd-voc', 'aggregate-count', 'person'), 
           ('tiny-yolov4', 'detection', 'car'), # Change to detection
           ('yolov4', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'detection', 'car'), # Change to detection
           ('tiny-yolov4', 'detection', 'car'), # Change to detection
           ('tiny-yolov4', 'binary-classification', 'person'), 
           ('yolov4', 'count', 'car'), 
           ('tiny-yolov4', 'binary-classification', 'person'), 
           ('ssd-voc', 'aggregate-count', 'person')]

    #15
    w12 = [('faster-rcnn', 'count', 'car'), 
           ('tiny-yolov4', 'binary-classification', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('yolov4', 'count', 'car'), 
           ('tiny-yolov4', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'aggregate-count', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'count', 'car'), 
           ('ssd-voc', 'count', 'car'), 
           ('faster-rcnn', 'count', 'car'), 
           ('ssd-voc', 'binary-classification', 'car'), 
           ('yolov4', 'binary-classification', 'car'), 
           ('ssd-voc', 'binary-classification', 'car'), 
           ('ssd-voc', 'count', 'person'), 
           ('yolov4', 'count', 'person'), 
           ('yolov4', 'binary-classification', 'car'), 
           ('faster-rcnn', 'aggregate-count', 'person'), 
           ('ssd-voc', 'detection', 'car')]# Change to detection
    #6
    w13 = [('tiny-yolov4', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'count', 'person'), 
           ('faster-rcnn', 'count', 'person'), 
           ('tiny-yolov4', 'detection', 'car'),  # Change to detection
           ('tiny-yolov4', 'binary-classification', 'person'), 
           ('yolov4', 'detection', 'person'), # Change to detection
           ('faster-rcnn', 'count', 'person'), 
           ('yolov4', 'aggregate-count', 'person'), 
           ('ssd-voc', 'aggregate-count', 'person')]
    #5
    w14 = [('faster-rcnn', 'aggregate-count', 'person'), 
           ('faster-rcnn', 'count', 'car'), 
           ('faster-rcnn', 'count', 'person')]
       
    # 20 
    w15 = [('faster-rcnn', 'binary-classification', 'person'), 
           ('yolov4', 'count', 'car'), 
           ('faster-rcnn', 'count', 'car'), 
           ('ssd-voc', 'count', 'car')]
#    return [w1, w2, w3,w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15]
#    return [w1, w6, w7, w8, w9, w10, w11, w12, w13, w14]
    return [w2, w4, w15]



def repair_count_dicts( frame_begin,
                         frame_limit,
                         frame_to_model_to_orientation_to_car_count, 
                         frame_to_model_to_orientation_to_person_count,
                         blacklisted_frames
                         ):
    orientations = generate_all_orientations()
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        if f not in frame_to_model_to_orientation_to_car_count:
            if f not in blacklisted_frames:
                blacklisted_frames.append(f)
            continue
        for m in frame_to_model_to_orientation_to_car_count[f]:
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_car_count[f][m]:
                    # Repair car counts
                    if f > frame_begin + SKIP and o in frame_to_model_to_orientation_to_car_count[f-SKIP][m]:
                        prev_count = frame_to_model_to_orientation_to_car_count[f-SKIP][m][o] 
                    else:
                        prev_count = -1
                    if f < frame_limit + SKIP and o in frame_to_model_to_orientation_to_car_count[f+SKIP][m]:
                        future_count = frame_to_model_to_orientation_to_car_count[f-SKIP][m][o] 
                    else:
                        future_count = -1
                    if prev_count != -1 and future_count != -1:
                        frame_to_model_to_orientation_to_car_count[f][m][o] = (prev_count + future_count) / 2
                    elif prev_count == -1 and future_count != -1:
                        frame_to_model_to_orientation_to_car_count[f][m][o] = future_count
                    elif prev_count != -1 and future_count == -1:
                        frame_to_model_to_orientation_to_car_count[f][m][o] = prev_count
                    else:
                        frame_to_model_to_orientation_to_car_count[f][m][o] = 0

        for m in frame_to_model_to_orientation_to_car_count[f]:
            if f not in frame_to_model_to_orientation_to_person_count:
                if f not in blacklisted_frames:
                    blacklisted_frames.append(f)
                continue
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_person_count[f][m]:
                    # Repair ppl counts
                    if f > frame_begin + SKIP and o in frame_to_model_to_orientation_to_person_count[f-SKIP][m]:
                        prev_count = frame_to_model_to_orientation_to_person_count[f-SKIP][m][o] 
                    else:
                        prev_count = -1
                    if f < frame_limit + SKIP and o in frame_to_model_to_orientation_to_person_count[f+SKIP][m]:
                        future_count = frame_to_model_to_orientation_to_person_count[f-SKIP][m][o] 
                    else:
                        future_count = -1
                    if prev_count != -1 and future_count != -1:
                        frame_to_model_to_orientation_to_person_count[f][m][o] = (prev_count + future_count) / 2
                    elif prev_count == -1 and future_count != -1:
                        frame_to_model_to_orientation_to_person_count[f][m][o] = future_count
                    elif prev_count != -1 and future_count == -1:
                        frame_to_model_to_orientation_to_person_count[f][m][o] = prev_count
                    else:
                        frame_to_model_to_orientation_to_person_count[f][m][o] = 0


# Repair map dicts but only do so for a region at atime
def repair_map_dicts_for_region( frame_begin,
                         frame_limit,frame_to_model_to_orientation_to_car_map, 
                       frame_to_model_to_orientation_to_person_map,
                       orientations,
                       blacklisted_frames):
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        if f not in frame_to_model_to_orientation_to_car_map:
            if f not in blacklisted_frames:
                blacklisted_frames.append(f)
            continue
        for m in frame_to_model_to_orientation_to_car_map[f]:

            for o in orientations:
                if o not in frame_to_model_to_orientation_to_car_map[f][m]:
                    # Repair car counts
                    if f > frame_begin + SKIP and o in frame_to_model_to_orientation_to_car_map[f-SKIP][m]:
                        prev_count = frame_to_model_to_orientation_to_car_map[f-SKIP][m][o] 
                    else:
                        prev_map = -1
                    if f < frame_limit + SKIP and o in frame_to_model_to_orientation_to_car_map[f+SKIP][m]:
                        future_map = frame_to_model_to_orientation_to_car_map[f-SKIP][m][o] 
                    else:
                        future_map = -1
                    if prev_map != -1 and future_map != -1:
                        frame_to_model_to_orientation_to_car_map[f][m][o] = (prev_map + future_map) / 2
                    elif prev_map == -1 and future_map != -1:
                        frame_to_model_to_orientation_to_car_map[f][m][o] = future_map
                    elif prev_map != -1 and future_map == -1:
                        frame_to_model_to_orientation_to_car_map[f][m][o] = prev_map
                    else:
                        frame_to_model_to_orientation_to_car_map[f][m][o] = 0

        for m in frame_to_model_to_orientation_to_car_map[f]:
            if f not in frame_to_model_to_orientation_to_person_map:
                if f not in blacklisted_frames:
                    blacklisted_frames.append(f)
                continue
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_person_map[f][m]:
                    # Repair ppl maps
                    if f > frame_begin + SKIP and o in frame_to_model_to_orientation_to_person_map[f-SKIP][m]:
                        prev_map = frame_to_model_to_orientation_to_person_map[f-SKIP][m][o] 
                    else:
                        prev_map = -1
                    if f < frame_limit - SKIP and o in frame_to_model_to_orientation_to_person_map[f+SKIP][m]:
                        future_map = frame_to_model_to_orientation_to_person_map[f-SKIP][m][o] 
                    else:
                        future_map = -1
                    if prev_map != -1 and future_map != -1:
                        frame_to_model_to_orientation_to_person_map[f][m][o] = (prev_map + future_map) / 2
                    elif prev_map == -1 and future_map != -1:
                        frame_to_model_to_orientation_to_person_map[f][m][o] = future_map
                    elif prev_map != -1 and future_map == -1:
                        frame_to_model_to_orientation_to_person_map[f][m][o] = prev_map
                    else:
                        frame_to_model_to_orientation_to_person_map[f][m][o] = 0
            
            
def main():
    with open('config.yml') as f_params:
        params = yaml.safe_load(f_params)

    rectlinear_dir = params['rectlinear_dir']
    inference_dir = params['inference_dir']
    map_project_location = params['map_project_location']
    map_results_file = params['map_results_file']
    mot_results_dir = params['mot_results_dir']
    project_name = params['project_name']
    save_to_pkl = params['save_to_pkl']
    num_frames_to_send = params['num_frames_to_send']
    num_frames_to_explore = params['num_frames_to_explore']

    saved_data_dir = params['saved_data_dir']

    frame_bound_to_regions_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_bound_to_regions.pkl'
    frame_to_model_to_orientation_to_car_count_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_car_count.pkl'
    frame_to_model_to_orientation_to_person_count_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_person_count.pkl'

    frame_to_model_to_orientation_to_cars_detected_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_cars_detected.pkl'
    frame_to_model_to_orientation_to_people_detected_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_people_detected.pkl'
    frame_to_model_to_orientation_to_car_map_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_car_map.pkl'
    frame_to_model_to_orientation_to_person_map_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_person_map.pkl'

    frame_to_model_to_orientation_to_object_ids_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_object_ids.pkl'
    frame_to_model_to_orientation_to_object_id_to_mot_detected_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_object_id_to_mot_detected.pkl'

    frame_to_model_to_orientation_to_efficientdet_cars_detected_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_efficeintdet_cars_detected.pkl'

    frame_to_model_to_orientation_to_efficientdet_people_detected_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_efficientdet_people_detected.pkl'
    frame_to_model_to_orientation_to_object_id_to_mot_detected_file = f'{saved_data_dir}/{project_name}/{project_name}-frame_to_model_to_orientation_to_efficientdet_person_count.pkl'
    
    frame_bound_to_regions = {}

    frame_to_model_to_orientation_to_car_map = {}
    frame_to_model_to_orientation_to_person_map = {}

    frame_to_model_to_orientation_to_car_count = {}
    frame_to_model_to_orientation_to_person_count = {}

    frame_to_model_to_orientation_to_efficientdet_cars_detected = {}
    frame_to_model_to_orientation_to_efficientdet_people_detected = {}

    frame_to_model_to_orientation_to_cars_detected = {}
    frame_to_model_to_orientation_to_people_detected = {}

    frame_to_model_to_orientation_to_object_ids = {}
    frame_to_model_to_orientation_to_object_id_to_mot_detected = {}
    print('Reading files')
    # POpualte all dicts with stored data
    if os.path.exists(frame_bound_to_regions_file):
        with open(frame_bound_to_regions_file, 'rb') as f:
            frame_bound_to_regions = pickle.load(f)
    if os.path.exists(frame_to_model_to_orientation_to_car_count_file):
        with open(frame_to_model_to_orientation_to_car_count_file, 'rb') as f:
            frame_to_model_to_orientation_to_car_count = pickle.load(f)
    if os.path.exists(frame_to_model_to_orientation_to_person_count_file):
        with open(frame_to_model_to_orientation_to_person_count_file, 'rb') as f:
            frame_to_model_to_orientation_to_person_count = pickle.load(f)
    if os.path.exists(frame_to_model_to_orientation_to_car_count_file):
        with open(frame_to_model_to_orientation_to_car_count_file, 'rb') as f:
            frame_to_model_to_orientation_to_car_count = pickle.load(f)

    if os.path.exists(frame_to_model_to_orientation_to_person_count_file):
        with open(frame_to_model_to_orientation_to_person_count_file, 'rb') as f:
            frame_to_model_to_orientation_to_person_count = pickle.load(f)

    if os.path.exists(frame_to_model_to_orientation_to_efficientdet_cars_detected_file):
        with open(frame_to_model_to_orientation_to_efficientdet_cars_detected_file, 'rb') as f:
            frame_to_model_to_orientation_to_efficientdet_cars_detected = pickle.load(f)
    if os.path.exists(frame_to_model_to_orientation_to_efficientdet_people_detected_file):
        with open(frame_to_model_to_orientation_to_efficientdet_people_detected_file, 'rb') as f:
            frame_to_model_to_orientation_to_efficientdet_people_detected= pickle.load(f)

    if os.path.exists(frame_to_model_to_orientation_to_cars_detected_file):
        with open(frame_to_model_to_orientation_to_cars_detected_file, 'rb') as f:
            frame_to_model_to_orientation_to_cars_detected = pickle.load(f)
    if os.path.exists(frame_to_model_to_orientation_to_people_detected_file):
        with open(frame_to_model_to_orientation_to_people_detected_file, 'rb') as f:
            frame_to_model_to_orientation_to_people_detected = pickle.load(f)

    if os.path.exists(frame_to_model_to_orientation_to_object_ids_file):
        with open(frame_to_model_to_orientation_to_object_ids_file, 'rb') as f:
            frame_to_model_to_orientation_to_object_ids = pickle.load(f)

    if os.path.exists(frame_to_model_to_orientation_to_object_id_to_mot_detected_file):
        with open(frame_to_model_to_orientation_to_object_id_to_mot_detected_file, 'rb') as f:
            frame_to_model_to_orientation_to_object_id_to_mot_detected = pickle.load(f)

#    if not os.path.exists(frame_to_model_to_orientation_to_cars_detected_file) or not os.path.exists(frame_to_model_to_orientation_to_people_detected_file):
    print('Populating map ')
    populate_map_dicts(map_results_file, frame_to_model_to_orientation_to_car_map, frame_to_model_to_orientation_to_person_map)

    if not os.path.exists(frame_to_model_to_orientation_to_object_id_to_mot_detected_file) or not os.path.exists(frame_to_model_to_orientation_to_object_ids_file):
        frame_to_model_to_orientation_to_object_ids, frame_to_model_to_orientation_to_object_id_to_mot_detected = mot_helper.get_mot_info_from_directory(mot_results_dir)

    print('Evaluating workload')
    orientations = generate_all_orientations()
    workloads = init_workloads()
#    workloads = init_fig14_workloads()

#    worklaods = [ [('yolov4', 'detection', 'car'),
#('yolov4', 'detection', 'person')
#] ]

#    w1 = [('faster-rcnn', 'detection', 'person'),]
#    workloads = [(w1)]
    # 2
#    get_misc_results( inference_dir,
#                       workloads,
#                       orientations,
#                       frame_bound_to_regions,
#                       frame_to_model_to_orientation_to_car_count,
#                       frame_to_model_to_orientation_to_person_count,
#                       frame_to_model_to_orientation_to_cars_detected ,
#                       frame_to_model_to_orientation_to_people_detected,
#                       frame_to_model_to_orientation_to_car_map,
#                       frame_to_model_to_orientation_to_person_map,
#                       frame_to_model_to_orientation_to_object_ids,
#                       )


    blacklisted_frames = []

    frames = []
    for f in frame_to_model_to_orientation_to_car_map:
        frames.append(f)
    frames.sort()
    print('Map frames ', frames)

    print('stored frame bounds')
    for frame_begin, frame_limit in frame_bound_to_regions:
        print(frame_begin, ' -- ', frame_limit)
    for idx,(frame_begin, frame_limit) in enumerate(FRAME_BOUNDS):
        print(f'\tframes {frame_begin} -- {frame_limit}')
        # For now let object type be boht; perhaps change to 'both' in the future?
        object_type = 'any'
        tmp_frame = frame_begin
        populate_count_data = False

        if (frame_begin, frame_limit) in frame_bound_to_regions:
            regions = frame_bound_to_regions[(frame_begin, frame_limit)]
            repair_count_dicts( frame_begin,
                                     frame_limit,
                                     frame_to_model_to_orientation_to_car_count, 
                                     frame_to_model_to_orientation_to_person_count,
                                    blacklisted_frames
                                     )
        else:
            populate_count_dicts(inference_dir, frame_begin, frame_limit, frame_to_model_to_orientation_to_car_count, frame_to_model_to_orientation_to_person_count, frame_to_model_to_orientation_to_cars_detected, frame_to_model_to_orientation_to_people_detected)
            repair_count_dicts( frame_begin,
                                     frame_limit,
                                     frame_to_model_to_orientation_to_car_count, 
                                     frame_to_model_to_orientation_to_person_count,
                                    blacklisted_frames
                                     )
            print('Determining regions')
            regions = determine_regions(frame_begin, frame_begin + int(0.3*(frame_limit - frame_begin)),orientations,
                                  frame_to_model_to_orientation_to_car_count,
                                  frame_to_model_to_orientation_to_person_count,
                                  frame_to_model_to_orientation_to_car_map,
                                  frame_to_model_to_orientation_to_person_map,
                                  object_type)
        for r in regions:
#            map_compute.compute_map(frame_begin, 
#                        frame_limit, 
#                        inference_dir, 
#                        map_project_location, 
#                        map_results_file, 
#                        rectlinear_dir, 
#                        'car', 
#                        r,
#                        frame_to_model_to_orientation_to_car_map,
#                        frame_to_model_to_orientation_to_person_map,
#                        frame_to_model_to_orientation_to_cars_detected,
#                        frame_to_model_to_orientation_to_people_detected
#                        )
#
#            map_compute.compute_map(frame_begin, 
#                        frame_limit, 
#                        inference_dir, 
#                        map_project_location ,
#                        map_results_file, 
#                        rectlinear_dir, 
#                        'person', 
#                        r,
#                        frame_to_model_to_orientation_to_car_map,
#                        frame_to_model_to_orientation_to_person_map,
#                        frame_to_model_to_orientation_to_cars_detected,
#                        frame_to_model_to_orientation_to_people_detected
#                        )
            repair_map_dicts_for_region( frame_begin,
                                     frame_limit,frame_to_model_to_orientation_to_car_map, 
                                   frame_to_model_to_orientation_to_person_map, r, blacklisted_frames)

            inflate_map_scores(frame_begin, frame_limit, r, frame_to_model_to_orientation_to_car_count, 
                                   frame_to_model_to_orientation_to_person_count,

                       frame_to_model_to_orientation_to_cars_detected,
                       frame_to_model_to_orientation_to_people_detected,
                                   frame_to_model_to_orientation_to_car_map, 
                                   frame_to_model_to_orientation_to_person_map)
#        continue
#        print('finlizing regions')
#        for region_idx,r in enumerate(regions):
#            fixed_orientations = []
#            fixed_accuracies = []
#            dyn_accuracies = []
#            for w_idx,w in enumerate(workloads):
#                gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(w, frame_begin, frame_limit, r, frame_to_model_to_orientation_to_object_ids)
#                #### Initial check to see if using different orientations works. If not, skip region
#                init_best_dynamic_score = find_best_dynamic_score(w,
#                                    frame_begin,
#                                    frame_limit,
#                                    r,
#                                    frame_to_model_to_orientation_to_car_count,
#                                    frame_to_model_to_orientation_to_person_count,
#                                    frame_to_model_to_orientation_to_car_map,
#                                    frame_to_model_to_orientation_to_person_map,
#                                    frame_to_model_to_orientation_to_object_ids,
#                                    gt_model_to_object_ids)
#
#                init_fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(w,
#                                    frame_begin,
#                                    frame_limit,
#                                    r,
#                                    frame_to_model_to_orientation_to_car_count,
#                                    frame_to_model_to_orientation_to_person_count,
#                                    frame_to_model_to_orientation_to_car_map,
#                                    frame_to_model_to_orientation_to_person_map,
#                                    frame_to_model_to_orientation_to_object_ids,
#                                    gt_model_to_object_ids
#                                    )
#                dyn_accuracies.append(init_best_dynamic_score)
#                fixed_accuracies.append(init_fixed_score)
#                fixed_orientations.append(best_fixed_orientation)
#            avg_dyn_accuracy = sum(dyn_accuracies) / len(dyn_accuracies)
#            avg_fixed_accuracy = sum(fixed_accuracies) / len(fixed_accuracies)
        frame_bound_to_regions[(frame_begin, frame_limit)] = regions


    print('Blacklisted frames ', blacklisted_frames)
    if save_to_pkl:
        with open(frame_bound_to_regions_file, 'wb') as f:
            pickle.dump(frame_bound_to_regions, f)
        with open(frame_to_model_to_orientation_to_car_count_file, 'wb') as f:
            pickle.dump(frame_to_model_to_orientation_to_car_count, f)
        with open(frame_to_model_to_orientation_to_person_count_file, 'wb') as f:
            pickle.dump(frame_to_model_to_orientation_to_person_count, f)

        with open(frame_to_model_to_orientation_to_efficientdet_cars_detected_file, 'wb') as f:
            pickle.dump(frame_to_model_to_orientation_to_efficientdet_cars_detected , f)

        with open(frame_to_model_to_orientation_to_efficientdet_people_detected_file, 'wb') as f:
            pickle.dump(frame_to_model_to_orientation_to_efficientdet_people_detected, f)

        with open(frame_to_model_to_orientation_to_cars_detected_file, 'wb') as f:
            pickle.dump(frame_to_model_to_orientation_to_cars_detected, f)
        with open(frame_to_model_to_orientation_to_people_detected_file, 'wb') as f:
            pickle.dump(frame_to_model_to_orientation_to_people_detected, f)
        if len(frame_to_model_to_orientation_to_car_map) > 0:
            with open(frame_to_model_to_orientation_to_car_map_file, 'wb') as f:
                pickle.dump(frame_to_model_to_orientation_to_car_map_file, f)
        if len(frame_to_model_to_orientation_to_person_map) > 0:
            with open(frame_to_model_to_orientation_to_person_map_file, 'wb') as f:
                pickle.dump(frame_to_model_to_orientation_to_person_map_file, f)


        if len(frame_to_model_to_orientation_to_object_ids) > 0:
            with open(frame_to_model_to_orientation_to_object_ids_file, 'wb') as f:
                pickle.dump(frame_to_model_to_orientation_to_object_ids, f)

        if len(frame_to_model_to_orientation_to_object_id_to_mot_detected) > 0:
            with open(frame_to_model_to_orientation_to_object_id_to_mot_detected_file, 'wb') as f:
                pickle.dump(frame_to_model_to_orientation_to_object_id_to_mot_detected, f)

    object_id_to_frame_to_model_to_orientations = mot_helper.get_object_presence_info_from_mot_files(frame_to_model_to_orientation_to_object_ids)

#    get_baseline_results(workloads,
#                       frame_bound_to_regions,
#                       frame_to_model_to_orientation_to_car_count,
#                       frame_to_model_to_orientation_to_person_count,
#                       frame_to_model_to_orientation_to_car_map,
#                       frame_to_model_to_orientation_to_person_map,
#                       frame_to_model_to_orientation_to_object_ids,
#                       frame_to_model_to_orientation_to_object_id_to_mot_detected,
#                       object_id_to_frame_to_model_to_orientations,
#                      blacklisted_frames
#                       )

#    get_initial_results(workloads,
#                       frame_bound_to_regions,
#                       frame_to_model_to_orientation_to_car_count,
#                       frame_to_model_to_orientation_to_person_count,
#                       frame_to_model_to_orientation_to_car_map,
#                       frame_to_model_to_orientation_to_person_map,
#                       frame_to_model_to_orientation_to_object_ids,
#                       blacklisted_frames
#                       )
#

#    print('blacklisted frames ', blacklisted_frames)
#    get_paper_results( frame_bound_to_regions,
#                       frame_to_model_to_orientation_to_car_count,
#                       frame_to_model_to_orientation_to_person_count,
#                       frame_to_model_to_orientation_to_car_map,
#                       frame_to_model_to_orientation_to_person_map,
#                       frame_to_model_to_orientation_to_object_ids,
#                        blacklisted_frames
#                       )
    get_madeye_results(project_name, inference_dir,
                       rectlinear_dir,
                       params,
                        workloads,
                       frame_bound_to_regions,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_cars_detected,
                       frame_to_model_to_orientation_to_people_detected,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                        frame_to_model_to_orientation_to_efficientdet_cars_detected,
                        frame_to_model_to_orientation_to_efficientdet_people_detected ,
                        blacklisted_frames,
                        num_frames_to_send=num_frames_to_send,
                        num_frames_to_explore=num_frames_to_explore
                       )


if __name__ == '__main__':
    main()
