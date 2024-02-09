

import evaluation_tools
import random
import math
import numpy as np
MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']
SKIP = 6 # Only consider frames where frame % SKIP == 0
PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0

from madeye_utils import parse_orientation_string, extract_pan, extract_tilt, extract_zoom, find_tilt_dist, find_pan_dist

SKIP = 6

def ewma(l):
    alpha = 0.7
    if len(l) == 0:
        return 1
    if len(l) == 1:
        return l[0]
    return alpha * l[-1] + (1-alpha) * ewma(l[:-1])

def mab_baseline(workload,
               frame_begin,
               frame_limit,
               orientations,
               frame_to_model_to_orientation_to_car_count,
               frame_to_model_to_orientation_to_person_count,
               frame_to_model_to_orientation_to_car_map,
               frame_to_model_to_orientation_to_person_map,
               frame_to_model_to_orientation_to_object_ids,
               gt_model_to_object_ids,
               blacklisted_frames=[]
               ):

    epsilon = 0.15

    car_query_weight = evaluation_tools.num_car_queries_in_workload(workload) / len(workload)
    person_query_weight = 1.0 - car_query_weight

    mab_num_frames = 0
    num_frames = 0
    assert len(orientations) > 0
    current_orientation = orientations[0]
    orientation_to_scores_observed = {}

    running_non_aggregate_accuracy = 0.0
    model_to_object_ids_found = {}
    orientation_to_arm_pulls = {}
    highest_score = 0.0
    window_size = 5
    for f in range(frame_begin, frame_begin + int(0.3*(frame_limit - frame_begin))):
        # Look at first frame and pull all arms
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        for o in orientations:
            car_count = 0
            people_count = 0    
            for q in workload:
                car_count += frame_to_model_to_orientation_to_car_count[f][q[0]][o]
                people_count += frame_to_model_to_orientation_to_person_count[f][q[0]][o]
            # TODO:CHange this computation once Mike's formula is done
            count_score = car_query_weight * car_count + person_query_weight * people_count

            if o not in orientation_to_scores_observed:
                orientation_to_scores_observed[o] = []
            orientation_to_scores_observed[o].append(count_score)
            if o not in orientation_to_arm_pulls:
                orientation_to_arm_pulls[o] = 0
            orientation_to_arm_pulls[o] += 1
            mab_num_frames += 1
        break

    for f in range(frame_begin + int(0.3*(frame_limit - frame_begin)), frame_limit):
        if f % SKIP != 0:
            continue
        if random.random() <= epsilon:
            # Explore
            current_orientation = random.choice(orientations)
        else:
            highest_score = 0.0
            for o in orientations:
                scores = orientation_to_scores_observed[o][ -min(window_size, len(orientation_to_scores_observed[o])-1): ]
                current_orientation_average_reward = ewma(scores)
                ucb = current_orientation_average_reward + math.sqrt( 2* np.log(mab_num_frames) / orientation_to_arm_pulls[o])
                if current_orientation_average_reward > highest_score:
                    highest_score = current_orientation_average_reward
                    current_orientation = o

        current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, current_orientation, orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, current_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        running_non_aggregate_accuracy += current_non_aggregate_accuracy
        num_frames += 1
        
        car_count = 0
        people_count = 0    
        for q in workload:
            car_count += frame_to_model_to_orientation_to_car_count[f][q[0]][current_orientation]
            people_count += frame_to_model_to_orientation_to_person_count[f][q[0]][current_orientation]
        # TODO:CHange this computation once Mike's formula is done
        count_score = car_query_weight * car_count + person_query_weight * people_count

        if current_orientation not in orientation_to_scores_observed:
            orientation_to_scores_observed[current_orientation] = []
        orientation_to_scores_observed[current_orientation].append(count_score)

    non_aggregate_accuracy = running_non_aggregate_accuracy  / num_frames
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * non_aggregate_accuracy
    return total_accuracy

def select_id_to_track(object_ids, object_id_to_mot_detected):
    max_area = 0
    best_obj_id = -1
    if len(object_ids) > 0:
        best_obj_id = object_ids[0]
    for obj_id in object_ids:
        width = object_id_to_mot_detected[obj_id][2] - object_id_to_mot_detected[obj_id][0]
        height = object_id_to_mot_detected[obj_id][3] - object_id_to_mot_detected[obj_id][1]
        area = width * height
        if area > max_area:
            max_area = area
            best_obj_id = obj_id
    return best_obj_id

def ptz_tracking(workload,
               frame_begin,
               frame_limit,
               orientations,
               anchor_orientation,
               frame_to_model_to_orientation_to_car_count,
               frame_to_model_to_orientation_to_person_count,
               frame_to_model_to_orientation_to_car_map,
               frame_to_model_to_orientation_to_person_map,
               frame_to_model_to_orientation_to_object_ids,
               frame_to_model_to_orientation_to_object_id_to_mot_detected,
               object_id_to_frame_to_model_to_orientations,
               gt_model_to_object_ids,
               blacklisted_frames=[]
               ):

    running_non_aggregate_accuracy = 0.0
    model_to_object_ids_found = {}
    current_formation = []
    orientation_to_historical_scores = {}

    car_query_weight = evaluation_tools.num_car_queries_in_workload(workload) / len(workload)
    person_query_weight = 1.0 - car_query_weight
    num_frames = 0
    orientation_idx = 0
    current_duration = 0
    id_being_tracked = -1
    model_for_tracking = 'faster-rcnn'
    current_orientation = anchor_orientation
    for f in range(frame_begin, frame_limit):
    
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        if id_being_tracked == -1:
            # CUrrent id is bad , get a new one
            current_orientation = anchor_orientation
            if f in frame_to_model_to_orientation_to_object_ids and model_for_tracking in frame_to_model_to_orientation_to_object_ids[f]:
                if current_orientation in frame_to_model_to_orientation_to_object_ids[f][model_for_tracking]:
                    if len(frame_to_model_to_orientation_to_object_ids[f][model_for_tracking][current_orientation]) == 0:
                        id_being_tracked = -1
                    else:
                        id_being_tracked = select_id_to_track(frame_to_model_to_orientation_to_object_ids[f][model_for_tracking][current_orientation], 
                            frame_to_model_to_orientation_to_object_id_to_mot_detected[f][model_for_tracking][current_orientation])
        else:
            if f in object_id_to_frame_to_model_to_orientations[id_being_tracked] and len(object_id_to_frame_to_model_to_orientations[id_being_tracked]) > 0 and model_for_tracking in object_id_to_frame_to_model_to_orientations[id_being_tracked][f]:
                current_orientation = object_id_to_frame_to_model_to_orientations[id_being_tracked][f][model_for_tracking][0]
            else:
                id_being_tracked = -1
        current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, current_orientation, orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, current_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        running_non_aggregate_accuracy += current_non_aggregate_accuracy
        num_frames += 1
        current_duration += 1

    non_aggregate_accuracy = running_non_aggregate_accuracy  / num_frames
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * non_aggregate_accuracy
    return total_accuracy

def run_panoptes(workload,
               frame_begin,
               frame_limit,
               orientations,
               anchor_orientation,
               frame_to_model_to_orientation_to_car_count,
               frame_to_model_to_orientation_to_person_count,
               frame_to_model_to_orientation_to_car_map,
               frame_to_model_to_orientation_to_person_map,
               frame_to_model_to_orientation_to_object_ids,
               gt_model_to_object_ids
               ):


    orientation_to_durations = {}
    panoptes_orientations = []
    # Get top k orientations to cycle between
    k = 4

    remaining_orientations = orientations.copy()
    # First get all best fixed orientations
    for i in range(k,0,-1):
        fixed_score, best_fixed_orientation = evaluation_tools.find_best_fixed(workload,
                            frame_begin,
                            frame_begin +  int(0.3*(frame_limit - frame_begin)),
                            remaining_orientations,
                            frame_to_model_to_orientation_to_car_count,
                            frame_to_model_to_orientation_to_person_count,
                            frame_to_model_to_orientation_to_car_map,
                            frame_to_model_to_orientation_to_person_map,
                            frame_to_model_to_orientation_to_object_ids,
                            gt_model_to_object_ids,
                            blacklisted_frames=[]
                            )
        remaining_orientations.remove(best_fixed_orientation)
        panoptes_orientations.append(best_fixed_orientation)
        orientation_to_durations[best_fixed_orientation] = i**2

    print('Panoptes orientations ', panoptes_orientations)
    print('durations ', orientation_to_durations)
    running_non_aggregate_accuracy = 0.0
    model_to_object_ids_found = {}
    current_formation = []
    orientation_to_historical_scores = {}

    car_query_weight = evaluation_tools.num_car_queries_in_workload(workload) / len(workload)
    person_query_weight = 1.0 - car_query_weight
    num_frames = 0
    orientation_idx = 0
    current_duration = 0
    current_orientation = panoptes_orientations[0]
    for f in range(frame_begin +  int(0.3*(frame_limit - frame_begin)), frame_limit):
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        schedule_interrupted = False
        prior_current_orientation = current_orientation
        largest_diff = 0
        new_orientation = current_orientation
        for o in panoptes_orientations:
            # CHeck if there's activity in any other orientation
            if o == current_orientation:
                continue

            prev_car_count = 0
            prev_people_count = 0    
            current_car_count = 0
            current_people_count = 0    
            for q in workload:
                current_car_count += frame_to_model_to_orientation_to_car_count[f][q[0]][o]
                current_people_count += frame_to_model_to_orientation_to_person_count[f][q[0]][o]
                prev_car_count += frame_to_model_to_orientation_to_car_count[f-SKIP][q[0]][o]
                prev_people_count += frame_to_model_to_orientation_to_person_count[f-SKIP][q[0]][o]

            prev_count_score = car_query_weight * prev_car_count + person_query_weight * prev_people_count
            current_count_score = car_query_weight * current_car_count + person_query_weight * current_people_count
            if current_count_score > prev_count_score and current_count_score - prev_count_score > largest_diff:
                new_orientation = o
                largest_diff = current_count_score - prev_count_score
                schedule_interrupted = True
        if schedule_interrupted:
            current_orientation = new_orientation

        if not schedule_interrupted and current_duration >= orientation_to_durations[current_orientation]:
            # Go to new orientation according to the schedule
            orientation_idx = (orientation_idx + 1) % len(panoptes_orientations)
            current_orientation = panoptes_orientations[orientation_idx]
        current_orientation = panoptes_orientations[orientation_idx]
        current_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, current_orientation, orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, current_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        running_non_aggregate_accuracy += current_non_aggregate_accuracy
        num_frames += 1
        if not schedule_interrupted:
            current_duration += 1
        else:
            current_orientation = prior_current_orientation

    non_aggregate_accuracy = running_non_aggregate_accuracy  / num_frames
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * non_aggregate_accuracy
    return total_accuracy
