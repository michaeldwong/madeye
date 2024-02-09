
import pandas as pd
import random
import json
import os
import argparse
import mot_helper
import time
import map_compute
import math
from DetectedObject import DetectedObject
MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']



SKIP = 6 # Only consider frames where frame % SKIP == 0
PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0

# this returns the weighted sum of car and person count. weight based on queries in workload. 
# if workload contains no people, this returns count of cars
# if workload contains no cars, this returns count of people 
# if workload contains half people and half cars, this return weight sum of both with weight = 0.5
def get_count_of_orientation(workload, 
                     current_frame,
                     current_orientation,
                     orientations,
                     frame_to_model_to_orientation_to_car_count,
                     frame_to_model_to_orientation_to_person_count,
                     frame_to_model_to_orientation_to_car_map,
                     frame_to_model_to_orientation_to_person_map):
    num_car_queries = 0
    num_person_queries = 0
    car_count = 0
    person_count = 0
    for q in workload:
        if q[2] == 'car':
            num_car_queries += 1
        elif q[2] == 'person':
            num_person_queries += 1
    
    car_count = frame_to_model_to_orientation_to_car_count[current_frame][q[0]][current_orientation]
    person_count = frame_to_model_to_orientation_to_person_count[current_frame][q[0]][current_orientation]

    car_weight = (num_car_queries / (num_person_queries + num_car_queries))
    person_weight = (num_person_queries / (num_person_queries + num_car_queries))
    return ( (car_weight * car_count) + (person_weight * person_count) )


def get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(list_of_bounding_boxes):
    class BoundingBox:
        def __init__(self, x0, y0, x1, y1):
            if x0 > x1:
                tmp = x0
                x0 = x1
                x1 = tmp
            if y0 > y1:
                tmp = y0
                y0 = y1
                y1 = tmp
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1

        def __str__(self) -> str:
            return f"{self.x0},{self.y0},{self.x1},{self.y1}"   

        def get_area(self):
            return (self.x1 - self.x0) * (self.y1 - self.y0)
    rectangles = []
    for obj in list_of_bounding_boxes:
        if isinstance(obj, DetectedObject):
            rectangles.append(BoundingBox(obj.left, obj.top, obj.right, obj.bottom) )
        else:
            rectangles.append(BoundingBox(obj[0], obj[1], obj[2], obj[3]))
    total_area = 0
    while rectangles:
        rect = rectangles.pop()
        total_area += rect.get_area()
        for other in rectangles:
            if rect.x0 < other.x1 and rect.x1 > other.x0 and rect.y0 < other.y1 and rect.y1 > other.y0:
                overlap = BoundingBox(
                    max(rect.x0, other.x0),
                    max(rect.y0, other.y0),
                    min(rect.x1, other.x1),
                    min(rect.y1, other.y1),
                )
                total_area -= overlap.get_area()
                rectangles.remove(other)
                rectangles.append(overlap)

    return total_area

def get_muralis_mike_factor(workload, 
                     current_frame,
                     current_orientation,
                     frames_since_last_visited,
                     model_to_orientation_to_efficientdet_car_count,
                     model_to_orientation_to_efficientdet_person_count,
                     model_to_orientation_to_efficientdet_car_boxes,
                     model_to_orientation_to_efficientdet_person_boxes):
    num_car_queries = 0
    num_person_queries = 0
    car_count = 0
    person_count = 0
    num_classification_queries = 0
    num_count_queries = 0
    num_detection_queries = 0
    num_agg_queries = 0
    # MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']
    # ssd = 1
    # frcnn = 0.7 for cars. 0.5 people. 
    # yolo = 0.8 for cars. 0.6 for people. 
    # tiny-yolo = 0.9 for cars. 0.8 for people.

    
    for q in workload:
        if q[2] == 'car':
            if q[0] == "ssd-voc":
                num_car_queries += 1
            elif q[0] == "yolov4":
                num_car_queries += 0.8
            elif q[0] == "tiny-yolov4":
                num_car_queries += 0.9
            else:
                num_car_queries += 0.7
        elif q[2] == 'person':
            if q[0] == "ssd-voc":
                num_person_queries += 1
            elif q[0] == "yolov4":
                num_person_queries += 0.6
            elif q[0] == "tiny-yolov4":
                num_person_queries += 0.8
            else:
                num_person_queries += 0.5
    for q in workload:
        if q[1] == 'binary-classification':
            num_classification_queries += 1
        elif q[1] == 'count':
            num_count_queries += 1
        elif q[1] == 'detection':
            num_detection_queries += 1
        elif q[1] == 'aggregate-count':
            num_agg_queries += 1

    num_faster_rcnn_queries = 0
    num_yolov4_queries = 0
    num_tiny_yolov4_queries = 0
    num_ssd_voc_queries = 0
    for q in workload:
        if q[0] == 'yolov4':
            num_yolov4_queries += 1
        elif q[0] == 'faster-rcnn':
            num_faster_rcnn_queries += 1
        elif q[0] == 'ssd-voc':
            num_ssd_voc_queries += 1
        elif q[0] == 'tiny-yolov4':
            num_tiny_yolov4_queries += 1
    car_count = 0
    person_count = 0
    person_cumulative_area = 0
    car_cumulative_area = 0
    # input(model_to_orientation_to_efficientdet_person_boxes)
    total_queries = num_faster_rcnn_queries + num_yolov4_queries + num_tiny_yolov4_queries + num_ssd_voc_queries
    if 'yolov4' in model_to_orientation_to_efficientdet_car_count:
        yolov4_weight = num_yolov4_queries / total_queries
        car_count += yolov4_weight *  model_to_orientation_to_efficientdet_car_count['yolov4'][current_orientation]
        person_count += yolov4_weight * model_to_orientation_to_efficientdet_person_count['yolov4'][current_orientation]
        try:
            person_cumulative_area += yolov4_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_person_boxes['yolov4'][current_orientation])
            car_cumulative_area += yolov4_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_car_boxes['yolov4'][current_orientation])
        except KeyError:
            # the orientation does not have any boxes
            pass
    if 'tiny-yolov4' in model_to_orientation_to_efficientdet_car_count:
        tiny_yolov4_weight = num_tiny_yolov4_queries / total_queries
        car_count += tiny_yolov4_weight *  model_to_orientation_to_efficientdet_car_count['tiny-yolov4'][current_orientation]
        person_count += tiny_yolov4_weight * model_to_orientation_to_efficientdet_person_count['tiny-yolov4'][current_orientation]
        try:
            person_cumulative_area += tiny_yolov4_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_person_boxes['tiny-yolov4'][current_orientation])
            car_cumulative_area += tiny_yolov4_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_car_boxes['tiny-yolov4'][current_orientation])
        except KeyError:
            # the orientation does not have any boxes
            pass
    if 'ssd-voc' in model_to_orientation_to_efficientdet_car_count:
        ssd_voc_weight = num_ssd_voc_queries / total_queries
        car_count += ssd_voc_weight *  model_to_orientation_to_efficientdet_car_count['ssd-voc'][current_orientation]
        person_count += ssd_voc_weight * model_to_orientation_to_efficientdet_person_count['ssd-voc'][current_orientation]
        try:
            person_cumulative_area += ssd_voc_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_person_boxes['ssd-voc'][current_orientation])
            car_cumulative_area += ssd_voc_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_car_boxes['ssd-voc'][current_orientation])
        except KeyError:
            # the orientation does not have any boxes
            pass
    if 'faster-rcnn' in model_to_orientation_to_efficientdet_car_count:
        faster_rcnn_weight = num_faster_rcnn_queries / total_queries
        car_count += faster_rcnn_weight *  model_to_orientation_to_efficientdet_car_count['faster-rcnn'][current_orientation]
        person_count += faster_rcnn_weight * model_to_orientation_to_efficientdet_person_count['faster-rcnn'][current_orientation]
        try:
            person_cumulative_area += faster_rcnn_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_person_boxes['faster-rcnn'][current_orientation])
            car_cumulative_area += faster_rcnn_weight * get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(model_to_orientation_to_efficientdet_car_boxes['faster-rcnn'][current_orientation])
        except KeyError:
            # the orientation does not have any boxes
            pass
    alpha = num_classification_queries / len(workload)
    m1 = 1 if car_count > 0 else 0
    w1 = num_car_queries / len(workload)
    m2 = 1 if person_count > 0 else 0
    w2 = num_person_queries / len(workload)
    beta = (num_count_queries) / len(workload)
    gamma = (num_agg_queries) / len(workload)
    delta = (num_detection_queries) / len(workload)
    T = (3 + min(frames_since_last_visited, 3))/6
    result_metric = (alpha * m1 * w1) + (alpha * m2 * w2) + (beta * w1 * car_count) + (beta * w2 * person_count) +  (beta * w1 * car_count) + (beta * w2 * person_count) +  (delta * w1 * car_cumulative_area) + (delta * w2 * person_cumulative_area) + (gamma * w2 * person_count * T)
    # input(f"result is {result_metric}")
    return result_metric



def best_dynamic_aggregate_ids(workload, frame_begin, frame_limit, orientations, 
                               frame_to_model_to_orientation_to_object_ids):
    
    # FInd best dynamic list for each aggregate query/model, this is used as the denominator when evaluating aggregate
    # queries 
    gt_model_to_object_ids = {}

    for q in workload:
        if q[1] != 'aggregate-count' or q[0] in gt_model_to_object_ids:
            continue
        current_ids_found = []
        for f in range(frame_begin, frame_limit+1):
            if f % SKIP != 0:
                continue
            if f not in frame_to_model_to_orientation_to_object_ids:
                continue
            if q[0] not in frame_to_model_to_orientation_to_object_ids[f]:
                continue
            best_list = [] 
            for o in orientations:
                if o not in frame_to_model_to_orientation_to_object_ids[f][q[0]]:
                    continue
                new_ids = frame_to_model_to_orientation_to_object_ids[f][q[0]][o]
                new_ids = [x for x in new_ids if x not in current_ids_found]
                if len(new_ids) > len(best_list):
                    best_list = new_ids
            current_ids_found.extend(best_list)
        gt_model_to_object_ids[q[0]] = current_ids_found
    return gt_model_to_object_ids


# Find best orientation and score for given frame
def find_best_score_and_orientation(current_frame, 
                                    model, 
                                    orientations, 
                                    frame_to_model_to_orientation_to_something):
    max_result = -0.0
    max_orientation = orientations[0]
    for o in orientations:
        if o in frame_to_model_to_orientation_to_something[current_frame][model] and frame_to_model_to_orientation_to_something[current_frame][model][o] > max_result:
            max_result = frame_to_model_to_orientation_to_something[current_frame][model][o]
            max_orientation = o
    return max_result, max_orientation

# get mike factor
def get_mikes_mike_factor(workload, 
                     current_frame,
                     current_orientation,
                     current_formation,
                     model_to_orientation_to_efficientdet_car_count,
                     model_to_orientation_to_efficientdet_person_count,
                     model_to_orientation_to_efficientdet_cars_detected,
                     model_to_orientation_to_efficientdet_people_detected,
                     orientation_to_frames_since_last_visit,
                     orientation_to_visits
                     ):

    total_frames = 0
    for o in orientation_to_visits:
        total_frames += orientation_to_visits[o]
    car_detection_count_weight = 0.98
    car_detection_bbox_weight = 0.02

    person_detection_count_weight = 0.96
    person_detection_bbox_weight = 0.04
    query_to_best_count = {}
    query_to_max_area = {}
    for q in workload:
        if q[1] == 'binary-classification':
            max_count = 0
            if q[2] == 'car':
                for o in current_formation:
                    count = model_to_orientation_to_efficientdet_car_count[q[0]][o] 
                    if count > 0:
                        max_count = 1
                        break
            elif q[2] == 'person':
                for o in current_formation:
                    count = model_to_orientation_to_efficientdet_person_count[q[0]][o] 
                    if count > 0:
                        max_count = 1
                        break
            query_to_best_count[q] = max_count

        elif q[1] == 'count':
            max_count = 0
            if q[2] == 'car':
                for o in current_formation:
                    count = model_to_orientation_to_efficientdet_car_count[q[0]][o] 
                    if count > max_count:
                        max_count = count
            elif q[2] == 'person':
                for o in current_formation:
                    count = model_to_orientation_to_efficientdet_person_count[q[0]][o] 
                    if count > max_count:
                        max_count = count
            query_to_best_count[q] = max_count

        elif q[1] == 'detection':
            if q[2] == 'car':
                max_total_area = 0.0
                max_count = 0
                for o in current_formation:
                    total_area = 0.0
                    for d in model_to_orientation_to_efficientdet_cars_detected[q[0]][o]:
                        total_area += (d[2] - d[0]) * (d[3] - d[1])
                    if total_area > max_total_area:
                        max_total_area = total_area

                    count = model_to_orientation_to_efficientdet_car_count[q[0]][o] 
                    if count > max_count:
                        max_count = count
                detection_count_weight = car_detection_count_weight
                detection_bbox_weight = car_detection_bbox_weight
            elif q[2] == 'person':
                max_total_area = 0.0
                max_count = 0
                for o in current_formation:
                    # Get total bbox area
                    total_area = 0.0
                    for d in model_to_orientation_to_efficientdet_people_detected[q[0]][o]:
                        total_area += (d[2] - d[0]) * (d[3] - d[1])
                    if total_area > max_total_area:
                        max_total_area = total_area
                    # GEt count
                    count = model_to_orientation_to_efficientdet_person_count[q[0]][o] 
                    if count > max_count:
                        max_count = count
                detection_count_weight = person_detection_count_weight
                detection_bbox_weight = person_detection_bbox_weight
            query_to_best_count[q] = max_count 
            query_to_max_area[q] = max_total_area
        elif q[1] == 'aggregate-count':
            max_count = 0
            if q[2] == 'car':
                for o in current_formation:
                    count = model_to_orientation_to_efficientdet_car_count[q[0]][o] 
                    if count > max_count:
                        max_count = count
            elif q[2] == 'person':
                for o in current_formation:
                    count = model_to_orientation_to_efficientdet_person_count[q[0]][o] 
                    if count > max_count:
                        max_count = count
            query_to_best_count[q] = max_count

    score_sum = 0.0
    for q in workload:
        current_score = 0.0
        if q[1] == 'binary-classification':
            if q[2] == 'car':
                if model_to_orientation_to_efficientdet_car_count[q[0]][current_orientation] > 0:
                    current_score = 1.0
                else:
                    current_score = 0.0
            elif q[2] == 'person':
                if model_to_orientation_to_efficientdet_person_count[q[0]][current_orientation] > 0:
                    current_score = 1.0
                else:
                    current_score = 0.0
        elif q[1] == 'count' :
            if q[2] == 'car':
                count = model_to_orientation_to_efficientdet_car_count[q[0]][current_orientation] 
                if query_to_best_count[q] > 0:
                    current_score = count / query_to_best_count[q]
            elif q[2] == 'person':
                count = model_to_orientation_to_efficientdet_person_count[q[0]][current_orientation] 
                if query_to_best_count[q] > 0:
                    current_score = count / query_to_best_count[q]
        elif q[1] == 'detection':
            if q[2] == 'car':
                total_area = 0.0
                for d in model_to_orientation_to_efficientdet_cars_detected[q[0]][current_orientation]:
                    total_area += (d[2] - d[0]) * (d[3] - d[1])
                count = model_to_orientation_to_efficientdet_car_count[q[0]][current_orientation] 
                detection_count_weight = car_detection_count_weight
                detection_bbox_weight = car_detection_bbox_weight
                area_weight = 0.4
                count_weight = 0.6

#                area_weight = 0.3
#                count_weight = 0.7
            elif q[2] == 'person':
                # Get total bbox area
                total_area = 0.0
                for d in model_to_orientation_to_efficientdet_people_detected[q[0]][current_orientation]:
                    total_area += (d[2] - d[0]) * (d[3] - d[1])
                # GEt count
                count = model_to_orientation_to_efficientdet_person_count[q[0]][current_orientation] 
                detection_count_weight = person_detection_count_weight
                detection_bbox_weight = person_detection_bbox_weight
                area_weight = 0.3
                count_weight = 0.7
            if query_to_best_count[q] > 0:
                count_score = count / query_to_best_count[q]
            else:
                count_score = 0.0
            if query_to_max_area[q] > 0:
                area_score = total_area / query_to_max_area[q]
            else:
                area_score = 0.0
            current_score = count_score * count_weight + area_score *area_weight 
        elif q[1] == 'aggregate-count':
            if q[2] == 'car':
                count = model_to_orientation_to_efficientdet_car_count[q[0]][current_orientation] 
                if query_to_best_count[q] > 0:
                    current_score = count / query_to_best_count[q]
            elif q[2] == 'person':
                count = model_to_orientation_to_efficientdet_person_count[q[0]][current_orientation] 
                if query_to_best_count[q] > 0:
                    current_score = count / query_to_best_count[q]

            if current_orientation not in orientation_to_visits:
                orientation_to_visits[current_orientation] = 0
            if total_frames > 10:
                if orientation_to_visits[current_orientation] / total_frames < 0.01:
                    current_score *= 1.4
                elif orientation_to_visits[current_orientation] / total_frames < 0.04:
                    current_score *= 1.2
                elif orientation_to_visits[current_orientation] / total_frames < 0.07:
                    current_score *= 1.1

#            if current_frame in orientation_to_frames_since_last_visit:
#                print('Weighing orientation ', current_orientation, ' at frame ', current_frame, ' with weight ', (15 + min(orientation_to_frames_since_last_visit[o],5 ))/ 20)
#                current_score *=  (15 + min(orientation_to_frames_since_last_visit[current_frame],5 ))/ 20
#        print('\tScore = ', current_score)
        score_sum += current_score
    return score_sum / len(workload)



def compute_when_an_orientation_was_last_visited(trace_of_our_orientations, current_formation):
    num_prior_shapes = len(trace_of_our_orientations)
    orientation_to_last_seen = {}
    for o in current_formation:
        orientation_to_last_seen[o] = 6
    if num_prior_shapes < 1:
        return orientation_to_last_seen # the formula we use for mike's metric prioritize orientations we have not seen before 
        # and not seeing an orientation for 6 seconds gives the max benefit. there isn't any higher preference to an orientation that
        # was seen 6 seconds ago vs 7 seconds ago
    
    index = num_prior_shapes - 1
    while index >= 0:
        for o in orientation_to_last_seen:
            if o in set(trace_of_our_orientations[index]):
                orientation_to_last_seen[o] = num_prior_shapes - index - 1
        index -= 1

    return orientation_to_last_seen


## get mike factor
def get_mike_factor(workload, 
                     current_frame,
                     current_orientation,
                     orientations,
                     frame_to_model_to_orientation_to_car_count,
                     frame_to_model_to_orientation_to_person_count,
                     frame_to_model_to_orientation_to_car_map,
                     frame_to_model_to_orientation_to_person_map):
    num_car_queries = 0
    num_person_queries = 0
    car_count = 0
    person_count = 0
    num_classification_queries = 0
    num_count_queries = 0
    num_detection_queries = 0
    for q in workload:
        if q[2] == 'car':
            num_car_queries += 1
        elif q[2] == 'person':
            num_person_queries += 1
    for q in workload:
        if q[1] == 'binary-classification':
            num_classification_queries += 1
        elif q[1] == 'count':
            num_count_queries += 1
        elif q[1] == 'detection':
            num_detection_queries += 1
    
    car_count = frame_to_model_to_orientation_to_car_count[current_frame][q[0]][current_orientation]
    person_count = frame_to_model_to_orientation_to_person_count[current_frame][q[0]][current_orientation]
    alpha = num_classification_queries / len(workload)
    m1 = 1 if car_count > 0 else 0
    w1 = num_car_queries / len(workload)
    m2 = 1 if person_count > 0 else 0
    w2 = num_person_queries / len(workload)
    beta = (num_count_queries + num_detection_queries) / len(workload)
    return (alpha * m1 * w1) + (alpha * m2 * w2) + (beta * w1 * car_count) + (beta * w2 * person_count)


# this returns the weighted sum of car and person count. weight based on queries in workload. 
# if workload contains no people, this returns count of cars
# if workload contains no cars, this returns count of people 
# if workload contains half people and half cars, this return weight sum of both with weight = 0.5
def get_count_of_orientation(workload, 
                     current_frame,
                     current_orientation,
                     orientations,
                     frame_to_model_to_orientation_to_car_count,
                     frame_to_model_to_orientation_to_person_count,
                     frame_to_model_to_orientation_to_car_map,
                     frame_to_model_to_orientation_to_person_map):
    num_car_queries = 0
    num_person_queries = 0
    car_count = 0
    person_count = 0
    for q in workload:
        if q[2] == 'car':
            num_car_queries += 1
        elif q[2] == 'person':
            num_person_queries += 1
    
    car_count = frame_to_model_to_orientation_to_car_count[current_frame][q[0]][current_orientation]
    person_count = frame_to_model_to_orientation_to_person_count[current_frame][q[0]][current_orientation]

    car_weight = (num_car_queries / (num_person_queries + num_car_queries))
    person_weight = (num_person_queries / (num_person_queries + num_car_queries))
    return ( (car_weight * car_count) + (person_weight * person_count) )
    

# A workload is a list of tuples [(model, query, object)]
#   - where a model is one of the entries in MODELS
#   - query is binary-classification, count, detection, or aggregate-count
#   - object is car or person
def compute_accuracy(workload, 
                     current_frame,
                     current_orientation,
                     orientations,
                     frame_to_model_to_orientation_to_car_count,
                     frame_to_model_to_orientation_to_person_count,
                     frame_to_model_to_orientation_to_car_map,
                     frame_to_model_to_orientation_to_person_map,
                    ):
    total_accuracy = 0.0

    for q in workload:

        current_score = 0.0
        if q[2] == 'car':
            if q[1] == 'binary-classification':
                car_count = frame_to_model_to_orientation_to_car_count[current_frame][q[0]][current_orientation]
                if car_count > 0.0:
                    current_score = 1.0
            elif q[1] == 'count':
                car_count = frame_to_model_to_orientation_to_car_count[current_frame][q[0]][current_orientation]
                max_count, _ = find_best_score_and_orientation(current_frame, q[0], orientations, frame_to_model_to_orientation_to_car_count)
                if max_count > 0.0:
                    current_score = car_count / max_count 
            elif q[1] == 'detection':
                map_score = frame_to_model_to_orientation_to_car_map[current_frame][q[0]][current_orientation]
                max_map, _ = find_best_score_and_orientation(current_frame, q[0], orientations, frame_to_model_to_orientation_to_car_map)
                if max_map > 0.0:
                    current_score += map_score  / max_map
                    
                

        elif q[2] == 'person':
            if q[1] == 'binary-classification':
                person_count = frame_to_model_to_orientation_to_person_count[current_frame][q[0]][current_orientation]
                if person_count > 0.0:
                    current_score = 1.0
            elif q[1] == 'count':
                person_count = frame_to_model_to_orientation_to_person_count[current_frame][q[0]][current_orientation]
                max_count, _ = find_best_score_and_orientation(current_frame, q[0], orientations, frame_to_model_to_orientation_to_person_count)

                if max_count > 0.0:
                    current_score = person_count / max_count 
            elif q[1] == 'detection':
                map_score = frame_to_model_to_orientation_to_person_map[current_frame][q[0]][current_orientation]
                max_map, _ = find_best_score_and_orientation(current_frame, q[0], orientations, frame_to_model_to_orientation_to_person_map)
                if max_map > 0.0:
                    current_score+= map_score  / max_map
        total_accuracy += current_score
    num_aggregate_queries = num_aggregate_queries_in_workload(workload)
    if len(workload) == num_aggregate_queries:
        return 0
    total_accuracy /= (len(workload) - num_aggregate_queries)
    return total_accuracy


def compute_aggregate_accuracy(workload,  
                                current_frame, 
                                frame_limit,
                                current_orientation, 
                                orientations, 
                                model_to_object_ids_found, 
                                frame_to_model_to_orientation_to_object_ids,
):

    # Estimate perf of aggregate queries
    num_aggregate_queries = num_aggregate_queries_in_workload(workload)
    if num_aggregate_queries == 0 or current_frame not in frame_to_model_to_orientation_to_object_ids:
        return 0.0


    model_to_object_id_to_weights = {}
    gt_model_to_object_ids_for_frame = {}

    for f in range(current_frame, frame_limit):
        # Lookahead to see how many frames each id is in
        for q in workload:
            if q[0] in model_to_object_id_to_weights:
                continue
            object_ids_seen_this_frame = []
            if q[1] != 'aggregate-count':
                continue
            if q[0] not in model_to_object_id_to_weights:
                model_to_object_id_to_weights[q[0]] = {}
            if q[0] not in model_to_object_id_to_weights:
                model_to_object_id_to_weights[q[0]] = {}
            if q[0] not in frame_to_model_to_orientation_to_object_ids[current_frame]:
                continue
            if current_orientation not in frame_to_model_to_orientation_to_object_ids[current_frame][q[0]]:
                continue 
            for o in orientations:
                new_object_ids = frame_to_model_to_orientation_to_object_ids[current_frame][q[0]][current_orientation]
                for obj_id in new_object_ids:
                    if obj_id in object_ids_seen_this_frame:
                        continue
                    if obj_id not in model_to_object_id_to_weights[q[0]]:
                        model_to_object_id_to_weights[q[0]][obj_id] = 0
                    object_ids_seen_this_frame.append(obj_id)
                    model_to_object_id_to_weights[q[0]][obj_id] += 0.2
    # Make weights <= 1
    for m in model_to_object_id_to_weights:
       for obj_id in model_to_object_id_to_weights[m]: 
            model_to_object_id_to_weights[m][obj_id] = max(1, model_to_object_id_to_weights[m][obj_id])


    for q in workload:
        # Get ground truth ids
        if q[1] != 'aggregate-count':
            continue
        highest_score= 0.0
        for o in orientations:
            if q[0] not in model_to_object_ids_found:
                model_to_object_ids_found[q[0]] = []
            if q[0] not in frame_to_model_to_orientation_to_object_ids[current_frame]:
                continue
            if current_orientation not in frame_to_model_to_orientation_to_object_ids[current_frame][q[0]]:
                continue 
            if q[0] not in model_to_object_id_to_weights:
                continue

            new_object_ids = frame_to_model_to_orientation_to_object_ids[current_frame][q[0]][current_orientation]
            # Add new object ids without duplicating
            new_ids = [
                x for x in new_object_ids
                if x not in model_to_object_ids_found[q[0]]
            ]
            score = 0.0
            for obj_id in new_ids:
                score +=  model_to_object_id_to_weights[q[0]][obj_id]
            if score > highest_score:
                highest_score = score
        gt_model_to_object_ids_for_frame[q[0]] = highest_score

    aggregate_accuracy = 0.0
    for q in workload:
        # Compute score for current frame
        if q[1] != 'aggregate-count':
            continue
        if q[0] not in model_to_object_ids_found:
            model_to_object_ids_found[q[0]] = []
        if q[0] not in frame_to_model_to_orientation_to_object_ids[current_frame]:
            continue
        if current_orientation not in frame_to_model_to_orientation_to_object_ids[current_frame][q[0]]:
            continue 
        new_object_ids = frame_to_model_to_orientation_to_object_ids[current_frame][q[0]][current_orientation]
        # Add new object ids without duplicating
        new_ids = [
            x for x in new_object_ids
            if x not in model_to_object_ids_found[q[0]]
        ]
        score = 0.0
        for obj_id in new_ids:
            score += model_to_object_id_to_weights[q[0]][obj_id]
        if gt_model_to_object_ids_for_frame[q[0]] == 0:
            continue

        current_score = score / gt_model_to_object_ids_for_frame[q[0]]

        aggregate_accuracy += current_score

    return aggregate_accuracy / num_aggregate_queries


def evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                               frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids
                              ):
    num_aggregate_queries = num_aggregate_queries_in_workload(workload)
    if num_aggregate_queries == 0:
        return 0.0
    total_aggregate_accuracy = 0.0
    for q in workload:
        if q[1] != 'aggregate-count' :
            continue
        if q[0] not in model_to_object_ids_found or q[0] not in gt_model_to_object_ids:
            continue
        if len(gt_model_to_object_ids[q[0]]) > 0:
            total_aggregate_accuracy += len(model_to_object_ids_found[q[0]]) / len(gt_model_to_object_ids[q[0]])
    return total_aggregate_accuracy / num_aggregate_queries

def find_aggregate_ids_for_frame_and_orientation(workload, current_frame, current_orientation,
                            frame_to_model_to_orientation_to_object_ids,
                            model_to_object_ids_found
                            ):
    if current_frame not in frame_to_model_to_orientation_to_object_ids:
        return
    for q in workload:
        if q[1] == 'aggregate-count':
            if q[0] not in frame_to_model_to_orientation_to_object_ids[current_frame]:
                continue
            if current_orientation not in frame_to_model_to_orientation_to_object_ids[current_frame][q[0]]:
                continue 
            if q[0] not in model_to_object_ids_found:
                model_to_object_ids_found[q[0]] = []
            new_object_ids = frame_to_model_to_orientation_to_object_ids[current_frame][q[0]][current_orientation]
            # Add new object ids without duplicating
            model_to_object_ids_found[q[0]].extend(
                x for x in new_object_ids
                if x not in model_to_object_ids_found[q[0]]
            )





def num_aggregate_queries_in_workload(workload):
    num_queries = 0
    for q in workload:
        if q[1] == 'aggregate-count':
            num_queries += 1
    return num_queries


def num_car_queries_in_workload(workload):
    num_queries = 0
    for q in workload:
        if q[2] == 'car':
            num_queries += 1
    return num_queries


def evaluate_workload_with_orientation(workload,
                    frame_begin,
                    frame_limit,
                    current_orientation,
                    orientations,
                    frame_to_model_to_orientation_to_car_count,
                    frame_to_model_to_orientation_to_person_count,
                    frame_to_model_to_orientation_to_car_map,
                    frame_to_model_to_orientation_to_person_map, 
                    frame_to_model_to_orientation_to_object_ids,
                    gt_model_to_object_ids,
                    blacklisted_frames=[]):

    model_to_object_ids_found = {}
    non_aggregate_accuracy = 0.0
    aggregate_accuracy = 0.0
    num_frames = 0
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        non_aggregate_accuracy += compute_accuracy(workload, f, current_orientation, orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        find_aggregate_ids_for_frame_and_orientation(workload, f, current_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        num_frames += 1
    if num_frames == 0:
        return None
    non_aggregate_accuracy /= num_frames

    num_aggregate_queries = num_aggregate_queries_in_workload(workload)
    aggregate_accuracy = evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * non_aggregate_accuracy
    return total_accuracy


# Find best orientation/score across a video
def find_best_fixed(workload,
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
    best_orientation = orientations[0]
    max_score = -1.0
    for o in orientations:
        total_accuracy = evaluate_workload_with_orientation(workload,
                            frame_begin,
                            frame_limit,
                            o,
                            orientations,
                            frame_to_model_to_orientation_to_car_count,
                            frame_to_model_to_orientation_to_person_count,
                            frame_to_model_to_orientation_to_car_map,
                            frame_to_model_to_orientation_to_person_map, 
                            frame_to_model_to_orientation_to_object_ids,
                            gt_model_to_object_ids,
                            blacklisted_frames=blacklisted_frames)
        if total_accuracy > max_score:
            max_score = total_accuracy
            best_orientation = o
    return max_score, best_orientation

