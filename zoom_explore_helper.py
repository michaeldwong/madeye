import statistics 
import copy 
import json
def compute_better_zoom_factor(orientation, orientation_to_current_car_boxes, orientation_to_current_person_boxes):
    all_boxes = []
    if orientation in orientation_to_current_car_boxes:
        all_boxes.extend(orientation_to_current_car_boxes[orientation])
    if orientation in orientation_to_current_person_boxes:
        all_boxes.extend(orientation_to_current_person_boxes[orientation])
    if len(all_boxes) < 1:
        return orientation[-1:]
    all_boxes = [tuple(list(x)) for x in all_boxes]
    better_zoom_factor = 1
    not_much_spills_outside_zoom_3 = False
    not_much_spills_outside_zoom_2 = False
    excess_spaces = []
    for box in all_boxes:
        excess_space = 0 # (320, 180) to (960, 540) is the inner zoom. anything outside that is considered excess
        if box[0] < 320:
            excess_space += 320 - box[0]
        if box[2] > 960:
            excess_space += box[2] - 960
        if box[1] < 180:
            excess_space += 180 - box[1]
        if box[3] > 540:
            excess_space += box[3] - 540
        excess_spaces.append(excess_space)
    if statistics.mean(excess_spaces) < 15:
        not_much_spills_outside_zoom_3 = True
    else: # check zoom 2
        excess_spaces = []
        for box in all_boxes:
            excess_space = 0 # (320, 180) to (960, 540) is the inner zoom. anything outside that is considered excess
            if box[0] < 160:
                excess_space += 160 - box[0]
            if box[2] > 1120:
                excess_space += box[2] - 1120
            if box[1] < 90:
                excess_space += 90 - box[1]
            if box[3] > 630:
                excess_space += box[3] - 630
            excess_spaces.append(excess_space)
        if statistics.mean(excess_spaces) < 10:
            not_much_spills_outside_zoom_2 = True

    distances = []
    for i in range(len(all_boxes)):
        for j in range(len(all_boxes)):
            if i != j:
                box1 = all_boxes[i]
                box2 = all_boxes[j]
                x_left_1 = box1[0]
                y_left_1 = box1[1]
                x_right_1 = box1[2]
                y_right_1 = box1[3]
                x_left_2 = box2[0]
                y_left_2 = box2[1]
                x_right_2 = box2[2]
                y_right_2 = box2[3]
                center_x_1 = (x_left_1+x_right_1)/2.0
                center_y_1 = (y_left_1+y_right_1)/2.0
                center_x_2 = (x_left_2+x_right_2)/2.0
                center_y_2 = (y_left_2+y_right_2)/2.0
                dist = ((center_y_2-center_y_1)**2 + (center_x_2-center_x_1)**2)**0.5
                distances.append(dist)
    if len(distances) <= 1:
        average_distance_between_boxes = 100000
    else:
        average_distance_between_boxes = statistics.mean(distances)

    if not_much_spills_outside_zoom_3 and statistics.mean(excess_spaces) > 300:
        better_zoom_factor = 3
    else:
        better_zoom_factor = 1

    return str(better_zoom_factor)
    

def add_zoom_factors(current_formation, orientation_to_current_car_boxes, orientation_to_current_person_boxes, zoom_explorations_in_progress):
    # for certain orientations we have been exploring zoom. lets add those back
    # print(f"adding zoom factors")
    # print(json.dumps(zoom_explorations_in_progress, indent=2))
    # print(f"that was zoom explorations in progress")
    current_formation_as_set = set(current_formation)
    for orientation in zoom_explorations_in_progress:
        # print(f"prior zoom restored for {orientation}")
        num_tries_remaining, target_zoom = zoom_explorations_in_progress[orientation]
        if num_tries_remaining < 0:
            continue
            zoom_explorations_in_progress[orientation] = (zoom_explorations_in_progress[orientation][0]-1, target_zoom)
            if orientation in current_formation_as_set:
                current_formation_as_set.remove(orientation)
                current_formation_as_set.add(set_zoom_factor(orientation, target_zoom))

    current_formation_as_set_backup = copy.deepcopy(current_formation_as_set)
    for orientation in current_formation_as_set_backup:
        if orientation.endswith('1'): # currently zoomed out
            if orientation in zoom_explorations_in_progress and zoom_explorations_in_progress[orientation][0] > 1 and orientation.endswith('1'):
                continue
            better_zoom_factor = compute_better_zoom_factor(orientation, orientation_to_current_car_boxes, orientation_to_current_person_boxes)
            if better_zoom_factor != '1' and not orientation.endswith('1'):
                current_formation_as_set.discard(orientation)
                zoom_explorations_in_progress[orientation] = (3, better_zoom_factor)
                orientation = set_zoom_factor(orientation, better_zoom_factor)
                current_formation_as_set.add(orientation)
    # print(json.dumps(zoom_explorations_in_progress, indent=2))
    # input(f"that was updated zoom explorations")
    return list(current_formation_as_set), zoom_explorations_in_progress

def set_zoom_factor(orientation, target_zoom):
    return orientation[:-1] + target_zoom

def reset_zoom_factor(orientation, anchor_orientation):
    base_zoom = anchor_orientation[-1:]
    return orientation[:-1] + base_zoom

def reset_zoom_factors(current_formation, anchor_orientation):
    base_zoom = anchor_orientation[-1:]
    temp_output = []
    for orientation in current_formation:
        temp_output.append(orientation[:-1] + base_zoom)
    return temp_output

if __name__ == "__main__":
    print(reset_zoom_factors(["270--15-3"], "180-2-2"))


