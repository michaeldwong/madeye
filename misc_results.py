import evaluation_tools


SKIP = 6

def generate_all_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    for r1 in range(0,360,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            orientations.append(f'{r1}-{r2}-1')
    return orientations


def generate_neighboring_orientations(current_orientation):
    items = current_orientation.split('-')
    pan = int(items[0])
    zoom = int(items[-1])
    if pan == 0:
        left_horz = 330
    else:
        left_horz = int(items[0]) - 30
    if pan == 330:
        right_horz = 0
    else:
        right_horz = int(items[0]) + 30

    if len(items) == 4:
        tilt = int(items[2]) * -1
    else:
        tilt = int(items[1])
    top_tilt = tilt + 15
    bottom_tilt = tilt - 15

    if tilt == 30:
        return [ f'{left_horz}-{tilt}-{zoom}',
                 f'{right_horz}-{tilt}-{zoom}',
                 f'{pan}-{bottom_tilt}-{zoom}' ]
    elif tilt == -30:
        return [ f'{left_horz}-{tilt}-{zoom}',
                 f'{right_horz}-{tilt}-{zoom}',
                 f'{pan}-{top_tilt}-{zoom}']
    return [ f'{left_horz}-{tilt}-{zoom}',
             f'{right_horz}-{tilt}-{zoom}',
             f'{pan}-{top_tilt}-{zoom}',
             f'{pan}-{bottom_tilt}-{zoom}' ]


def generate_n2_neighboring_orientations(current_orientation):
    items = current_orientation.split('-')
    pan = int(items[0])
    zoom = int(items[-1])
    if pan == 0:
        left_horz = 330
        left_left_horz = 300
    elif pan == 30:
        left_horz = 330
        left_left_horz = 300
    else:
        left_horz = int(items[0]) - 30
        left_left_horz = int(items[0]) - 60


    if pan == 330:
        right_horz = 0
        right_right_horz = 30
    elif pan == 300:
        right_horz = 330
        right_right_horz = 0
    else:
        right_horz = int(items[0]) + 30
        right_right_horz = int(items[0]) + 60

    if len(items) == 4:
        tilt = int(items[2]) * -1
    else:
        tilt = int(items[1])

    top_top_tilt = tilt + 30
    top_tilt = tilt + 15
    bottom_tilt = tilt - 15
    bottom_bottom_tilt = tilt - 30

    if tilt == 30 or tilt == 15:
        return [ 
#                 f'{left_horz}-{tilt}-{zoom}',
#                 f'{right_horz}-{tilt}-{zoom}',
#                 f'{pan}-{bottom_tilt}-{zoom}',
                 f'{left_left_horz}-{tilt}-{zoom}',
                 f'{right_right_horz}-{tilt}-{zoom}',
                 f'{pan}-{bottom_bottom_tilt}-{zoom}'

                 ]
    elif tilt == -30 or tilt == -15:
        return [ 
#                 f'{left_horz}-{tilt}-{zoom}',
#                 f'{right_horz}-{tilt}-{zoom}',
#                 f'{pan}-{top_tilt}-{zoom}',
                 f'{left_left_horz}-{tilt}-{zoom}',
                 f'{right_right_horz}-{tilt}-{zoom}',
                 f'{pan}-{top_top_tilt}-{zoom}'
                ]
    return [ 


             f'{left_left_horz}-{tilt}-{zoom}',
#             f'{left_horz}-{tilt}-{zoom}',
             f'{right_right_horz}-{tilt}-{zoom}',
#             f'{right_horz}-{tilt}-{zoom}',
#             f'{pan}-{top_tilt}-{zoom}',
             f'{pan}-{top_top_tilt}-{zoom}',
#             f'{pan}-{bottom_tilt}-{zoom}' ,
             f'{pan}-{bottom_bottom_tilt}-{zoom}' 
            ]

def delta_of_delta_graph(inference_dir,
                         frame_begin,
                         frame_limit,
                         frame_to_model_to_orientation_to_count, 
                         ):
    orientations = generate_all_orientations()
    all_deltas = []
    for f in range(frame_begin+SKIP, frame_limit+1):
        if f % SKIP != 0:
            continue
        for model in frame_to_model_to_orientation_to_count[f]:
            for o1 in orientations:
                if frame_to_model_to_orientation_to_count[f][model][o1]  == 0:
                    continue
                for o2 in generate_neighboring_orientations(o1):
                    delta1 = frame_to_model_to_orientation_to_count[f][model][o1] - frame_to_model_to_orientation_to_count[f-SKIP][model][o1]
                    delta2 = frame_to_model_to_orientation_to_count[f][model][o2] - frame_to_model_to_orientation_to_count[f-SKIP][model][o2]
                    all_deltas.append(abs(delta1-delta2))
    return all_deltas



def distance_between_top_k_orientations_each_frame(workload,
                                                   frame_begin,
                                                   frame_limit,
                                                   orientations,
                                                   frame_to_model_to_orientation_to_car_count,
                                                   frame_to_model_to_orientation_to_person_count,
                                                   frame_to_model_to_orientation_to_car_map,
                                                   frame_to_model_to_orientation_to_person_map,
                                                   frame_to_model_to_orientation_to_object_ids):

    pan_dists = []
    tilt_dists = []
    intersection_lengths = []
    k = 7

    gt_model_to_object_ids = evaluation_tools.best_dynamic_aggregate_ids(workload, frame_begin, frame_limit, orientations, frame_to_model_to_orientation_to_object_ids)

    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    num_non_aggregate_queries = len(workload) - num_aggregate_queries
    model_to_object_ids_found = {}
    prev_best_orientations = []
    for f in range(frame_begin, frame_limit+1):
        if f % SKIP != 0:
            continue
        best_orientations = []
        remaining_orientations = orientations.copy()
        for i in range(0,k):
            max_score = -100.0
            current_top_orientation = orientations[0]
            for o in remaining_orientations: 
                non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                    frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
                aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)

                est_current_total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * non_aggregate_accuracy
                if est_current_total_accuracy > max_score:
                    max_score = est_current_total_accuracy
                    top_orientation = o
            best_orientations.append(top_orientation)
            remaining_orientations.remove(top_orientation)

        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, best_orientations[0], frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        pan_dists = []
        tilt_dists = []
        print('best orientations ', best_orientations) 
        for o1 in best_orientations:
            for o2 in best_orientations:
                if o1 == o2:
                    continue
                pan = evaluation_tools.find_pan_dist(evaluation_tools.extract_pan(o1), evaluation_tools.extract_pan(o2))
                tilt = evaluation_tools.find_tilt_dist(evaluation_tools.extract_tilt(o1), evaluation_tools.extract_tilt(o2))

                print('distance between ', o1 , ' and ', o2, ' -> ', pan, ' pan and ', tilt, ' tilt')
                pan_dists.append(pan)
                tilt_dists.append(tilt)

        if len(prev_best_orientations) > 0:
            intersection_length = 0
            for o1 in prev_best_orientations:
                for o2 in best_orientations:
                    if o1 == o2:
                        intersection_length += 1
            intersection_lengths.append(intersection_length)
        prev_best_orientations = best_orientations
    return pan_dists, tilt_dists, intersection_lengths

