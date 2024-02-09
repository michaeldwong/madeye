# helper file
# used to ingest mot files and generate dictionaries as needed

import csv
SKIP = 6
def get_frame_and_id_from_mot_file( mot_file_path):
    print(f"loading file {mot_file_path}")
    frame_and_id_info = {}
    frame_to_object_id_to_detected = {}
    with open(mot_file_path, "r") as f:
        s = csv.reader(f)
        for row in s:
            # row is an array. 0th element is frame num. 
            frame_num = int(row[0]) 
            while frame_num % SKIP != 0:
                frame_num += 1
            object_id = int(row[1])
            left = int(row[2])
            top = int(row[3])
            right = int(row[4])
            bottom = int(row[5])
            if frame_num not in frame_and_id_info:
                frame_and_id_info[frame_num] = []
            if frame_num not in frame_to_object_id_to_detected:
                frame_to_object_id_to_detected[frame_num] = {}
            frame_and_id_info[frame_num].append(object_id)
            frame_to_object_id_to_detected[frame_num][object_id] = [left, top, right, bottom]
    return frame_and_id_info, frame_to_object_id_to_detected


# frame_to_model_to_orientaiton_detected_object_ids
def get_mot_info_from_directory(input_dir):
    import os
    import glob
    from pathlib import Path;

    frame_to_model_to_orientation_to_object_ids = {}
    frame_to_model_to_orientation_to_object_id_to_mot_detected = {}
    models = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    for model in models:
        model_folder = os.path.join(input_dir, model)
        for frame_dir in os.listdir(str(model_folder)):

            csv_files = glob.glob(os.path.join(model_folder, frame_dir, '*.txt'), recursive=True)
            
            for mot_file in csv_files:
                orientation = Path(mot_file).stem
                frame_and_ids, frame_to_object_id_to_detected = get_frame_and_id_from_mot_file( mot_file)
                for f, objs in frame_and_ids.items():
                    if f not in frame_to_model_to_orientation_to_object_ids:
                        frame_to_model_to_orientation_to_object_ids[f] = {}
                        frame_to_model_to_orientation_to_object_id_to_mot_detected[f] = {}
                    if model not in frame_to_model_to_orientation_to_object_ids[f]:
                        frame_to_model_to_orientation_to_object_ids[f][model] = {}
                        frame_to_model_to_orientation_to_object_id_to_mot_detected[f][model] = {}
                    if orientation not in frame_to_model_to_orientation_to_object_ids[f][model]:
                        frame_to_model_to_orientation_to_object_ids[f][model][orientation] = []
                        frame_to_model_to_orientation_to_object_id_to_mot_detected[f][model][orientation] = {}
                    for obj_id in objs:
                        frame_to_model_to_orientation_to_object_id_to_mot_detected[f][model][orientation][obj_id] = frame_to_object_id_to_detected[f][obj_id]
                    frame_to_model_to_orientation_to_object_ids[f][model][orientation].extend(objs)
    return frame_to_model_to_orientation_to_object_ids, frame_to_model_to_orientation_to_object_id_to_mot_detected

# object_id_to_frame_to_model_to_orientations
def get_object_presence_info_from_mot_files(frame_to_model_to_orientation_to_object_ids):
    object_id_to_frame_to_model_to_orientations = {}
    # frame, {'frcnn': {'1-1-1':[obj1, obj2] } }
    for f, v1 in frame_to_model_to_orientation_to_object_ids.items():
        # frcnn, {'1-1-1': [obj1, obj2]}
        for m, v2 in v1.items():
            for o, objs in v2.items():
                for obj in objs:
                    if obj not in object_id_to_frame_to_model_to_orientations:
                        object_id_to_frame_to_model_to_orientations[obj] = {}
                    if f not in object_id_to_frame_to_model_to_orientations[obj]:
                        object_id_to_frame_to_model_to_orientations[obj][f] = {}
                    if m not in object_id_to_frame_to_model_to_orientations[obj][f]:
                        object_id_to_frame_to_model_to_orientations[obj][f][m] = []
                    object_id_to_frame_to_model_to_orientations[obj][f][m].append(o)
    return object_id_to_frame_to_model_to_orientations

if __name__ == "__main__":
    # this directory must be the equivalent of `/scratch/mdwong/mot-results/seattle-traffic-mot-results`
    mot_files = get_mot_info_from_directory("/scratch/mdwong/mot-results/seattle-traffic-mot-results")
    print(mot_files)

    # object_occurence_info = get_object_presence_info_from_mot_files("/scratch/mdwong/mot-results/seattle-traffic-mot-results")
    # print(object_occurence_info)
