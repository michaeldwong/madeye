
def find_tilt_dist(current_tilt, target_tilt):
    return abs(current_tilt - target_tilt)

def find_pan_dist(current_pan, target_pan):
    if current_pan > target_pan:
        if current_pan - target_pan <= 180:
            # Rotating left
            return current_pan - target_pan
        # Rotating right
        return (360 - current_pan) + target_pan
    else:
        if target_pan - current_pan <= 180:
            # Rotating right
            return target_pan - current_pan
        # Rotating left
        return (360 - target_pan) + current_pan

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

def extract_pan(orientation):
    return int(parse_orientation_string(orientation)[0])

def extract_tilt(orientation):
    return int(parse_orientation_string(orientation)[1])

def extract_zoom(orientation):
    return int(orientation[-1])
