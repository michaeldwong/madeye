import random

random.seed(100)

class DetectedObject:
    def __init__(self):
        self.left = -1
        self.top = -1
        self.right = -1
        self.bottom = -1
        self.type = "None"
        self.confidence = -1
        self.obj_id = -1
        self.orientation = ""
        self.frame_num = -1
        self.model = ""
        self.zoomed_out = False
    
    # new_obj_with_id
    def __init__(self, l, t, r, b, ty, c, i, o, f, m):
        self.left = l
        self.top = t
        self.right = r
        self.bottom = b
        self.type = ty
        self.confidence = c
        self.obj_id = i
        self.orientation = o
        self.frame_num = f
        self.model = m

    def __repr__(self):
        representation = f"<Object Type: {self.type},Confidence: {self.confidence},ID:{self.obj_id},Bounds (l,t,r,b):[{self.left},{self.top},{self.right},{self.bottom}]>"
        return representation

    def __eq__(self, other):
        return self.obj_id == other.obj_id
            
    def area(self):
        length = self.right - self.left
        height = self.bottom - self.top
        return length * height

    def move_far_away(self):
        self.left += 1280
        self.right += 1280
        self.bottom += 720
        self.top += 720
    
    def overlay_on_another_object(self, another_detected_object):
        my_length = self.right - self.left
        my_height = self.bottom - self.top
        self.left = another_detected_object.left
        self.top = another_detected_object.top
        self.right = self.left + my_length
        self.bottom = self.top + my_height

    def recenter_on_another_object(self, another_detected_object):
        target_center_x = ((another_detected_object.right - another_detected_object.left) / 2) + another_detected_object.left
        target_center_y = ((another_detected_object.bottom - another_detected_object.top) / 2) + another_detected_object.top
        my_length = self.right - self.left
        my_height = self.bottom - self.top
        self.left = target_center_x - (my_length / 2)
        self.right = target_center_x + (my_length / 2)
        self.bottom = target_center_y + (my_height / 2)
        self.top = target_center_y - (my_height / 2)

    def is_zoomed_in(self):
        return self.orientation.endswith("-2") or self.orientation.endswith("-3")

    def zoom_out_bounding_box(self):
        WIDTH = 720
        HEIGHT = 1280
        converted_self_left = (2 * self.left) + (int(self.orientation[-1]) - 1) * WIDTH
        converted_self_top = (2 * self.top) + (int(self.orientation[-1]) - 1) * HEIGHT
        converted_self_right = (2 * self.right) + (int(self.orientation[-1]) - 1) * WIDTH
        converted_self_bottom = (2 * self.bottom) + (int(self.orientation[-1]) - 1) * HEIGHT
        self.left = int(converted_self_left / (2 * int(self.orientation[-1])))
        self.right = int(converted_self_right / (2 * int(self.orientation[-1])))
        self.top = int(converted_self_top / (2 * int(self.orientation[-1])))
        self.bottom = int(converted_self_bottom / (2 * int(self.orientation[-1])))
        self.orientation = self.orientation[:-1] + "1"
        self.zoomed_out = True
        
