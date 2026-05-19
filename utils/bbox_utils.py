def get_center_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    center_x = int((xmax + xmin) / 2)
    center_y = int((ymax + ymin) / 2)
    return center_x, center_y

def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_court_center(court_keypoints):
    court_x_coords = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
    court_y_coords = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]
    return sum(court_x_coords) / len(court_x_coords), sum(court_y_coords) / len(court_y_coords)

def get_foot_position(bbox):
    xmin, ymin, xmax, ymax = bbox
    return int(xmin+xmax)/2, ymax

def get_closet_kps_index(point, keypoints, keypoint_indices):
    closest_keypoint = float('inf')
    kps_ind = keypoint_indices[0]
    for index in keypoint_indices:
        kps = keypoints[index*2], keypoints[index*2+1]
        dist = abs(point[1] - kps[1])

        if dist < closest_keypoint:
            closest_keypoint = dist
            kps_ind = index

    return kps_ind

def get_height_of_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    return ymax - ymin

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
