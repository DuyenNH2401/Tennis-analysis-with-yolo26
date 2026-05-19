import cv2
from constants import Config
config = Config()
from utils import *

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

        self.drawing_keypoints = self.set_court_drawing_keypoints()
        self.lines = config.lines

        # For Heatmap
        import numpy as np
        self.heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0] * (14*2)
        #Point 0
        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        #Point 1
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        #Point 2
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = self.court_start_y + self.convert_meters_pixels(config.HALF_COURT_LINE_HEIGHT * 2)
        #Point 3
        drawing_keypoints[6] = drawing_keypoints[2]
        drawing_keypoints[7] = drawing_keypoints[5]
        #Point 4
        drawing_keypoints[8] = drawing_keypoints[0] + self.convert_meters_pixels(config.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1]
        #point 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_pixels(config.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5]
        #Point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_pixels(config.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[1]
        #Point 7
        drawing_keypoints[14] = drawing_keypoints[12]
        drawing_keypoints[15] = drawing_keypoints[5]
        #Point 8
        drawing_keypoints[16] = drawing_keypoints[8]
        drawing_keypoints[17] = drawing_keypoints[1] + self.convert_meters_pixels(config.NO_MANS_LAND_HEIGHT)
        #Point 9
        drawing_keypoints[18] = drawing_keypoints[12]
        drawing_keypoints[19] = drawing_keypoints[17]
        #Point 10
        drawing_keypoints[20] = drawing_keypoints[8]
        drawing_keypoints[21] = drawing_keypoints[5] - self.convert_meters_pixels(config.NO_MANS_LAND_HEIGHT)
        #Point 11
        drawing_keypoints[22] = drawing_keypoints[12]
        drawing_keypoints[23] = drawing_keypoints[21]
        #Point 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18]) /2)
        drawing_keypoints[25] = drawing_keypoints[17]
        #Point 13
        drawing_keypoints[26] = drawing_keypoints[24]
        drawing_keypoints[27] = drawing_keypoints[21]

        return drawing_keypoints

    def convert_meters_pixels(self, meters):
        return convert_meters_to_pixels(meters,
                                        config.DOUBLE_LINE_WIDTH,
                                        self.court_drawing_width)

    def get_meters_from_mini_court_pixels(self, pixels):
        return (pixels * config.DOUBLE_LINE_WIDTH) / self.court_drawing_width

    def draw_background_rectangle(self, frame):
        overplay = frame.copy()
        cv2.rectangle(overplay, (self.start_x, self.start_y),(self.end_x, self.end_y),(255,255,255),cv2.FILLED)
        alpha = 0.5
        out = cv2.addWeighted(overplay, alpha, frame, 1 - alpha, 0)
        return out

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        net_start_points = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] +  self.drawing_keypoints[5])/2))
        net_end_points = (self.drawing_keypoints[2], int((self.drawing_keypoints[3] +  self.drawing_keypoints[7])/2))
        cv2.line(frame, net_start_points, net_end_points, (0, 0, 255), 3)

        return frame


    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return self.court_start_x, self.court_start_y

    def get_mini_court_coordinates(self, object_position,
                                   closest_keypoint,
                                   closest_keypoint_index,
                                   player_height_in_pixels,
                                   player_height_in_meters):

        distance_from_kps_x_pixels, distance_from_kps_y_pixels = measure_xy_distance(object_position, closest_keypoint)

        distance_from_kps_x_meters = covert_pixel_to_meters(distance_from_kps_x_pixels,
                                                            player_height_in_meters, player_height_in_pixels)
        distance_from_kps_y_meters = covert_pixel_to_meters(distance_from_kps_y_pixels,
                                                            player_height_in_meters, player_height_in_pixels)

        mini_court_x_distance_pixels = self.convert_meters_pixels(distance_from_kps_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_pixels(distance_from_kps_y_meters)

        closest_mini_court_keypoint =(self.drawing_keypoints[closest_keypoint_index*2],
                                      self.drawing_keypoints[closest_keypoint_index*2+1])

        mini_court_player_position = (closest_mini_court_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_court_keypoint[1]+mini_court_y_distance_pixels)

        return mini_court_player_position


    def convert_bboxes_to_mini_court_coordinates(self, player_bboxes, ball_bboxes, original_court_kps):
        player_height = {1:config.PLAYER_1_HEIGHT_METERS, 2:config.PLAYER_2_HEIGHT_METERS}
        output_player_bboxes = []
        output_ball_bboxes = []

        for frame_num, player_bbox in enumerate(player_bboxes):
            ball_box = ball_bboxes[frame_num].get(1, [0, 0, 0, 0])
            ball_pos = get_center_bbox(ball_box)
            if not player_bbox:
                output_player_bboxes.append({})
                output_ball_bboxes.append({})
                continue
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_point_distance(ball_pos,
                                                                                               get_center_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            ball_mini_court_position = None
            for player_id, bbox in player_bbox.items():
                foot_pos = get_foot_position(bbox)

                closest_keypoint_index = get_closet_kps_index(foot_pos, original_court_kps, [0, 2, 12, 13])
                closest_keypoint = (original_court_kps[closest_keypoint_index*2],
                                    original_court_kps[closest_keypoint_index*2+1])

                frame_inx_min = max(0, frame_num-20)
                frame_inx_max = min(len(player_bboxes), frame_num+50)

                bboxes_height_in_pixel = [
                    get_height_of_bbox(player_bboxes[i][player_id])
                    for i in range(frame_inx_min, frame_inx_max)
                    if player_id in player_bboxes[i]
                ]

                if not bboxes_height_in_pixel:
                    continue

                max_player_height_in_pixel = max(bboxes_height_in_pixel)

                mini_court_player_position = self.get_mini_court_coordinates(foot_pos,
                                                 closest_keypoint,
                                                 closest_keypoint_index,
                                                 max_player_height_in_pixel,
                                                 player_height.get(player_id,
                                                                   config.PLAYER_1_HEIGHT_METERS))

                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closet_kps_index(ball_pos, original_court_kps, [0, 2, 12, 13])
                    closest_keypoint = (original_court_kps[closest_keypoint_index * 2],
                                        original_court_kps[closest_keypoint_index * 2 + 1])

                    ball_mini_court_position = self.get_mini_court_coordinates(ball_pos,
                                                     closest_keypoint,
                                                     closest_keypoint_index,
                                                     max_player_height_in_pixel,
                                                     player_height.get(player_id,
                                                                       config.PLAYER_1_HEIGHT_METERS))

            output_player_bboxes.append(output_player_bboxes_dict)
            if ball_mini_court_position is not None:
                output_ball_bboxes.append({1: ball_mini_court_position})
            else:
                output_ball_bboxes.append({})
        return output_player_bboxes, output_ball_bboxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            if frame_num < len(positions) and positions[frame_num]:
                for _, position in positions[frame_num].items():
                    x, y = position
                    x = int(x)
                    y = int(y)
                    cv2.circle(frame, (x, y), 3, color, -1)

        return frames

    def draw_heatmap_on_mini_court(self, frame, current_positions):
        import numpy as np
        # Update heatmap layer with current positions
        for _, position in current_positions.items():
            x, y = int(position[0]), int(position[1])
            if 0 <= y < self.heatmap_layer.shape[0] and 0 <= x < self.heatmap_layer.shape[1]:
                self.heatmap_layer[y, x] += 1.0
                
        # Apply gaussian blur to create the heat effect
        heatmap_blurred = cv2.GaussianBlur(self.heatmap_layer, (31, 31), 0)
        
        # Normalize
        max_val = np.max(heatmap_blurred)
        if max_val > 0:
            heatmap_norm = heatmap_blurred / max_val
        else:
            heatmap_norm = heatmap_blurred
            
        # Colorize
        heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create a mask where heatmap is active (value > small threshold)
        mask = heatmap_norm > 0.05
        
        # Overlay only within the mini court area
        output_frame = frame.copy()
        
        # Crop mask to mini court boundaries
        x1, y1 = int(self.start_x), int(self.start_y)
        x2, y2 = int(self.end_x), int(self.end_y)
        
        court_mask = np.zeros_like(mask)
        court_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
        # Apply overlay
        alpha = 0.5
        output_frame[court_mask] = cv2.addWeighted(frame[court_mask], 1 - alpha, heatmap_color[court_mask], alpha, 0).squeeze()
        
        return output_frame