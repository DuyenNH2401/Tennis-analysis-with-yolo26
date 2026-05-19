from ultralytics import YOLO
import cv2
import pickle
from utils.bbox_utils import get_center_bbox, measure_distance, get_court_center

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter(self, court_keypoints, player_detections):
        vote_frames = player_detections[:20]
        chosen_player = self._choose_player(court_keypoints, vote_frames)
        filtered_player_detections = []
        for player_dict in player_detections:
            filter_player_dict = {track_id:bbox
                                  for track_id, bbox in player_dict.items()
                                  if track_id in chosen_player}
            filtered_player_detections.append(filter_player_dict)
        return filtered_player_detections

    @staticmethod
    def _choose_player(court_keypoints, player_detections_frames):
        from collections import defaultdict
        player_votes = defaultdict(list)
        player_y_positions = defaultdict(list)
        court_center = get_court_center(court_keypoints)
        
        for player_dict in player_detections_frames:
            for track_id, bbox in player_dict.items():
                player_center = get_center_bbox(bbox)
                foot_y = bbox[3] # ymax is the foot position
                
                # Euclidean distance to court center
                distance_to_center = measure_distance(player_center, court_center)
                player_votes[track_id].append(distance_to_center)
                player_y_positions[track_id].append(foot_y)

        top_half_candidates = []
        bottom_half_candidates = []
        
        for track_id, dist_list in player_votes.items():
            avg_dist = sum(dist_list) / len(dist_list)
            avg_y = sum(player_y_positions[track_id]) / len(player_y_positions[track_id])
            
            # Only consider players that appear in at least 50% of the voting frames
            if len(dist_list) >= max(1, len(player_detections_frames) // 2):
                if avg_y < court_center[1]:
                    top_half_candidates.append((track_id, avg_dist))
                else:
                    bottom_half_candidates.append((track_id, avg_dist))
        
        top_half_candidates.sort(key=lambda x: x[1])
        bottom_half_candidates.sort(key=lambda x: x[1])
        
        chosen_player = []
        if top_half_candidates:
            chosen_player.append(top_half_candidates[0][0])
        if bottom_half_candidates:
            chosen_player.append(bottom_half_candidates[0][0])
            
        # Fallback if somehow we don't have exactly 2 players (e.g., both in same half)
        if len(chosen_player) < 2:
            distances = [(track_id, sum(d)/len(d)) for track_id, d in player_votes.items()]
            distances.sort(key=lambda x: x[1])
            chosen_player = [distances[0][0], distances[1][0]] if len(distances) >= 2 else []

        return chosen_player

    def detect_frames(self,
                      frames: list,
                      read_from_stubs=False,
                      stubs_path=None,
                      conf = 0.2):
        player_detect = []

        if read_from_stubs and stubs_path is not None:
            with open(stubs_path, 'rb') as f:
                player_detect = pickle.load(f)
            return player_detect

        for frame in frames:
            player_dict = self._detect_frame(frame, conf)
            player_detect.append(player_dict)

        if stubs_path is not None:
            import os
            os.makedirs(os.path.dirname(stubs_path), exist_ok=True)
            with open(stubs_path, 'wb') as f:
                pickle.dump(player_detect, f)

        return player_detect

    def _detect_frame(self, frame, conf):
        results = self.model.track(frame, persist=True, conf=conf)[0]

        id_name_dict = results.names

        player_dict = {}

        for box in results.boxes:
            if box.id is not None:
                track_id = box.id.int().tolist()[0]
                bbox = box.xyxy.int().tolist()[0]
                obj_class_id = box.cls.tolist()[0]
                obj_class_name = id_name_dict[obj_class_id]
                if obj_class_name == 'person':
                    player_dict[track_id] = bbox

        return player_dict
