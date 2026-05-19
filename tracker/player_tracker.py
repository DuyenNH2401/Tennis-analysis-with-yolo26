from ultralytics import YOLO
import cv2
import pickle
from utils.bbox_utils import get_center_bbox, measure_distance

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player =self._choose_player(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filter_player_dict = {track_id:bbox
                                  for track_id, bbox in player_dict.items()
                                  if track_id in chosen_player}
            filtered_player_detections.append(filter_player_dict)
        return filtered_player_detections

    @staticmethod
    def _choose_player(court_keypoints, player_dict):

        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_bbox(bbox)
            # Tính khoảng cách theo trục X từ người đó đến trục giữa sân
            distance_to_center_x = measure_distance(player_center, court_keypoints)

            distances.append((track_id, distance_to_center_x))
        distances.sort(key=lambda x: x[1])
        chosen_player = [distances[0][0], distances[1][0]]

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

    @staticmethod
    def draw_bboxes(frames, detection):
        output_frames = []
        for frame, player_dict in zip(frames, detection):
            # Draw bboxes
            for track_id, bbox in player_dict.items():
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, f"Player Id: {track_id}", (int(xmin), int(ymin)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            output_frames.append(frame)

        return output_frames