from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    @staticmethod
    def interpolate_ball_positions(ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_pos = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill()
        ball_positions = [{1: x} for x in df_ball_pos.to_numpy().tolist()]
        return ball_positions

    @staticmethod
    def get_ball_shot_frames(ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1,
                                                                                     center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (
                df_ball_positions['delta_y'].iloc[i] > 0 > df_ball_positions['delta_y'].iloc[i + 1])
            positive_position_change = (
                df_ball_positions['delta_y'].iloc[i] < 0 < df_ball_positions['delta_y'].iloc[i + 1])

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 > \
                                                               df_ball_positions['delta_y'].iloc[change_frame]
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 < \
                                                               df_ball_positions['delta_y'].iloc[change_frame]

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self,
                      frames: list,
                      read_from_stubs=False,
                      stubs_path=None,
                      conf = 0.2):
        ball_detect = []

        if read_from_stubs and stubs_path is not None:
            with open(stubs_path, 'rb') as f:
                ball_detect = pickle.load(f)
            return ball_detect

        for frame in frames:
            ball_dict = self._detect_frame(frame, conf)
            ball_detect.append(ball_dict)

        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(ball_detect, f)

        return ball_detect

    def _detect_frame(self, frame, conf):
        results = self.model.predict(frame, conf=conf)[0]
        ball_dict = {}

        if len(results.boxes) > 0:
            box = results.boxes[0]
            bbox = box.xyxy.int().tolist()[0]
            ball_dict[1] = bbox

        return ball_dict

    @staticmethod
    def draw_bboxes(frames, detection):
        output_frames = []
        for frame, ball_dict in zip(frames, detection):
            # Draw bboxes
            for track_id, bbox in ball_dict.items():
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 225, 0), 2)
                cv2.putText(frame, f"Ball Id: {track_id}", (int(xmin), int(ymin)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 0), 2, cv2.LINE_AA)
            output_frames.append(frame)

        return output_frames