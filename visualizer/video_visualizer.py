import cv2

class VideoVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def draw_player_bboxes(frames, detection):
        output_frames = []
        for frame, player_dict in zip(frames, detection):
            for track_id, bbox in player_dict.items():
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, f"Player Id: {track_id}", (int(xmin), int(ymin)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            output_frames.append(frame)
        return output_frames

    @staticmethod
    def draw_ball_bboxes(frames, detection):
        output_frames = []
        for frame, ball_dict in zip(frames, detection):
            for track_id, bbox in ball_dict.items():
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 225, 0), 2)
                cv2.putText(frame, f"Ball Id: {track_id}", (int(xmin), int(ymin)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 0), 2, cv2.LINE_AA)
            output_frames.append(frame)
        return output_frames

    @staticmethod
    def draw_court_keypoints(frames, keypoints):
        output_frames = []
        for frame in frames:
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i + 1])
                cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            output_frames.append(frame)
        return output_frames

    @staticmethod
    def draw_frame_numbers(frames):
        for i, frame in enumerate(frames):
            cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        return frames
