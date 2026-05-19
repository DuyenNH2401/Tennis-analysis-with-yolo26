import cv2

from tracker import BallTracker
from utils.video_utils import read_video, save_video
from tracker.player_tracker import PlayerTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

def main():
    input_video_path = './input_videos/input_video_10s.mp4'
    output_video_path = './output_videos/output_video_2.mp4'
    video_frames, fps, size = read_video(input_video_path)

    #Player Detection
    player_tracker = PlayerTracker(model_path='yolo26l.pt')
    player_detect = player_tracker.detect_frames(video_frames,
                                                 read_from_stubs=True,
                                                 stubs_path='tracker_stubs/player_detect.pkl',
                                                 conf=0.2)
    #Ball Detection
    ball_tracker = BallTracker(model_path='models/yolo26m_best.pt')
    ball_detect = ball_tracker.detect_frames(video_frames,
                                             read_from_stubs=True,
                                             stubs_path='tracker_stubs/ball_detect.pkl',
                                             conf=0.25)
    ball_detect = ball_tracker.interpolate_ball_positions(ball_detect)
    #Court Detection
    court_model_path = 'models/best_keypoints_model.pt'
    court_line_detect = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detect.predict(video_frames[0])

    #Choose Player
    player_detect = player_tracker.choose_and_filter(court_keypoints, player_detect)

    #Mini court
    mini_court = MiniCourt(video_frames[0])

    #Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detect)
    print(ball_shot_frames)

    #Convert positions to mini court positions
    player_mini_court_detection, ball_mini_court_detection = mini_court.convert_bboxes_to_mini_court_coordinates(
                                                                    player_detect, ball_detect, court_keypoints)

    #Draw Output
    #Draw Player Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detect)
    #Draw Ball Box
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detect)
    #Draw Court Box
    output_video_frames = court_line_detect.draw_on_frames(output_video_frames)

    #Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,
                                                               player_mini_court_detection)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,
                                                               ball_mini_court_detection, color=(0, 255, 255))


    # #Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    save_video(output_video_frames, output_video_path, fps=fps, size=size)

if __name__ == '__main__':
    main()