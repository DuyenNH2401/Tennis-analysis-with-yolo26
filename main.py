import cv2
import argparse

from tracker import BallTracker
from utils.video_utils import get_video_properties, read_video_stream, save_video
from utils.draw_utils import draw_video_stream
from tracker.player_tracker import PlayerTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from utils.analytics_utils import Analytics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./input_videos/input_video_10s_2.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='./output_videos/output_video.mp4', help='Path to output video')
    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output
    
    fps, size = get_video_properties(input_video_path)
    
    # Read first frame for court detection
    cap = cv2.VideoCapture(input_video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Error reading video")
        return

    # Initialize Trackers
    player_tracker = PlayerTracker(model_path='yolo26l.pt')
    ball_tracker = BallTracker(model_path='models/yolo26m_best.pt')

    # Detect Players
    player_detect = player_tracker.detect_frames(read_video_stream(input_video_path), read_from_stubs=True, stubs_path='tracker_stubs/player_detect.pkl')

    # Detect Balls
    ball_detect = ball_tracker.detect_frames(read_video_stream(input_video_path), read_from_stubs=True, stubs_path='tracker_stubs/ball_detect.pkl')

    ball_detect = ball_tracker.interpolate_ball_positions(ball_detect)

    # Court Detection
    court_model_path = 'models/best_keypoints_model.pt'
    court_line_detect = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detect.predict(first_frame)

    # Choose Player
    player_detect = player_tracker.choose_and_filter(court_keypoints, player_detect)

    # Mini court
    mini_court = MiniCourt(first_frame)

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detect)

    # Convert positions to mini court positions
    player_mini_court_detection, ball_mini_court_detection = mini_court.convert_bboxes_to_mini_court_coordinates(
                                                                    player_detect, ball_detect, court_keypoints)

    # Analytics calculation
    analytics = Analytics(fps, mini_court)
    analytics.compute_analytics(player_mini_court_detection, ball_mini_court_detection, ball_shot_frames)

    # Draw Output
    video_stream = read_video_stream(input_video_path)
    output_stream = draw_video_stream(
        video_stream, player_detect, ball_detect,
        player_mini_court_detection, ball_mini_court_detection,
        court_keypoints, player_tracker, ball_tracker,
        court_line_detect, mini_court, analytics
    )
    
    save_video(output_stream, output_video_path, fps, size)

if __name__ == '__main__':
    main()