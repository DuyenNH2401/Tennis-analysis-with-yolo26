import cv2

def draw_video_stream(video_stream, player_detect, ball_detect, 
                      player_mini_court_detection, ball_mini_court_detection,
                      court_keypoints, player_tracker, ball_tracker, 
                      court_line_detect, mini_court, analytics):
    """
    Generator that takes a stream of frames, applies all tracking and analytics drawings,
    and yields the drawn frames one by one.
    """
    for i, frame in enumerate(video_stream):
        if i >= len(player_detect):
            break

        current_player_detect = [player_detect[i]]
        current_ball_detect = [ball_detect[i]]
        current_player_mini = [player_mini_court_detection[i]]
        current_ball_mini = [ball_mini_court_detection[i]]

        # Draw main video components
        output_frame = player_tracker.draw_bboxes([frame], current_player_detect)[0]
        output_frame = ball_tracker.draw_bboxes([output_frame], current_ball_detect)[0]
        output_frame = court_line_detect.draw_on_frames([output_frame], court_keypoints)[0]

        # Draw mini court components
        output_frame = mini_court.draw_mini_court([output_frame])[0]
        output_frame = mini_court.draw_heatmap_on_mini_court(output_frame, current_player_mini[0])
        output_frame = mini_court.draw_points_on_mini_court([output_frame], current_player_mini)[0]
        output_frame = mini_court.draw_points_on_mini_court([output_frame], current_ball_mini, color=(0, 255, 255))[0]

        # Put frame number
        cv2.putText(output_frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw Analytics Dashboard
        dashboard_x = 10
        player_ids_list = list(analytics.player_ids)
        num_players = len(player_ids_list)
        dashboard_h = 90 + num_players * 30
        
        cv2.rectangle(output_frame, (dashboard_x, 50), (dashboard_x + 300, 50 + dashboard_h), (255,255,255), -1)
        cv2.putText(output_frame, "Analytics", (dashboard_x + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        frame_stats, b_speed = analytics.get_frame_analytics(i)
        
        y_offset = 115
        for pid in player_ids_list:
            p_dist = frame_stats.get(pid, {'distance': 0.0})['distance']
            cv2.putText(output_frame, f"P{pid} Dist: {p_dist:.1f} m", (dashboard_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            y_offset += 30
        
        if b_speed > 0:
            cv2.putText(output_frame, f"Shot Speed: {b_speed:.1f} km/h", (dashboard_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,150,0), 2)

        yield output_frame
