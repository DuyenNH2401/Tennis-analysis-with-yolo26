from utils.bbox_utils import measure_point_distance

class Analytics:
    def __init__(self, fps, mini_court):
        self.fps = fps
        self.mini_court = mini_court
        self.player_distances = {} # {player_id: distance_in_meters}
        self.player_stats_data = [] # List of dicts for each frame
        self.ball_speed_data = [] # List of speeds for each frame
        self.player_ids = set()

    def compute_analytics(self, player_mini_court_detection, ball_mini_court_detection, ball_shot_frames):
        num_frames = len(player_mini_court_detection)
        self.player_stats_data = [{} for _ in range(num_frames)]
        
        # 1. Player Distances
        # Find unique player IDs
        for frame_dict in player_mini_court_detection:
            for pid in frame_dict.keys():
                self.player_ids.add(pid)
                self.player_distances[pid] = 0.0

        for i in range(1, num_frames):
            for pid in self.player_ids:
                if pid in player_mini_court_detection[i] and pid in player_mini_court_detection[i-1]:
                    p1 = player_mini_court_detection[i-1][pid]
                    p2 = player_mini_court_detection[i][pid]
                    dist_pixels = measure_point_distance(p1, p2)
                    dist_meters = self.mini_court.get_meters_from_mini_court_pixels(dist_pixels)
                    self.player_distances[pid] += dist_meters
                
                # Keep track of distance up to frame i
                self.player_stats_data[i][pid] = {'distance': self.player_distances[pid]}
        
        # Initialize frame 0
        for pid in self.player_ids:
            self.player_stats_data[0][pid] = {'distance': 0.0}

        # 2. Ball Speed
        self.ball_speed_data = [0.0] * num_frames
        
        for i in range(len(ball_shot_frames) - 1):
            start_frame = ball_shot_frames[i]
            end_frame = ball_shot_frames[i+1]
            
            if start_frame < num_frames and end_frame < num_frames and start_frame != end_frame:
                if 1 in ball_mini_court_detection[start_frame] and 1 in ball_mini_court_detection[end_frame]:
                    p1 = ball_mini_court_detection[start_frame][1]
                    p2 = ball_mini_court_detection[end_frame][1]
                    dist_pixels = measure_point_distance(p1, p2)
                    dist_meters = self.mini_court.get_meters_from_mini_court_pixels(dist_pixels)
                    time_sec = (end_frame - start_frame) / self.fps
                    
                    if time_sec > 0:
                        # Convert to km/h
                        # Multiplying by a factor (e.g., 1.2) to estimate peak initial speed from the average flight speed
                        speed_kmh = (dist_meters / time_sec) * 3.6 * 1.2 
                        
                        # Apply this speed until the next shot
                        for frame_idx in range(start_frame, end_frame):
                            self.ball_speed_data[frame_idx] = speed_kmh

    def get_frame_analytics(self, frame_num):
        if frame_num < len(self.player_stats_data):
            return self.player_stats_data[frame_num], self.ball_speed_data[frame_num]
        return {}, 0.0
