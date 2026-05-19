class Config:
    SINGLE_LINE_WIDTH = 8.23
    DOUBLE_LINE_WIDTH = 10.97
    HALF_COURT_LINE_HEIGHT = 11.88
    SERVE_LINE_WIDTH = 6.4
    DOUBLE_ALLY_DIFFERENCE = 1.37
    NO_MANS_LAND_HEIGHT = 5.48

    PLAYER_1_HEIGHT_METERS = 1.88
    PLAYER_2_HEIGHT_METERS = 1.91


    #Mini court Config
    lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3),
            (12, 13)
        ]

    # Tracking Config
    PLAYER_CONFIDENCE = 0.2
    BALL_CONFIDENCE = 0.25
    MINIMUM_CHANGE_FRAMES_FOR_HIT = 25
    ROLLING_WINDOW_SIZE = 5