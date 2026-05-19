import cv2

def read_video(video_path:str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)

def save_video(frames, video_path:str, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, size)

    for frame in frames:
        out.write(frame)
    out.release()
