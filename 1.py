import cv2

def slow_down_video(input_file, output_file, slowdown_factor):
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    new_fps = fps / slowdown_factor  # Reduce the FPS to slow down the playback
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, new_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video slowed down successfully to {new_fps} FPS.")

# Slow down the video by 4x to make it 20 minutes long
slow_down_video('input.mp4', 'output_slowed.mp4', slowdown_factor=4)