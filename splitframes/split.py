import cv2
import os
import numpy as np
# Load video
vid_list = ["01.avi", "02.avi", "03.avi", "04.avi", "05.avi", "06.avi", "07.avi", "08.avi", "09.avi", "10.avi"]
video_path_base = "/media/nilum/New_Volume/01.projects/Dataset-cubes/vids/"
video_output_path = "/media/nilum/New_Volume/01.projects/Dataset-cubes/raw-data/"

total_saved_count = 0


for vid in vid_list:
    video_path = os.path.join(video_path_base, vid)
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    saved_count = 0
    threshold = 30  # Adjust this for sensitivity

    vid_prefix = os.path.splitext(vid)[0]  # e.g., '01' from '01.avi'

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute absolute difference between current and previous frame
        diff = cv2.absdiff(prev_frame, gray_frame)
        non_zero_count = np.count_nonzero(diff)
        # Save frame if difference exceeds threshold
        if non_zero_count > threshold * frame.size / 100:
            frame_filename = os.path.join(video_output_path, f"{vid_prefix}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        prev_frame = gray_frame
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {saved_count} frames with scene changes in '{video_output_path}'")
    total_saved_count += saved_count

print(f"Total frames extracted from all videos: {total_saved_count}")
