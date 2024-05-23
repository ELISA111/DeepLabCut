import cv2
import os
import re

# Directory containing the frames
frames_dir = '/home/elisa/Documents/bees data/test'

# Output video file
output_video = '/home/elisa/Documents/bees data/output_video_test.mp4'

# Frame rate (frames per second)
frame_rate = 30

# Function to extract numerical value from frame filenames
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1

# Get the list of all frames in the directory
frames = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
frames.sort(key=numerical_sort)  # Sort frames numerically

# Read the first frame to get the width and height
frame_path = os.path.join(frames_dir, frames[0])
frame = cv2.imread(frame_path)
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 format
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop through all frames and write them to the video
for frame_name in frames:
    frame_path = os.path.join(frames_dir, frame_name)
    frame = cv2.imread(frame_path)
    video.write(frame)

# Release the VideoWriter object
video.release()

print(f"Video saved to {output_video}")
