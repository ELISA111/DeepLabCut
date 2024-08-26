from moviepy.editor import VideoFileClip

def crop_video(input_path, output_path, left_top, right_bottom):
    """
    Crop a video based on the given coordinates.

    :param input_path: Path to the input video.
    :param output_path: Path to save the cropped video.
    :param left_top: Tuple of (x1, y1) coordinates for the top left corner.
    :param right_bottom: Tuple of (x2, y2) coordinates for the bottom right corner.
    """
    # Load the video
    video = VideoFileClip(input_path)

    # Extract the coordinates
    x1, y1 = left_top
    x2, y2 = right_bottom

    # Crop the video
    cropped_video = video.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    # Write the result to a file
    cropped_video.write_videofile(output_path, codec='libx264')

# Example usage
input_video_path = "/home/elisa/Documents/bees data/train3_1.mp4"
output_video_path = "/home/elisa/Documents/bees data/train3_1_cropped.mp4"
left_top_coords = (480, 270)
right_bottom_coords = (1440, 810)

crop_video(input_video_path, output_video_path, left_top_coords, right_bottom_coords)
