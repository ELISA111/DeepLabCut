import cv2
import os


def video_to_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 初始化帧计数器
    frame_count = 0

    # 逐帧读取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 构建每个帧的文件名
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')

        # 保存帧为图像文件
        cv2.imwrite(frame_filename, frame)

        # 更新帧计数器
        frame_count += 1

    # 释放视频对象
    cap.release()

    print(f"视频分解为 {frame_count} 帧，保存在 '{output_folder}' 文件夹中。")


# 使用示例
video_path = '/home/elisa/Desktop/Antenna-Elisa-2024-06-13 (copy)/videos/train_2DLC_resnet50_AntennaJun13shuffle1_100000_full.mp4'
output_folder = '/home/elisa/Desktop/Antenna-Elisa-2024-06-13 (copy)/videos/train_2DLC_resnet50_AntennaJun13shuffle1_100000_full'
video_to_frames(video_path, output_folder)
