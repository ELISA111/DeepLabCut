import cv2
import numpy as np
import os
# from keras.models import load_model
#
# # 加载预训练的超分辨率模型（例如，ESRGAN模型）
# sr_model = load_model('path_to_pretrained_esrgan_model.h5')
#
#
# def enhance_image(image):
#     # 将图像转换为所需的格式
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # 归一化图像
#     image = image / 255.0
#     # 扩展维度以适应模型输入
#     image = np.expand_dims(image, axis=0)
#
#     # 应用超分辨率
#     enhanced_image = sr_model.predict(image)
#     # 去掉不必要的维度
#     enhanced_image = np.squeeze(enhanced_image, axis=0)
#     # 将图像从归一化状态还原为原始像素值
#     enhanced_image = (enhanced_image * 255.0).astype(np.uint8)
#     # 将图像转换回BGR格式
#     enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
#
#     return enhanced_image
#
#

#
# def sharpen_image(image):
#     # 定义拉普拉斯锐化卷积核
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#
#     # 应用卷积核进行锐化处理
#     sharpened = cv2.filter2D(image, -1, kernel)
#
#     return sharpened
#
#
# # 读取图像
# input_image_path = '/home/elisa/Documents/bees data/test_enhance/3000.png'
# output_image_path = '/home/elisa/Documents/bees data/test_enhance/3000_enhanced.png'
# image = cv2.imread(input_image_path)
# #
#
#
#
# # 检查是否成功读取图像
# if image is None:
#     print(f"Failed to read the image from {input_image_path}")
# else:
#     # 锐化图像
#     sharpened_image = sharpen_image(image)
#
#     # 保存锐化后的图像
#     cv2.imwrite(output_image_path, sharpened_image)
#     print(f"Sharpened image saved to {output_image_path}")
#


# # 检查是否成功读取图像
# if image is None:
#     print(f"Failed to read the image from {input_image_path}")
# else:
#     # 增强图像
#     enhanced_image = enhance_image(image)
#
#     # 保存增强后的图像
#     cv2.imwrite(output_image_path, enhanced_image)
#     print(f"Enhanced image saved to {output_image_path}")



#
#
# def sharpen_image(image):
#     # 定义拉普拉斯锐化卷积核
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#
#     # 应用卷积核进行锐化处理
#     sharpened = cv2.filter2D(image, -1, kernel)
#
#     return sharpened
#
#
# # 输入图像序列的路径和输出图像序列的路径
# input_image_folder = '/home/elisa/Documents/bees data/test'  # 替换为包含图像序列的文件夹路径
# output_image_folder = '/home/elisa/Documents/bees data/test_sharpened'  # 替换为保存输出图像序列的文件夹路径
#
# # 确保输出文件夹存在，如果不存在则创建
# if not os.path.exists(output_image_folder):
#     os.makedirs(output_image_folder)
#
# # 遍历输入图像序列文件夹中的每个图像
# for filename in os.listdir(input_image_folder):
#     if filename.endswith('.jpg') or filename.endswith('.png'):  # 假设图像格式为 JPG 或 PNG
#         # 读取图像
#         image_path = os.path.join(input_image_folder, filename)
#         image = cv2.imread(image_path)
#
#         if image is None:
#             print(f"Failed to read the image from {image_path}")
#         else:
#             # 锐化图像
#             sharpened_image = sharpen_image(image)
#
#             # 保存锐化后的图像
#             output_image_path = os.path.join(output_image_folder, filename)
#             cv2.imwrite(output_image_path, sharpened_image)
#             print(f"Sharpened image saved to {output_image_path}")


import cv2
import numpy as np
import os
import re
import torch
import torchvision.transforms as transforms

# Load the ESRGAN model from the local file
checkpoint = torch.load('/home/elisa/Documents/bees data/models/RRDB_PSNR_x4.pth', map_location=torch.device('cpu'))
esrgan_model = checkpoint['state_dict']
esrgan_model = {k.replace("module.", ""): v for k, v in esrgan_model.items()}  # Remove 'module.' prefix for DataParallel

# Define a function to preprocess and enhance the image using ESRGAN
def enhance_image(image):
    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to fit model input size
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    # Generate high-resolution image using ESRGAN
    with torch.no_grad():
        enhanced_image = esrgan_model(image).clamp(0.0, 1.0)

    # Convert the generated image back to numpy array
    enhanced_image = (enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

    return enhanced_image


# Define input and output folders
input_folder = '/home/elisa/Documents/bees data/test'  # Replace with the path to your input image folder
output_folder = '/home/elisa/Documents/bees data/test_enhanced'  # Replace with the path to save the enhanced images

# Ensure output folder exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Define numerical sorting function
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1


# Process each image in the input folder
for filename in sorted(os.listdir(input_folder), key=numerical_sort):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is not None:
            # Enhance the image using ESRGAN
            enhanced_image = enhance_image(image)

            # Save the enhanced image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, enhanced_image)
            print(f"Enhanced image saved to {output_path}")
        else:
            print(f"Failed to read the image from {image_path}")
