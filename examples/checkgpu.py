import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())

# 输出当前CUDA设备数量
print(torch.cuda.device_count())

# 输出当前CUDA设备的名称
print(torch.cuda.get_device_name(0))  # 如果有多个设备，可以更改索引

# 输出当前PyTorch使用的CUDA版本
print(torch.version.cuda)



