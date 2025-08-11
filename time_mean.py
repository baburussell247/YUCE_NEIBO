import h5py
import numpy as np

def calculate_and_save_time_mean(h5_file_path, output_npy_path):
    # 读取 H5 文件
    with h5py.File(h5_file_path, 'r') as f:
        # 假设数据存储在 "fields" 数据集中，您可以根据实际情况修改
        data = f['fields'][:]

    # 计算时间均值，保持变量维度
    time_mean = np.mean(data, axis=0)  # (2, 2208, 624)

    # 调整形状为 (1, 2, 2208, 624)
    time_mean = np.expand_dims(time_mean, axis=0)

    # 保存为 npy 文件
    np.save(output_npy_path, time_mean)

    return time_mean

# 使用函数计算并保存 time_mean
h5_file_path = 'D:\\pythonprojects\\fourcastnet21\\datasets\\train\\2023.h5'  # 将此路径替换为您的 H5 文件路径
output_npy_path = 'stats/time_mean.npy'  # 将此路径替换为您希望保存的 .npy 文件路径
time_mean = calculate_and_save_time_mean(h5_file_path, output_npy_path)

# 打印 time_mean 的形状和前几个元素
print('time_mean shape:', time_mean.shape)
# 打印 time_mean 的所有数据
print('time_mean content:')
print(time_mean)
