import h5py
import numpy as np

def save_to_npy(value, file_path):
    try:
        # 将值存储到 npy 文件中
        np.save(file_path, np.array(value))
        print(f"Value saved to {file_path} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_mean_std_h5(file_path, mean_file_path, std_file_path):
    try:
        # 打开 H5 文件
        with h5py.File(file_path, 'r') as f:
            # 读取数据
            data = f['fields'][:]

        # 计算每个变量的非零值均值和标准差
        means = []
        stds = []

        for i in range(data.shape[1]):  # 遍历变量的维度
            non_zero_data = data[:, i, :, :][data[:, i, :, :] != 0]
            mean = np.mean(non_zero_data)
            std = np.std(non_zero_data)
            means.append(mean)
            stds.append(std)

        # 将均值和标准差调整为 (1, 2, 1, 1) 的格式
        mean_array = np.array(means).reshape(1, 2, 1, 1)
        std_array = np.array(stds).reshape(1, 2, 1, 1)

        # 存储均值和标准差到两个 npy 文件中
        save_to_npy(mean_array, mean_file_path)
        save_to_npy(std_array, std_file_path)

        return mean_array, std_array

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# H5 文件路径
file_path = 'D:\\pythonprojects\\fourcastnet21\\datasets\\train\\2023.h5'
# 均值和标准差的 npy 文件路径
mean_file_path = 'stats/mean.npy'
std_file_path = 'stats/std.npy'

# 计算并打印结果，并保存到 npy 文件中
mean, std = calculate_mean_std_h5(file_path, mean_file_path, std_file_path)
if mean is not None and std is not None:
    print(f"Mean of non-zero values: {mean}")
    print(f"Standard deviation of non-zero values: {std}")
