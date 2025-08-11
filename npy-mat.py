import numpy as np
import scipy.io

def npy_to_mat(npy_file, mat_file):
    """
    将npy文件转换为mat文件。

    参数：
    npy_file: str, 输入的npy文件路径。
    mat_file: str, 输出的mat文件路径。
    """
    # 读取npy文件
    data = np.load(npy_file, allow_pickle=True)

    # 保存为mat文件
    scipy.io.savemat(mat_file, {'data': data})


# 示例用法
npy_file = 'D:\\pythonprojects\\fourcastnet21\\outputs_fourcastnet_finetune\\result\\2024_4_9\\48\\true_data_U50.npy'  # 输入的npy文件路径
mat_file = 'D:\\pythonprojects\\fourcastnet21\\outputs_fourcastnet_finetune\\result\\2024_4_9\\48\\true_data_U50.mat'  # 输出的mat文件路径
npy_to_mat(npy_file, mat_file)
