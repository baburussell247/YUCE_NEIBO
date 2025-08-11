import os
import h5py
import numpy as np

def extract_temp_u_from_mat(mat_folder, output_h5_file):
    # 获取指定文件夹中的所有.mat文件，并按文件名排序
    mat_files = sorted([f for f in os.listdir(mat_folder) if f.endswith('.mat')])

    with h5py.File(output_h5_file, 'w') as h5f:
        data_list = []
        for mat_file in mat_files:
            # 构建完整的.mat文件路径
            mat_path = os.path.join(mat_folder, mat_file)

            # 读取MATLAB v7.3及以上版本的.mat文件
            with h5py.File(mat_path, 'r') as mat_data:
                if 'Temp' in mat_data and 'U' in mat_data:
                    # 读取Temp变量
                    temp = mat_data['Temp'][()]
                    # 转换数据类型为float32，并重置维度
                    temp = temp.astype(np.float32)
                    # 缩减网格数，从右往左去除最后6列
                    temp = temp[:, :, :1256]
                    # 翻转数据以匹配正确的显示顺序
                    temp = temp[:, ::-1, :]

                    # 读取U变量
                    u = mat_data['U'][()]
                    # 转换数据类型为float32，并重置维度
                    u = u.astype(np.float32)
                    # 缩减网格数，从右往左去除最后6列
                    u = u[:, :, :1256]
                    # 翻转数据以匹配正确的显示顺序
                    u = u[:, ::-1, :]

                    # 添加新轴到 Temp 和 U，以便合并它们
                    temp = temp[:, np.newaxis, :, :]
                    u = u[:, np.newaxis, :, :]

                    # 合并Temp和U数据到一个数组，形成 (时间, 2, 336, 1256) 的形状
                    combined_data = np.concatenate((temp, u), axis=1)

                    # 添加到列表中
                    data_list.append(combined_data)
                else:
                    print(f"{mat_file} does not contain both 'Temp' and 'U' variables.")

        # 将所有的Temp和U数据整合到一个数组中
        all_data = np.concatenate(data_list, axis=0)

        # 创建fields数据集
        h5f.create_dataset('fields', data=all_data)

# 使用示例
mat_folder = 'D:\\Desktop\\南海\\nanhai-1h\\50mT_U\\test1\\'  # 替换为你的.mat文件所在的文件夹路径
output_h5_file = 'datasets/test/2023_3_8.h5'  # 替换为你希望输出的HDF5文件名
extract_temp_u_from_mat(mat_folder, output_h5_file)
