import h5py

# 读取 .mat 文件
with h5py.File('100m_T_1h_05_22.mat', 'r') as file:
    # 查看 .mat 文件中的所有数据集名称
    print("Datasets in the .mat file:")
    print(list(file.keys()))

    # 访问 'Temp' 数据集的数据
    temp_data = file['Temp']
    print("Shape of temp_data:", temp_data.shape)
    print("Data of temp_data:", temp_data[:])  # 打印数据集中的内容
