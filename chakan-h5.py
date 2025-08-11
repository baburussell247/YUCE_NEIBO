import h5py
import matplotlib.pyplot as plt

def print_h5_file_structure(file_name):
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    Attribute: {key} = {val}")

    with h5py.File(file_name, 'r') as f:
        f.visititems(print_attrs)

def print_h5_file_data(file_name):
    def print_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            print(f"    Shape: {obj.shape}")
            print(f"    Data type: {obj.dtype}")
            print(f"    Data: {obj[()]}\n")

    with h5py.File(file_name, 'r') as f:
        f.visititems(print_data)

def visualize_sample(file_name, sample_index):
    with h5py.File(file_name, 'r') as f:
        data = f['fields'][sample_index, 1, :, :]  # 选择指定的样本

        plt.figure(figsize=(10, 6))
        plt.imshow(data, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Sample {sample_index}')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.show()


file_name = 'D:\\pythonprojects\\fourcastnet21\datasets\\test\\2024_4_9.h5'  # 替换成你的 h5 文件名
sample_index = 1  # 替换成你想要可视化的样本索引
visualize_sample(file_name, sample_index)
print("File structure:")
print_h5_file_structure(file_name)
print("\nFile data:")
print_h5_file_data(file_name)
