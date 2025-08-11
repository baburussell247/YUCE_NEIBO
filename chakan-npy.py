import numpy as np

def inspect_npy_file(file_path: str, show_data: bool = False, max_elements: int = 10):
    """
    Inspect the structure and content of a .npy file.

    Parameters:
    - file_path: str, path to the .npy file.
    - show_data: bool, whether to print the data content. Default is False.
    - max_elements: int, maximum number of elements to print if show_data is True. Default is 10.
    """
    try:
        # 加载 .npy 文件，允许加载包含对象数组
        data = np.load(file_path, allow_pickle=True)
        print(f"Data type: {data.dtype}")
        print(f"Data shape: {data.shape}")

        if show_data:
            # Flatten the data to handle multi-dimensional arrays easily
            flat_data = data.flatten()
            print(f"Data content (first {max_elements} elements):")
            print(flat_data[:max_elements])

        # 如果数据量不大，打印整个数据
        if show_data and flat_data.size <= max_elements:
            print("Full data content:")
            print(data)

        # 打印第一个变量的所有数据
        first_variable_data = data[0]  # 获取第一个变量的数据
        print("Data of the first variable:")
        print(first_variable_data)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = 'D:\pythonprojects\\fourcastnet21\\outputs_fourcastnet_finetune\\result\\2024_4_9\\true_data_U50.npy'
inspect_npy_file(file_path, show_data=True, max_elements=20)
