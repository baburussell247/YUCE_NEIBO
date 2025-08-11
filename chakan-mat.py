import h5py


def print_mat_file_info_hdf5(file_path):
    # Open the MAT file using h5py
    with h5py.File(file_path, 'r') as mat_data:
        print("Variables in the MAT file:")
        # List all keys (variable names)
        print(list(mat_data.keys()))

        # Iterate through the variables
        for var_name in mat_data.keys():
            var_data = mat_data[var_name]
            print(f"\nVariable Name: {var_name}")
            print(f"Shape: {var_data.shape}")
            print(f"Data Type: {var_data.dtype}")
            print("Data:")
            print(var_data[...])  # Extract and print the data


# Replace with the path to your MAT file
file_path = 'D:\\Desktop\\南海\\nanhai-1h\\50mT_U\\test1\\20240409.mat'
print_mat_file_info_hdf5(file_path)
