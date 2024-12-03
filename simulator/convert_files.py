"""
Converting data files (npy) to csv files (csv) to be read in C++.
"""

import os
import numpy as np

def convert_files(input_path: str, save_path: str) -> bool:
    """
    Convert data files (npy) to csv files (csv) to be read in C++.

    Args:
    - path (str): path to the directory containing the data files.
    - save_path (str): path to save the converted files.

    Returns:
    - bool: True if the conversion is successful, False otherwise.
    """

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return False

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(input_path)
    for file in files:
        if file.endswith('.npy'):
            print(f"Converting {file} to csv.")
            data = np.load(os.path.join(input_path, file))
            # file_name = file.split('.')[0]
            # save_file = os.path.join(save_path, file_name + '.csv')
            # np.savetxt(save_file, data, delimiter=',')
            print(f"Size of {file}: {data.shape}")

    return True

if __name__ == '__main__':
    PATH = 'data'
    SAVE_PATH = 'data_txt'
    convert_files(PATH, SAVE_PATH)
    