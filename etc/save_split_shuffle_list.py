import sys
import gridsearch


args = sys.argv
file_name = args[1]

with open(file_name, 'r')as f:
    data = f.readlines()

slice_size = len(data) // 5
split_data = gridsearch.shuffle_list(data)
