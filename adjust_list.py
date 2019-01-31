import sys
import re


def adjust(file_name):
    adjust_data = []
    pattern = re.compile(r'^ ')

    with open(file_name, 'r')as f:
        data = f.readlines()
    for d in data:
        m = re.match(pattern, d)
        if m:
            adjust_data[-1] = adjust_data[-1].strip() + d
        else:
            adjust_data.append(d)

    with open(file_name, 'w')as f:
        [f.write(d) for d in adjust_data]


def main():
    args = sys.argv
    dir_path = args[1]
    file_type = args[2]

    for i in range(1, 10 + 1):
        if file_type == 'label':
            file_name = dir_path + 'model_epoch_{}.label'.format(i)
        elif file_type == 'align':
            file_name = dir_path + 'model_epoch_{}.align'.format(i)
        adjust(file_name)


if __name__ == '__main__':
    main()