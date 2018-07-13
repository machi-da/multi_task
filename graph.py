import sys
import matplotlib.pyplot as plt
import pandas as pd

def main():
    args = sys.argv
    file_name = args[1]
    df = pd.read_csv(file_name, index_col='epoch')
    df.plot(xticks=[i for i in range(1, 20+1)])
    plt.savefig(file_name + '.png')


main()