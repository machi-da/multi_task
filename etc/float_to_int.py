import sys
from tqdm import tqdm

args = sys.argv

with open(args[1], 'r')as f:
    data = f.readlines()

res = []
for d in tqdm(data):
    d = d.strip().split('\t')
    score = [float(s) for s in d[0].split(',')]
    index = score.index(max(score)) + 1
    res.append('{}\t{}\n'.format(index, d[1]))

with open(args[1] + '.int', 'w')as f:
    [f.write(r) for r in res]