import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('baseline_score_file')
    parser.add_argument('compared_score_file')
    args = parser.parse_args()
    return args


def main(baseline_score_file, compared_score_file):
    with open(baseline_score_file, 'r')as f:
        baseline = f.readlines()
    with open(compared_score_file, 'r')as f:
        compared = f.readlines()

    correct_symbol = 'T'

    counter = 0
    better = 0
    dic = {'TT': 0, 'TF': 0, 'FT': 0, 'FF': 0}
    for b, c in zip(baseline, compared):
        b = b.strip().split('\t')[1]
        c = c.strip().split('\t')[1]

        if b == correct_symbol:
            if c == correct_symbol:
                dic['TT'] += 1
            else:
                dic['TF'] += 1
        else:
            if c == correct_symbol:
                dic['FT'] += 1
            else:
                dic['FF'] += 1

        if b != c:
            counter += 1
            if c == correct_symbol:
                better += 1

    print(dic)
    print('=BINOMDIST({}, {}, 0.5, FALSE)'.format(better, counter))


def make_ishigaki_score():
    result = []
    count = 1
    for i in range(1, 6):
        with open('/home/lr/ishigaki/yahoo/emnlp_script/machida/proposed_{}.txt'.format(i), 'r')as f:
            data = f.readlines()
        for d in data:
            result.append('{}\t{}'.format(count, d))
            count += 1

    with open('s_res.csv', 'w')as f:
        [f.write(r) for r in result]


def convert_s_res(file_name):
    result = []
    with open(file_name, 'r')as f:
        data = f.readlines()
    count = 1
    for d in data[1].strip().split(',')[2:]:
        if d == '1':
            result.append('{}\tT\n'.format(count))
        else:
            result.append('{}\tF\n'.format(count))
        count += 1
    with open(file_name + '.convert', 'w')as f:
        [f.write(r) for r in result]


if __name__ == '__main__':
    args = parse_args()
    # baseline_score_file = args.baseline_score_file
    baseline_score_file = '/home/lr/machida/yahoo/evaluate/s_res.csv'
    compared_score_file = args.compared_score_file

    main(baseline_score_file, compared_score_file)
    # convert_s_res(compared_score_file)

    # make_ishigaki_score()