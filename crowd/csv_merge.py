import sys
from collections import defaultdict
'''
3
csv.resファイルと問題文をマージして評価用データを作成
csv.resファイルとque_no.txtとans_no.txtを入力する
'''


def main():
    args = sys.argv
    csv_result_file = args[1]
    que_file = args[2]
    ans_file = args[3]

    csv_result_list = []
    csv_result_dic = {}
    with open(csv_result_file, 'r')as f:
        data = f.readlines()[1:]
    for d in data:
        if d.strip() == '':
            continue
        d = d.strip().split(',')
        que_no = int(d[0])
        sent_no = int(d[1])
        csv_result_list.append(que_no)
        csv_result_dic[que_no] = sent_no

    sent_dic = defaultdict(int)
    q_res = []
    a_res = []

    with open(que_file, 'r')as f:
        q_data = f.readlines()
    with open(ans_file, 'r')as f:
        a_data = f.readlines()

    for q, a in zip(q_data, a_data):
        q = q.strip().split('\t')
        no = int(q[0])
        sentence = '\t'.join(q[1:])

        a = a.strip().split('\t')[1]

        if no in csv_result_list:
            sent_dic[len(q[1:])] += 1
            q_res.append('{}\t{}\n'.format(csv_result_dic[no], sentence))
            a_res.append('{}\n'.format(a))

    # 評価データ本体
    with open(csv_result_file + '.correct.que', 'w')as f:
        [f.write(r) for r in q_res]
    with open(csv_result_file + '.correct.ans', 'w')as f:
        [f.write(r) for r in a_res]

    print(sent_dic)


if __name__ == '__main__':
    main()