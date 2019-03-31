from tqdm import tqdm
'''
1
ランサーズ用のチェックボックスhtmlファイル，質問事例htmlファイル作成
'''

def main():
    # que_num = {
    #     '02': 740,
    #     '03': 500,
    #     '04': 310,
    #     '05': 190,
    #     '06': 120,
    #     '07': 70,
    #     '08': 40,
    #     '09': 20,
    #     '10': 10
    # }
    que_num = {
        '02': 800,
        '03': 600,
        '04': 400,
        '05': 300,
        '06': 200,
        '07': 100,
        '08': 100,
        '09': 100,
        '10': 100
    }

    # 50問ごとのサンプルを作成
    dic_que50 = {
        1: {'set': 20, 'bunsu': [18, 13, 7, 5, 3, 2, 1, 1, 0]},
        2: {'set': 15, 'bunsu': [20, 11, 8, 5, 3, 2, 1, 0, 0]},
        3: {'set': 5, 'bunsu': [16, 15, 10, 3, 3, 0, 1, 0, 2]},
        4: {'set': 1, 'bunsu': [0, 5, 10, 10, 5, 5, 5, 5, 5]} # test用
    }

    # チェックボックスhtmlファイル
    for k, v in dic_que50.items():
        res = []
        count = 1
        total = sum(v['bunsu'])
        for sent_num, num in enumerate(v['bunsu'], start=2):
            for _ in range(num):
                lit = ['<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />',
                       '<p>',
                       '<span style="color:#0000FF"><strong>{}/{}.</strong></span>&nbsp;質問が含まれた文章に対し、要約として適切な文を1文選択してください。</p>'.format(count, total)
                       ]
                for i in range(1, sent_num + 1):
                    lit.append('<p>')
                    lit.append('<input name="q{}" type="radio" value="{}" />文番号:({})</p>'.format(count, i, i))
                else:
                    lit.append('<p>')
                    lit.append('<input name="q{}" type="radio" value="{}" />[1文では要約できない]</p>'.format(count, -1))
                res.extend(lit)
                count += 1
        with open('html/init_{}.html'.format(k), 'w')as f:
            with open('start.txt', 'r')as ff:
                data = ff.readlines()
            [f.write(d) for d in data]
            [f.write(r + '\n') for r in res]
            with open('end.txt', 'r')as ff:
                data = ff.readlines()
            [f.write(d) for d in data]

    # 質問事例htmlファイル
    que_dic = {}
    ans_dic = {}
    for k, v in que_num.items():
        que_list = []
        ans_list = []

        with open('q{}.txt'.format(k), 'r')as f:
            que = f.readlines()[:v + 1]
        with open('a{}.txt'.format(k), 'r')as f:
            ans = f.readlines()[:v + 1]

        for q, a in zip(que, ans):
            q = q.strip().split('\t')
            a = a.strip()
            que_list.append(q)
            ans_list.append(a)
        que_dic[int(k)] = que_list
        ans_dic[int(k)] = ans_list

    qa_data = []
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    file_count = 1
    for k, v in dic_que50.items():
        for _ in tqdm(range(v['set'])):
            res = ['<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />']
            count = 1
            total = sum(v['bunsu'])
            for sent_num, num in enumerate(v['bunsu'], start=2):
                for _ in range(num):
                    qa_data.append(['\t'.join(que_dic[sent_num][counter[sent_num-2]]), ans_dic[sent_num][counter[sent_num-2]]])
                    lit = ['<p>',
                           '<span style="color:#0000FF"><strong>{}/{}.</strong></span>&nbsp;質問が含まれた文章に対し、要約として適切な文を1文選択してください。</p>'.format(count, total),
                           '<p>',
                           '[文章]<br />',
                           ' '.join(que_dic[sent_num][counter[sent_num-2]]) + '</p>',
                           '<ul style="list-style:none;">'
                           ]
                    for i, sentence in enumerate(que_dic[sent_num][counter[sent_num-2]], start=1):
                        lit.append('<li>({}) {}</li>'.format(i, sentence))
                    else:
                        lit.append('<li>({}) [1文では要約できない]</li>'.format(i + 1))
                    lit.append('</ul>')
                    lit.append('</p>')
                    lit.append('<br />')

                    counter[sent_num-2] += 1
                    count += 1
                    res.extend(lit)

            if file_count < 10:
                file_count_str = '0{}'.format(file_count)
            else:
                file_count_str = '{}'.format(file_count)

            with open('html/{}/q_{}.html'.format(k, file_count_str), 'w')as f:
                [f.write(r + '\n') for r in res]
            file_count += 1

    # csv_merge.pyで使用
    with open('html/que_no.txt', 'w')as f:
        [f.write('{}\t{}\n'.format(i, d[0])) for i, d in enumerate(qa_data, start=1)]
    with open('html/ans_no.txt', 'w')as f:
        [f.write('{}\t{}\n'.format(i, d[1])) for i, d in enumerate(qa_data, start=1)]


if __name__ == '__main__':
    main()